"""
Day 4 — Ancillary layers: CHIRPS rainfall anomaly, ERA5-Land soil moisture and LST,
SRTM elevation/slope/TWI, ESA WorldCover cropland mask v1.

All outputs go to GEE Assets. Compute footprint is small — these are
pre-processed public datasets.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ee
from src.gee_utils import init_ee, load_aoi, export_image_to_asset

# --- Time windows -------------------------------------------------------------
BASELINE_START = "2015-01-01"
BASELINE_END = "2021-12-31"
SHOCK_START = "2022-06-01"
SHOCK_END = "2022-10-31"

ERA5_START = "2021-01-01"
ERA5_END = "2026-08-01"  # ERA5-Land is a few months behind real-time

# --- Asset paths --------------------------------------------------------------
PROJECT_ASSETS = "projects/flood-recovery-sindh/assets"
ASSET_RAINFALL = f"{PROJECT_ASSETS}/chirps_rainfall_anomaly_2022"
ASSET_TERRAIN = f"{PROJECT_ASSETS}/srtm_terrain"
ASSET_CROPLAND = f"{PROJECT_ASSETS}/esa_cropland_mask"
COLL_SOIL_M = f"{PROJECT_ASSETS}/era5_soil_moisture"
COLL_LST = f"{PROJECT_ASSETS}/era5_lst"


# --- Layer builders -----------------------------------------------------------
def chirps_rainfall_anomaly(aoi: ee.FeatureCollection) -> ee.Image:
    """
    2022 monsoon total rainfall minus the 2015-2021 monsoon mean.
    Positive = wetter than baseline (i.e. the flood signal).
    """
    chirps = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY").filterBounds(aoi.geometry())

    baseline_years = ee.List.sequence(2015, 2021)

    def year_total(y):
        y = ee.Number(y)
        start = ee.Date.fromYMD(y, 6, 1)
        end = ee.Date.fromYMD(y, 10, 31)
        return chirps.filterDate(start, end).sum().set("year", y)

    baseline_totals = ee.ImageCollection.fromImages(baseline_years.map(year_total))
    baseline_mean = baseline_totals.mean()

    shock_total = chirps.filterDate(SHOCK_START, SHOCK_END).sum()
    anomaly = (
        shock_total.subtract(baseline_mean).rename("rainfall_anomaly_mm").toFloat()
    )
    return anomaly


def srtm_terrain(aoi: ee.FeatureCollection) -> ee.Image:
    """Elevation, slope, and a topographic wetness index (TWI)."""
    dem = ee.Image("USGS/SRTMGL1_003").clip(aoi.geometry())
    slope = ee.Terrain.slope(dem)

    # TWI = ln(a / tan(slope)), where a is upslope contributing area.
    # A widely used approximation uses flow accumulation from HydroSHEDS.
    # For a fast first pass, use slope alone; add real TWI on Day 5 if needed.
    slope_rad = slope.multiply(3.14159265 / 180)
    twi_approx = slope_rad.tan().max(0.001).log().multiply(-1).rename("twi_approx")

    return (
        dem.rename("elevation_m")
        .addBands(slope.rename("slope_deg"))
        .addBands(twi_approx)
        .toFloat()
    )


def esa_cropland_mask(aoi: ee.FeatureCollection) -> ee.Image:
    """ESA WorldCover 2021 v200. Class 40 = cropland."""
    wc = ee.ImageCollection("ESA/WorldCover/v200").first().clip(aoi.geometry())
    return wc.eq(40).rename("cropland").toByte()


def era5_monthly_stack(
    aoi: ee.FeatureCollection, band: str, start: str, end: str, out_band: str
) -> ee.ImageCollection:
    """Monthly ERA5-Land aggregation for a given band."""
    coll = (
        ee.ImageCollection("ECMWF/ERA5_LAND/MONTHLY_AGGR")
        .filterBounds(aoi.geometry())
        .filterDate(start, end)
        .select(band)
    )

    def rename(img):
        return img.rename(out_band).copyProperties(img, ["system:time_start"])

    return coll.map(rename)


def month_iter(start: str, end: str):
    """Yield (year, month) tuples covering [start, end)."""
    s = ee.Date(start).getInfo()
    e = ee.Date(end).getInfo()
    from datetime import date

    s_d = date.fromtimestamp(s["value"] / 1000)
    e_d = date.fromtimestamp(e["value"] / 1000)
    y, m = s_d.year, s_d.month
    while (y, m) < (e_d.year, e_d.month):
        yield y, m
        m += 1
        if m > 12:
            m, y = 1, y + 1


# --- Batch export helpers -----------------------------------------------------
def export_single_image(
    image: ee.Image,
    description: str,
    asset_id: str,
    region: ee.Geometry,
    scale: int = 30,
) -> None:
    task = export_image_to_asset(
        image=image,
        description=description,
        asset_id=asset_id,
        region=region,
        scale=scale,
    )
    task.start()
    print(f"  submitted: {description} → {asset_id}  (task {task.id})")


def export_era5_collection(
    aoi: ee.FeatureCollection,
    coll: ee.ImageCollection,
    asset_coll: str,
    prefix: str,
    scale: int = 1000,
) -> None:
    """Loop over months and export each ERA5 image separately to the collection."""
    region = aoi.geometry().bounds()
    submitted = 0
    for y, m in month_iter(ERA5_START, ERA5_END):
        month_str = f"{y}-{m:02d}"
        desc = f"{prefix}_{month_str}"
        asset_id = f"{asset_coll}/{desc}"

        d0 = ee.Date.fromYMD(y, m, 1)
        d1 = d0.advance(1, "month")
        img = ee.Image(coll.filterDate(d0, d1).first())
        img = img.set("system:time_start", d0.millis()).set("month", month_str)

        task = export_image_to_asset(
            image=img,
            description=desc,
            asset_id=asset_id,
            region=region,
            scale=scale,
        )
        task.start()
        submitted += 1
        print(f"  [{submitted:3d}] {desc}")
        time.sleep(0.3)
    print(f"  Submitted {submitted} tasks to {asset_coll}\n")


# --- Main ---------------------------------------------------------------------
def main() -> None:
    init_ee()
    aoi = load_aoi()
    region = aoi.geometry().bounds()

    print("=== 1. CHIRPS rainfall anomaly (2022 vs 2015-2021 baseline) ===")
    export_single_image(
        chirps_rainfall_anomaly(aoi),
        description="chirps_rainfall_anomaly_2022",
        asset_id=ASSET_RAINFALL,
        region=region,
        scale=5000,
    )

    print("\n=== 2. SRTM terrain (elevation, slope, TWI approx) ===")
    export_single_image(
        srtm_terrain(aoi),
        description="srtm_terrain",
        asset_id=ASSET_TERRAIN,
        region=region,
        scale=30,
    )

    print("\n=== 3. ESA WorldCover cropland mask ===")
    export_single_image(
        esa_cropland_mask(aoi),
        description="esa_cropland_mask",
        asset_id=ASSET_CROPLAND,
        region=region,
        scale=10,
    )

    print("\n=== 4. ERA5-Land soil moisture (monthly) ===")
    sm_coll = era5_monthly_stack(
        aoi,
        band="volumetric_soil_water_layer_1",
        start=ERA5_START,
        end=ERA5_END,
        out_band="soil_moisture",
    )
    export_era5_collection(aoi, sm_coll, COLL_SOIL_M, prefix="era5_sm", scale=1000)

    print("=== 5. ERA5-Land land surface temperature (monthly) ===")
    lst_coll = era5_monthly_stack(
        aoi,
        band="skin_temperature",
        start=ERA5_START,
        end=ERA5_END,
        out_band="lst_kelvin",
    )
    export_era5_collection(aoi, lst_coll, COLL_LST, prefix="era5_lst", scale=1000)

    print("\nAll Day 4 exports submitted.")
    print(
        "Monitor at https://code.earthengine.google.com/tasks?project=flood-recovery-sindh"
    )


if __name__ == "__main__":
    main()
