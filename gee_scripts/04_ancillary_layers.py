"""
Day 4 — Ancillary layers: CHIRPS rainfall anomaly, ERA5-Land soil moisture and LST,
SRTM elevation/slope/TWI, ESA WorldCover cropland mask v1.

Fill in on Day 4; skeleton is here so you don't have to think about structure.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ee
from src.gee_utils import init_ee, load_aoi


def chirps_rainfall_anomaly(aoi: ee.FeatureCollection,
                            baseline: tuple[str, str] = ("2015-01-01", "2021-12-31"),
                            shock: tuple[str, str] = ("2022-06-01", "2022-10-31")) -> ee.Image:
    """
    2022 monsoon rainfall minus the multi-year monthly mean, summed over the shock window.
    Result is a single ee.Image of mm anomaly per pixel.
    """
    chirps = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY").filterBounds(aoi.geometry())
    baseline_mean = chirps.filterDate(*baseline).mean().multiply(30)  # daily → monthly approx
    shock_total = chirps.filterDate(*shock).sum()
    return shock_total.subtract(baseline_mean.multiply(5)).rename("rainfall_anomaly_mm")


def era5_soil_moisture_stack(aoi: ee.FeatureCollection,
                             start: str = "2021-01-01",
                             end: str = "2026-01-01") -> ee.ImageCollection:
    """Monthly ERA5-Land volumetric soil water layer 1 (0-7cm) across the study window."""
    return (ee.ImageCollection("ECMWF/ERA5_LAND/MONTHLY_AGGR")
            .filterBounds(aoi.geometry())
            .filterDate(start, end)
            .select("volumetric_soil_water_layer_1"))


def srtm_terrain(aoi: ee.FeatureCollection) -> ee.Image:
    """Elevation, slope, and a simple TWI-like index for the AOI."""
    dem = ee.Image("USGS/SRTMGL1_003").clip(aoi.geometry())
    slope = ee.Terrain.slope(dem)
    return dem.rename("elevation_m").addBands(slope.rename("slope_deg"))


def esa_worldcover_cropland(aoi: ee.FeatureCollection) -> ee.Image:
    """ESA WorldCover 2021 v200; class 40 = cropland."""
    wc = ee.ImageCollection("ESA/WorldCover/v200").first().clip(aoi.geometry())
    return wc.eq(40).rename("cropland_mask")


def main() -> None:
    init_ee()
    aoi = load_aoi()
    print("Ancillary layer builders ready — call them from a notebook or an export script.")
    print(f"AOI extent: {aoi.geometry().bounds().getInfo()['coordinates']}")


if __name__ == "__main__":
    main()
