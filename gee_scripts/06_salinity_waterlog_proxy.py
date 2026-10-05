"""
Day 5a — Salinity and waterlogging persistence proxies.

Both are "how long did the post-flood signature linger" metrics, computed
from the monthly S2 composites in our asset collection.

Design (per working plan Day 5 and the Option C discussion):
- NDWI persistence  = # months Nov 2022 → Oct 2023 where NDWI
  stayed anomalously high vs. the pre-flood baseline (indicates waterlogging)
- SBI persistence   = # months Nov 2022 → Oct 2023 where SBI
  stayed anomalously high vs. the pre-flood baseline (indicates salt-crust /
  bare-soil brightening — a salinity signature)

Baseline for both is the per-pixel mean over the same calendar months
in 2021 (one year before the flood, same seasonal position).

Outputs:
- projects/.../ndwi_persistence_months
- projects/.../sbi_persistence_months
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ee
from src.gee_utils import init_ee, load_aoi, export_image_to_asset

BASE = "projects/flood-recovery-sindh/assets"
S2_COLL = f"{BASE}/s2_monthly_composites"
ASSET_NDWI = f"{BASE}/ndwi_persistence_months"
ASSET_SBI = f"{BASE}/sbi_persistence_months"

# Post-flood window to measure persistence over.
POST_START_Y, POST_START_M = 2022, 11
POST_END_Y, POST_END_M = 2023, 11  # exclusive: 12 months

# Baseline window: same months, one year before the flood.
BASELINE_START_Y, BASELINE_START_M = 2021, 11
BASELINE_END_Y, BASELINE_END_M = 2022, 11  # exclusive

# Anomaly thresholds — a pixel is "persistent" in a given month if its
# anomaly vs. baseline exceeds this. Tunable; sensitivity-test on Day 14.
NDWI_ANOMALY_THRESHOLD = 0.1  # NDWI units (dimensionless)
SBI_ANOMALY_THRESHOLD = 0.05  # SBI is sqrt(RED²+NIR²), reflectance-scale


def months_between(sy, sm, ey, em):
    """Yield (year, month) tuples for [start, end)."""
    y, m = sy, sm
    while (y, m) < (ey, em):
        yield y, m
        m += 1
        if m > 12:
            m, y = 1, y + 1


def composite_for(y: int, m: int) -> ee.Image:
    """Load the S2 monthly composite asset for a given year-month."""
    return ee.Image(f"{S2_COLL}/s2_composite_{y}-{m:02d}")


def add_ndwi_sbi_from_composite(img: ee.Image) -> ee.Image:
    """
    Our monthly composites only have NDVI and EVI bands; we didn't export
    raw reflectance. So we approximate from NDVI + EVI... actually we can't.
    Instead, re-derive NDWI and SBI from the underlying S2 SR directly for
    the needed months. See build_month_index() below.
    """
    raise NotImplementedError("Use build_month_index() instead.")


def build_month_index(aoi: ee.FeatureCollection, year: int, month: int) -> ee.Image:
    """
    Build NDWI and SBI monthly composites directly from S2 SR for one month,
    using the same cloud masking as Day 2 but computing different indices.

    Returns an ee.Image with 2 bands: NDWI, SBI.
    """
    d0 = ee.Date.fromYMD(year, month, 1)
    d1 = d0.advance(1, "month")

    s2 = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(aoi.geometry())
        .filterDate(d0, d1)
    )
    prob = (
        ee.ImageCollection("COPERNICUS/S2_CLOUD_PROBABILITY")
        .filterBounds(aoi.geometry())
        .filterDate(d0, d1)
    )

    joined = ee.ImageCollection(
        ee.Join.saveFirst("cloud_prob").apply(
            primary=s2,
            secondary=prob,
            condition=ee.Filter.equals(
                leftField="system:index", rightField="system:index"
            ),
        )
    )

    def mask_and_index(img):
        cloud_prob = ee.Image(img.get("cloud_prob")).select("probability")
        is_cloud = cloud_prob.gt(40)
        masked = img.updateMask(is_cloud.Not())

        # Scale SR to reflectance.
        green = masked.select("B3").divide(10000)
        red = masked.select("B4").divide(10000)
        nir = masked.select("B8").divide(10000)

        ndwi = green.subtract(nir).divide(green.add(nir)).rename("NDWI")
        sbi = red.pow(2).add(nir.pow(2)).sqrt().rename("SBI")
        return ndwi.addBands(sbi)

    monthly = joined.map(mask_and_index).select(["NDWI", "SBI"]).median()
    return monthly.toFloat().set("system:time_start", d0.millis())


def baseline_mean(aoi: ee.FeatureCollection, sy, sm, ey, em) -> ee.Image:
    """Mean NDWI and SBI across the baseline window (per-pixel)."""
    images = [build_month_index(aoi, y, m) for y, m in months_between(sy, sm, ey, em)]
    return ee.ImageCollection(images).mean().rename(["NDWI_base", "SBI_base"])


def persistence_counts(aoi: ee.FeatureCollection) -> tuple[ee.Image, ee.Image]:
    """
    For each post-flood month, flag pixels where anomaly > threshold.
    Sum flags across months = persistence count.
    """
    base = baseline_mean(
        aoi, BASELINE_START_Y, BASELINE_START_M, BASELINE_END_Y, BASELINE_END_M
    )

    ndwi_flags = []
    sbi_flags = []
    for y, m in months_between(POST_START_Y, POST_START_M, POST_END_Y, POST_END_M):
        monthly = build_month_index(aoi, y, m)
        ndwi_anom = monthly.select("NDWI").subtract(base.select("NDWI_base"))
        sbi_anom = monthly.select("SBI").subtract(base.select("SBI_base"))
        ndwi_flags.append(ndwi_anom.gt(NDWI_ANOMALY_THRESHOLD))
        sbi_flags.append(sbi_anom.gt(SBI_ANOMALY_THRESHOLD))

    ndwi_persist = (
        ee.ImageCollection(ndwi_flags).sum().rename("ndwi_persist_months").toInt16()
    )
    sbi_persist = (
        ee.ImageCollection(sbi_flags).sum().rename("sbi_persist_months").toInt16()
    )
    return ndwi_persist, sbi_persist


def main() -> None:
    init_ee()
    aoi = load_aoi()
    region = aoi.geometry().bounds()

    print("Building NDWI + SBI persistence proxies for the post-flood year...")
    print(
        f"  Post-flood window: {POST_START_Y}-{POST_START_M:02d} → "
        f"{POST_END_Y}-{POST_END_M:02d} (exclusive)"
    )
    print(
        f"  Baseline window:   {BASELINE_START_Y}-{BASELINE_START_M:02d} → "
        f"{BASELINE_END_Y}-{BASELINE_END_M:02d} (exclusive)"
    )
    print(f"  NDWI anomaly threshold: {NDWI_ANOMALY_THRESHOLD}")
    print(f"  SBI anomaly threshold:  {SBI_ANOMALY_THRESHOLD}")

    ndwi_persist, sbi_persist = persistence_counts(aoi)

    task_ndwi = export_image_to_asset(
        image=ndwi_persist,
        description="ndwi_persistence_months",
        asset_id=ASSET_NDWI,
        region=region,
        scale=30,
    )
    task_ndwi.start()
    print(f"\n  NDWI persistence task: {task_ndwi.id}")

    task_sbi = export_image_to_asset(
        image=sbi_persist,
        description="sbi_persistence_months",
        asset_id=ASSET_SBI,
        region=region,
        scale=30,
    )
    task_sbi.start()
    print(f"  SBI persistence task:  {task_sbi.id}")

    print(
        "\nMonitor at https://code.earthengine.google.com/tasks?project=flood-recovery-sindh"
    )


if __name__ == "__main__":
    main()
