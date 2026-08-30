"""
Day 2 — Sentinel-2 SR monthly NDVI/EVI composites.

Structured to build ONE MONTH at a time (avoids the memory limit that hits
if you try to construct 5+ years of joined S2/s2cloudless collections in a
single call). main() runs a smoke test on June 2022. Full-batch loop lives
in a separate script we'll add after the smoke test passes.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ee
from src.gee_utils import init_ee, load_aoi, export_image_to_drive
from src.indices import add_ndvi_s2, add_evi_s2

CLOUD_PROB_THRESHOLD = 40
NIR_DRK_THRESH = 0.15
CLD_PRJ_DIST_KM = 1
BUFFER_M = 50


def build_month_composite(aoi: ee.FeatureCollection, year: int, month: int) -> ee.Image:
    """Build one monthly NDVI/EVI median composite over the AOI."""
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

    def mask_clouds_shadows(img):
        cloud_prob = ee.Image(img.get("cloud_prob")).select("probability")
        is_cloud = cloud_prob.gt(CLOUD_PROB_THRESHOLD)
        not_water = img.select("SCL").neq(6)
        dark = img.select("B8").lt(NIR_DRK_THRESH * 10000).multiply(not_water)
        shadow_az = ee.Number(90).subtract(
            ee.Number(img.get("MEAN_SOLAR_AZIMUTH_ANGLE"))
        )
        cloud_proj = (
            is_cloud.directionalDistanceTransform(shadow_az, CLD_PRJ_DIST_KM * 10)
            .reproject(crs=img.select(0).projection(), scale=100)
            .select("distance")
            .mask()
        )
        shadows = cloud_proj.multiply(dark)
        mask = is_cloud.add(shadows).gt(0).focalMin(2).focalMax(BUFFER_M * 2 / 20)
        return img.updateMask(mask.Not())

    masked = joined.map(mask_clouds_shadows).map(add_ndvi_s2).map(add_evi_s2)
    composite = masked.select(["NDVI", "EVI"]).median().toFloat()

    return composite.set("month", d0.format("YYYY-MM")).set(
        "system:time_start", d0.millis()
    )


def main() -> None:
    init_ee()
    aoi = load_aoi()

    # Smoke test: one month.
    year, month = 2022, 6
    print(f"Building smoke-test composite for {year}-{month:02d}")

    composite = build_month_composite(aoi, year, month)

    task = export_image_to_drive(
        image=composite,
        description=f"s2_composite_{year}-{month:02d}_smoketest",
        region=aoi.geometry().bounds(),
        scale=30,
    )
    task.start()
    print(f"Export submitted: {task.id}")
    print("Track it at https://code.earthengine.google.com/tasks")


if __name__ == "__main__":
    main()
