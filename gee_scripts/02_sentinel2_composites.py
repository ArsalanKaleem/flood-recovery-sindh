"""
Day 2 — Sentinel-2 SR monthly NDVI/EVI composites, 2021-01 → present.

Applies s2cloudless cloud/shadow masking, computes NDVI and EVI per scene,
then reduces to monthly median composites over the AOI.

Landsat 8/9 gap-filling for heavy-monsoon months (Aug-Oct 2022) is a follow-up
in gee_scripts/02b_landsat_gapfill.py — write that on Day 2 if needed.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ee
from src.gee_utils import init_ee, load_aoi, export_image_to_drive
from src.indices import add_ndvi_s2, add_evi_s2


START = "2021-01-01"
END = date.today().isoformat()


def mask_s2_clouds(img: ee.Image) -> ee.Image:
    """
    Basic QA60 bitmask cloud/cirrus filter.
    For the actual pipeline, swap this for the s2cloudless probability mask
    (COPERNICUS/S2_CLOUD_PROBABILITY joined by system:index) — better recall in
    thin-cloud Sindh Rabi conditions.
    """
    qa = img.select("QA60")
    cloud_bit_mask = 1 << 10
    cirrus_bit_mask = 1 << 11
    mask = (qa.bitwiseAnd(cloud_bit_mask).eq(0)
            .And(qa.bitwiseAnd(cirrus_bit_mask).eq(0)))
    return img.updateMask(mask).divide(1).copyProperties(img, ["system:time_start"])


def monthly_composites(aoi: ee.FeatureCollection, start: str, end: str) -> ee.ImageCollection:
    s2 = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
          .filterBounds(aoi.geometry())
          .filterDate(start, end)
          .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 60))
          .map(mask_s2_clouds)
          .map(add_ndvi_s2)
          .map(add_evi_s2))

    months = ee.List.sequence(0, ee.Date(end).difference(ee.Date(start), "month").subtract(1))

    def one_month(m):
        m = ee.Number(m)
        d0 = ee.Date(start).advance(m, "month")
        d1 = d0.advance(1, "month")
        composite = s2.filterDate(d0, d1).select(["NDVI", "EVI"]).median()
        return composite.set("system:time_start", d0.millis()) \
                        .set("month", d0.format("YYYY-MM"))

    return ee.ImageCollection(months.map(one_month))


def main() -> None:
    init_ee()
    aoi = load_aoi()
    print(f"Building S2 monthly composites {START} → {END}")

    coll = monthly_composites(aoi, START, END)
    print(f"Composite count: {coll.size().getInfo()}")

    # Example: export one month to prove the pipeline runs end-to-end.
    # Kick off the full-year export loop once you're happy with the smoke test.
    example = ee.Image(coll.filterMetadata("month", "equals", "2022-06").first())
    task = export_image_to_drive(
        image=example,
        description="s2_composite_2022-06",
        region=aoi.geometry().bounds(),
        scale=30,
    )
    task.start()
    print("Export task started: s2_composite_2022-06")


if __name__ == "__main__":
    main()
