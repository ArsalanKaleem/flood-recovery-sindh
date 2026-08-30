"""
Day 3 — Sentinel-1 SAR flood extent and per-pixel flood-duration proxy.

Method:
1. Build S1 GRD VV/VH collection over Aug-Oct 2022 for the AOI.
2. Apply a VV threshold (default -17 dB) to flag inundated pixels per scene.
3. Mask out permanent water using JRC Global Surface Water.
4. Sum per-pixel inundation flags across all dates → flood-duration proxy.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ee
from src.gee_utils import init_ee, load_aoi, export_image_to_drive


FLOOD_START = "2022-08-01"
FLOOD_END = "2022-11-01"
VV_THRESHOLD_DB = -17.0


def s1_collection(aoi: ee.FeatureCollection, start: str, end: str) -> ee.ImageCollection:
    return (ee.ImageCollection("COPERNICUS/S1_GRD")
            .filterBounds(aoi.geometry())
            .filterDate(start, end)
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
            .filter(ee.Filter.eq("instrumentMode", "IW"))
            .select("VV"))


def flag_water(img: ee.Image, threshold_db: float = VV_THRESHOLD_DB) -> ee.Image:
    """Simple VV threshold; add a Refined Lee speckle filter here if needed."""
    return img.lt(threshold_db).rename("water").copyProperties(img, ["system:time_start"])


def flood_duration(coll_flags: ee.ImageCollection, permanent_water: ee.Image) -> ee.Image:
    """Sum of flags across the collection, minus permanent water bodies."""
    duration = coll_flags.sum().rename("flood_duration_days")
    return duration.updateMask(permanent_water.Not())


def main() -> None:
    init_ee()
    aoi = load_aoi()

    s1 = s1_collection(aoi, FLOOD_START, FLOOD_END)
    print(f"S1 scene count over AOI ({FLOOD_START} → {FLOOD_END}): {s1.size().getInfo()}")

    flags = s1.map(flag_water)

    # JRC Global Surface Water: pixels flagged as permanent water (>= 90% occurrence).
    gsw = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("occurrence")
    permanent = gsw.gte(90).unmask(0)

    duration = flood_duration(flags, permanent)

    task = export_image_to_drive(
        image=duration.toInt16(),
        description="flood_duration_2022",
        region=aoi.geometry().bounds(),
        scale=30,
    )
    task.start()
    print("Export task started: flood_duration_2022")


if __name__ == "__main__":
    main()
