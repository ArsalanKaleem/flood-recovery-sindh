"""
Day 9 — Secondary driver indicators per cell.

Three exports:
  1. flood_duration_2022 → mean per cell (one value)
  2. ERA5 soil moisture  → full 68-month trajectory per cell (needed for
     persistence metric computed locally on Day 9b)
  3. ERA5 LST            → mean post-flood anomaly per cell (one value)

22 districts × 3 metrics = 66 tasks. Each lands as a CSV in Drive.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ee
from src.gee_utils import init_ee, load_aoi

BASE = "projects/flood-recovery-sindh/assets"
GRID = f"{BASE}/analysis_grid"
FLOOD_IMG = f"{BASE}/flood_duration_2022"
SM_COLL = f"{BASE}/era5_soil_moisture"
LST_COLL = f"{BASE}/era5_lst"

# Post-flood window for LST anomaly: Nov 2022 - Apr 2023 (6 Rabi season months)
LST_POST_START = "2022-11-01"
LST_POST_END = "2023-05-01"
LST_BASE_START = "2021-11-01"
LST_BASE_END = "2022-05-01"


def extract_flood_duration(district: str, cells: ee.FeatureCollection) -> ee.batch.Task:
    img = ee.Image(FLOOD_IMG)
    reduced = img.reduceRegions(
        collection=cells,
        reducer=ee.Reducer.mean(),
        scale=30,
        tileScale=4,
    )

    def trim(f):
        return ee.Feature(
            None,
            {
                "cell_id": f.get("cell_id"),
                "flood_duration": f.get("mean"),
            },
        )

    reduced = reduced.map(trim)
    desc = f"flood_dur_{district.replace(' ', '_')}"
    return (
        ee.batch.Export.table.toDrive(
            collection=reduced,
            description=desc,
            folder="flood-recovery-exports",
            fileNamePrefix=desc,
            fileFormat="CSV",
            selectors=["cell_id", "flood_duration"],
        ),
        desc,
    )


def extract_sm_trajectory(district: str, cells: ee.FeatureCollection) -> ee.batch.Task:
    sm = ee.ImageCollection(SM_COLL).sort("system:time_start")

    def reduce_month(img):
        month = img.get("month")
        reduced = img.reduceRegions(
            collection=cells,
            reducer=ee.Reducer.mean(),
            scale=1000,
            tileScale=4,
        )
        return reduced.map(lambda f: f.set("month", month))

    all_months = sm.map(reduce_month).flatten()

    def trim(f):
        return ee.Feature(
            None,
            {
                "cell_id": f.get("cell_id"),
                "month": f.get("month"),
                "soil_moisture": f.get("mean"),
            },
        )

    all_months = all_months.map(trim)
    desc = f"sm_traj_{district.replace(' ', '_')}"
    return (
        ee.batch.Export.table.toDrive(
            collection=all_months,
            description=desc,
            folder="flood-recovery-exports",
            fileNamePrefix=desc,
            fileFormat="CSV",
            selectors=["cell_id", "month", "soil_moisture"],
        ),
        desc,
    )


def extract_lst_anomaly(district: str, cells: ee.FeatureCollection) -> ee.batch.Task:
    lst = ee.ImageCollection(LST_COLL)
    post = lst.filterDate(LST_POST_START, LST_POST_END).mean()
    base = lst.filterDate(LST_BASE_START, LST_BASE_END).mean()
    anom = post.subtract(base).rename("lst_anom_K")

    reduced = anom.reduceRegions(
        collection=cells,
        reducer=ee.Reducer.mean(),
        scale=1000,
        tileScale=4,
    )

    def trim(f):
        return ee.Feature(
            None,
            {
                "cell_id": f.get("cell_id"),
                "lst_anom_K": f.get("mean"),
            },
        )

    reduced = reduced.map(trim)
    desc = f"lst_anom_{district.replace(' ', '_')}"
    return (
        ee.batch.Export.table.toDrive(
            collection=reduced,
            description=desc,
            folder="flood-recovery-exports",
            fileNamePrefix=desc,
            fileFormat="CSV",
            selectors=["cell_id", "lst_anom_K"],
        ),
        desc,
    )


def main() -> None:
    init_ee()
    aoi = load_aoi()
    grid = ee.FeatureCollection(GRID)
    districts = sorted(aoi.aggregate_array("district").getInfo())

    print(
        f"Submitting 3 × {len(districts)} = {3*len(districts)} secondary indicator tasks...\n"
    )

    for i, d in enumerate(districts, 1):
        cells = grid.filter(ee.Filter.eq("district", d))

        t1, desc1 = extract_flood_duration(d, cells)
        t1.start()
        print(f"  [{i:2d}] {desc1}")

        t2, desc2 = extract_sm_trajectory(d, cells)
        t2.start()
        print(f"       {desc2}")

        t3, desc3 = extract_lst_anomaly(d, cells)
        t3.start()
        print(f"       {desc3}")

        time.sleep(1.5)

    print("\nAll tasks submitted.")
    print("Each will land in Google Drive → flood-recovery-exports/ as CSV.")
    print(
        "Monitor at https://code.earthengine.google.com/tasks?project=flood-recovery-sindh"
    )


if __name__ == "__main__":
    main()
