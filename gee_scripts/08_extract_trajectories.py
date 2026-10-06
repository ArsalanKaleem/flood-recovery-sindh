"""
Day 7 — Extract per-cell NDVI/EVI monthly trajectories.

For each district, run reduceRegions across all 69 S2 monthly composites
and export one CSV with (cell_id, month, ndvi, evi) rows.

22 districts × ~25k cells × 69 months each ≈ 38M total output rows.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ee
from src.gee_utils import init_ee, load_aoi

BASE = "projects/flood-recovery-sindh/assets"
S2_COLL = f"{BASE}/s2_monthly_composites"
GRID = f"{BASE}/analysis_grid"
EXPORT_FOLDER = "flood-recovery-exports/trajectories"


def extract_district(
    district_name: str, grid: ee.FeatureCollection, composites: ee.ImageCollection
) -> ee.batch.Task:
    """Submit one export task for one district's full NDVI/EVI time series."""
    cells = grid.filter(ee.Filter.eq("district", district_name))

    def reduce_month(img):
        month = img.get("month")
        reduced = img.select(["NDVI", "EVI"]).reduceRegions(
            collection=cells,
            reducer=ee.Reducer.mean(),
            scale=30,
            tileScale=4,  # helps avoid memory errors on dense polygons
        )
        # Attach month to each feature.
        return reduced.map(lambda f: f.set("month", month))

    # Flatten: ImageCollection of FeatureCollections → single FeatureCollection.
    all_months = composites.map(reduce_month).flatten()

    # Keep only the fields we need.
    def trim(f):
        return ee.Feature(
            None,
            {
                "cell_id": f.get("cell_id"),
                "month": f.get("month"),
                "ndvi": f.get("NDVI"),
                "evi": f.get("EVI"),
            },
        )

    all_months = all_months.map(trim)

    desc = f"trajectory_{district_name.replace(' ', '_')}"
    task = ee.batch.Export.table.toDrive(
        collection=all_months,
        description=desc,
        folder="flood-recovery-exports",
        fileNamePrefix=desc,
        fileFormat="CSV",
        selectors=["cell_id", "month", "ndvi", "evi"],
    )
    return task, desc


def main() -> None:
    init_ee()
    aoi = load_aoi()
    grid = ee.FeatureCollection(GRID)
    composites = ee.ImageCollection(S2_COLL).sort("system:time_start")

    districts = sorted(aoi.aggregate_array("district").getInfo())
    print(f"Submitting {len(districts)} district trajectory exports...")

    for i, d in enumerate(districts, 1):
        task, desc = extract_district(d, grid, composites)
        task.start()
        print(f"  [{i:2d}] {desc:40s} task={task.id}")
        time.sleep(1.0)  # pace submissions

    print(f"\nAll {len(districts)} tasks submitted.")
    print("Each will land in Google Drive → flood-recovery-exports/ as a CSV.")
    print(
        "Monitor at https://code.earthengine.google.com/tasks?project=flood-recovery-sindh"
    )


if __name__ == "__main__":
    main()
