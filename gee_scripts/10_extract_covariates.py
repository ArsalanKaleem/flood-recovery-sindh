"""
Day 10 — Per-cell covariate extraction. Resume-safe.
"""

from __future__ import annotations
import sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ee
from src.gee_utils import init_ee, load_aoi

BASE = "projects/flood-recovery-sindh/assets"
GRID = f"{BASE}/analysis_grid"

LAYERS = [
    ("terrain", f"{BASE}/srtm_terrain", ["elevation_m", "slope_deg", "twi_approx"], 30),
    ("rainfall", f"{BASE}/chirps_rainfall_anomaly_2022", ["rainfall_anomaly_mm"], 5000),
    ("ndwi_persist", f"{BASE}/ndwi_persistence_months", ["ndwi_persist_months"], 30),
    ("sbi_persist", f"{BASE}/sbi_persistence_months", ["sbi_persist_months"], 30),
    ("worldpop", f"{BASE}/worldpop_density", ["population"], 100),
]


def active_task_descriptions() -> set[str]:
    """Descriptions of tasks currently PENDING, RUNNING, or COMPLETED (last 10d)."""
    try:
        ops = ee.data.listOperations()
    except Exception:
        return set()
    active = set()
    for t in ops:
        md = t.get("metadata", {})
        state = md.get("state", "")
        if state in ("PENDING", "RUNNING", "SUCCEEDED", "COMPLETED"):
            desc = md.get("description")
            if desc:
                active.add(desc)
    return active


def extract_layer(
    district: str,
    cells: ee.FeatureCollection,
    prefix: str,
    asset_id: str,
    bands: list[str],
    scale: int,
) -> tuple[ee.batch.Task, str]:
    img = ee.Image(asset_id).select(bands)
    reduced = img.reduceRegions(
        collection=cells,
        reducer=ee.Reducer.mean(),
        scale=scale,
        tileScale=4,
    )

    # For multi-band images, reduceRegions puts each band's mean in a column
    # named after the band itself. For single-band it uses 'mean'.
    def trim(f):
        props = {"cell_id": f.get("cell_id")}
        if len(bands) == 1:
            props[bands[0]] = f.get("mean")
        else:
            for b in bands:
                props[b] = f.get(b)
        return ee.Feature(None, props)

    reduced = reduced.map(trim)
    desc = f"{prefix}_{district.replace(' ', '_')}"
    task = ee.batch.Export.table.toDrive(
        collection=reduced,
        description=desc,
        folder="flood-recovery-exports",
        fileNamePrefix=desc,
        fileFormat="CSV",
        selectors=["cell_id"] + bands,
    )
    return task, desc


def main() -> None:
    init_ee()
    aoi = load_aoi()
    grid = ee.FeatureCollection(GRID)
    districts = sorted(aoi.aggregate_array("district").getInfo())

    print("Fetching current task list to skip in-flight / completed...")
    skip = active_task_descriptions()
    print(f"  {len(skip)} tasks already known.")

    submitted, skipped = 0, 0
    print(f"\nSubmitting any missing tasks...\n")
    for d in districts:
        cells = grid.filter(ee.Filter.eq("district", d))
        for prefix, asset_id, bands, scale in LAYERS:
            desc = f"{prefix}_{d.replace(' ', '_')}"
            if desc in skip:
                skipped += 1
                continue
            try:
                task, _ = extract_layer(d, cells, prefix, asset_id, bands, scale)
                task.start()
                submitted += 1
                print(f"  [{submitted:3d}] {desc}")
                time.sleep(0.4)
            except Exception as e:
                print(f"  ERROR submitting {desc}: {e}")

    print(f"\nSubmitted {submitted}, skipped {skipped} already in flight.")
    print(
        "Monitor at https://code.earthengine.google.com/tasks?project=flood-recovery-sindh"
    )


if __name__ == "__main__":
    main()
