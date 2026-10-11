"""
Day 10 — Per-cell extraction of remaining covariates from GEE assets.

Six single-image assets × 22 districts = 132 small CSV exports.
Each lands in Google Drive → flood-recovery-exports/ with a prefix identifying
the covariate.
"""

from __future__ import annotations
import sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ee
from src.gee_utils import init_ee, load_aoi

BASE = "projects/flood-recovery-sindh/assets"
GRID = f"{BASE}/analysis_grid"

# One row per layer: (prefix, asset_id, bands_to_keep, output_scale, selectors)
LAYERS = [
    (
        "terrain",
        f"{BASE}/srtm_terrain",
        ["elevation_m", "slope_deg", "twi_approx"],
        30,
        ["elevation_m", "slope_deg", "twi_approx"],
    ),
    (
        "rainfall",
        f"{BASE}/chirps_rainfall_anomaly_2022",
        ["rainfall_anomaly_mm"],
        5000,
        ["rainfall_anomaly_mm"],
    ),
    (
        "ndwi_persist",
        f"{BASE}/ndwi_persistence_months",
        ["ndwi_persist_months"],
        30,
        ["ndwi_persist_months"],
    ),
    (
        "sbi_persist",
        f"{BASE}/sbi_persistence_months",
        ["sbi_persist_months"],
        30,
        ["sbi_persist_months"],
    ),
    ("worldpop", f"{BASE}/worldpop_density", ["population"], 100, ["population"]),
]


def extract_layer(
    district: str,
    cells: ee.FeatureCollection,
    prefix: str,
    asset_id: str,
    band_names: list[str],
    scale: int,
    selectors: list[str],
) -> tuple[ee.batch.Task, str]:
    img = ee.Image(asset_id).select(band_names)
    reduced = img.reduceRegions(
        collection=cells,
        reducer=ee.Reducer.mean(),
        scale=scale,
        tileScale=4,
    )

    # The default reducer returns `mean` for a single band or `band_mean` for
    # multi-band; rename for consistency.
    if len(band_names) == 1:
        rename_map = {"mean": band_names[0]}
    else:
        rename_map = {f"{b}": b for b in band_names}

    def trim(f):
        props = {"cell_id": f.get("cell_id")}
        for src_key, dst_key in rename_map.items():
            props[dst_key] = f.get(src_key)
        return ee.Feature(None, props)

    reduced = reduced.map(trim)
    desc = f"{prefix}_{district.replace(' ', '_')}"
    task = ee.batch.Export.table.toDrive(
        collection=reduced,
        description=desc,
        folder="flood-recovery-exports",
        fileNamePrefix=desc,
        fileFormat="CSV",
        selectors=["cell_id"] + selectors,
    )
    return task, desc


def main() -> None:
    init_ee()
    aoi = load_aoi()
    grid = ee.FeatureCollection(GRID)
    districts = sorted(aoi.aggregate_array("district").getInfo())

    print(
        f"Submitting {len(LAYERS)} × {len(districts)} = "
        f"{len(LAYERS)*len(districts)} covariate tasks...\n"
    )

    submitted = 0
    for d in districts:
        cells = grid.filter(ee.Filter.eq("district", d))
        for prefix, asset_id, bands, scale, sels in LAYERS:
            task, desc = extract_layer(d, cells, prefix, asset_id, bands, scale, sels)
            task.start()
            submitted += 1
            print(f"  [{submitted:3d}] {desc}")
            time.sleep(0.4)

    print(f"\nSubmitted {submitted} tasks.")
    print(
        "Monitor at https://code.earthengine.google.com/tasks?project=flood-recovery-sindh"
    )


if __name__ == "__main__":
    main()
