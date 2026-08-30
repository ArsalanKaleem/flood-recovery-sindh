"""
Day 2 — Submit S2 monthly composite exports to a GEE ImageCollection asset.

Loops over 2021-01 → current month and submits one export task per month.
Each image is written to projects/<project>/assets/s2_monthly_composites/,
so downstream code can pull the whole time series with a single
ee.ImageCollection() call — no local storage needed.
"""

from __future__ import annotations

import sys
import time
from datetime import date
from importlib import import_module
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gee_scripts"))

from src.gee_utils import init_ee, load_aoi, export_image_to_asset

_mod = import_module("02_sentinel2_composites")
build_month_composite = _mod.build_month_composite

START_YEAR, START_MONTH = 2021, 1
END = date.today().replace(day=1)  # first of current month, exclusive
ASSET_COLLECTION = "projects/flood-recovery-sindh/assets/s2_monthly_composites"


def month_iter(y0: int, m0: int, end: date):
    y, m = y0, m0
    while (y, m) < (end.year, end.month):
        yield y, m
        m += 1
        if m > 12:
            m = 1
            y += 1


def main() -> None:
    init_ee()
    aoi = load_aoi()
    region = aoi.geometry().bounds()

    submitted = 0
    for y, m in month_iter(START_YEAR, START_MONTH, END):
        desc = f"s2_composite_{y}-{m:02d}"
        asset_id = f"{ASSET_COLLECTION}/{desc}"
        composite = build_month_composite(aoi, y, m)
        task = export_image_to_asset(
            image=composite,
            description=desc,
            asset_id=asset_id,
            region=region,
            scale=30,
        )
        task.start()
        submitted += 1
        print(f"  [{submitted:3d}] submitted {desc}  task={task.id}")
        time.sleep(0.5)

    print(f"\nDone. Submitted {submitted} export tasks to {ASSET_COLLECTION}")
    print(
        "Monitor at https://code.earthengine.google.com/tasks?project=flood-recovery-sindh"
    )


if __name__ == "__main__":
    main()
