"""
Day 5 — Consolidated export pipeline.

One script to submit all remaining exports for the raw data stack.
Runs everything from scripts 02-04 in batch, tags each task with a
readable description, and prints a manifest you can paste into
docs/data_dictionary.md.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ee
from src.gee_utils import init_ee, load_aoi, export_image_to_drive


def submit_all(aoi: ee.FeatureCollection) -> list[dict]:
    """Return a manifest of every submitted task so you can track them in the Tasks tab."""
    manifest: list[dict] = []

    # Sentinel-2 monthly composites, year by year — fill in the loop when composites are ready.
    # Sentinel-1 flood duration — already handled in script 03.
    # CHIRPS anomaly, ERA5 stacks, SRTM terrain, ESA cropland — call script 04 builders.

    # TODO Day 5: assemble the actual export list here.
    return manifest


def main() -> None:
    init_ee()
    aoi = load_aoi()
    manifest = submit_all(aoi)
    print(f"Submitted {len(manifest)} export tasks.")
    for row in manifest:
        print(f"  - {row['description']:40s} scale={row['scale']}")


if __name__ == "__main__":
    main()
