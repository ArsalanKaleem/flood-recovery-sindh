"""
Day 1 — AOI setup and GEE connection test.

Confirms:
1. GEE Python API authenticates against your project.
2. The 7-district AOI GeoJSON loads and reduceRegions runs against it.
3. A trivial CHIRPS extraction returns numbers (proves the whole stack works).
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make src/ importable when this script is run directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ee
from src.gee_utils import init_ee, load_aoi


def main() -> None:
    init_ee()
    print("Earth Engine initialized.")

    aoi = load_aoi()
    n = aoi.size().getInfo()
    print(f"AOI loaded: {n} districts.")

    # Trivial reduceRegions sanity check — mean CHIRPS rainfall for one recent month.
    chirps = (
        ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")
        .filterDate("2025-07-01", "2025-08-01")
        .sum()
        .rename("rainfall_mm")
    )
    reduced = chirps.reduceRegions(
        collection=aoi,
        reducer=ee.Reducer.mean(),
        scale=5000,
    ).getInfo()

    print("\nPer-district rainfall (July 2025, mm):")
    for f in reduced["features"]:
        props = f["properties"]
        name = props.get("ADM2_EN") or props.get("district") or props.get("name") or "?"
        rain = props.get("mean")
        print(f"  {name:25s} {rain:.1f}" if rain is not None else f"  {name:25s} <no data>")


if __name__ == "__main__":
    main()
