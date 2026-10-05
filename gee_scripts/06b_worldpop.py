"""Day 5b — WorldPop population density for Sindh AOI."""

from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ee
from src.gee_utils import init_ee, load_aoi, export_image_to_asset

ASSET_ID = "projects/flood-recovery-sindh/assets/worldpop_density"


def main() -> None:
    init_ee()
    aoi = load_aoi()
    region = aoi.geometry().bounds()

    # Most recent WorldPop global 100 m for Pakistan.
    wp = (
        ee.ImageCollection("WorldPop/GP/100m/pop")
        .filter(ee.Filter.eq("country", "PAK"))
        .filter(ee.Filter.eq("year", 2020))
        .first()
        .clip(aoi.geometry())
        .rename("population")
        .toFloat()
    )

    task = export_image_to_asset(
        image=wp,
        description="worldpop_density",
        asset_id=ASSET_ID,
        region=region,
        scale=100,
    )
    task.start()
    print(f"Submitted: {task.id}")


if __name__ == "__main__":
    main()
