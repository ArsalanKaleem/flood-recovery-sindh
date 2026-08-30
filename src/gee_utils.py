"""
Google Earth Engine helpers: initialization, AOI loading, common export patterns.

Every GEE script and notebook should start with:

    from src.gee_utils import init_ee, load_aoi
    init_ee()
    aoi = load_aoi()
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import ee
from dotenv import load_dotenv

# Repo root, resolved from this file's location.
REPO_ROOT = Path(__file__).resolve().parents[1]
AOI_PATH = REPO_ROOT / "config" / "aoi_districts.geojson"


def init_ee(project: Optional[str] = None) -> None:
    """
    Initialize Earth Engine.

    Reads GEE_PROJECT_ID from .env if `project` is not passed.
    Run `earthengine authenticate` once at the shell first.
    """
    load_dotenv(REPO_ROOT / ".env")
    project_id = project or os.getenv("GEE_PROJECT_ID")
    if not project_id:
        raise RuntimeError(
            "GEE_PROJECT_ID not set. Copy .env.example to .env and fill it in, "
            "or pass project= explicitly."
        )
    ee.Initialize(project=project_id)


def load_aoi(path: Path = AOI_PATH) -> ee.FeatureCollection:
    """Load the 7-district AOI as an ee.FeatureCollection."""
    import json

    if not path.exists():
        raise FileNotFoundError(
            f"AOI not found at {path}. Create it on Day 1 from GAUL/OSM boundaries "
            "for Dadu, Jamshoro, Larkana, Qambar-Shahdadkot, Jacobabad, Shikarpur, Sanghar."
        )
    with open(path) as f:
        gj = json.load(f)
    return ee.FeatureCollection(gj)


def export_image_to_drive(
    image: ee.Image,
    description: str,
    region: ee.Geometry,
    scale: int = 30,
    folder: Optional[str] = None,
    crs: str = "EPSG:32642",  # UTM 42N covers all seven Sindh districts
) -> ee.batch.Task:
    """
    Standard image export to Drive. Returns the task — call .start() yourself
    so you stay in control of task submission ordering.
    """
    folder = folder or os.getenv("GEE_DRIVE_FOLDER", "flood-recovery-exports")
    return ee.batch.Export.image.toDrive(
        image=image,
        description=description,
        folder=folder,
        fileNamePrefix=description,
        region=region,
        scale=scale,
        crs=crs,
        maxPixels=1e13,
    )


def export_image_to_asset(
    image: ee.Image,
    description: str,
    asset_id: str,
    region: ee.Geometry,
    scale: int = 30,
    crs: str = "EPSG:32642",
) -> ee.batch.Task:
    """
    Export an image to a GEE asset (server-side storage inside your project).

    asset_id is the full path where this image will live, e.g.
    'projects/flood-recovery-sindh/assets/s2_monthly_composites/s2_2022-06'.
    Parent collection or folder must already exist.
    """
    return ee.batch.Export.image.toAsset(
        image=image,
        description=description,
        assetId=asset_id,
        region=region,
        scale=scale,
        crs=crs,
        maxPixels=1e13,
        pyramidingPolicy={".default": "mean"},
    )


def export_table_to_drive(
    features: ee.FeatureCollection,
    description: str,
    folder: Optional[str] = None,
    file_format: str = "CSV",
) -> ee.batch.Task:
    """Standard tabular export to Drive."""
    folder = folder or os.getenv("GEE_DRIVE_FOLDER", "flood-recovery-exports")
    return ee.batch.Export.table.toDrive(
        collection=features,
        description=description,
        folder=folder,
        fileNamePrefix=description,
        fileFormat=file_format,
    )
