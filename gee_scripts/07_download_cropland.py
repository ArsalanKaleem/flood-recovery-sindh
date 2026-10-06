"""One-time: download the ESA cropland mask from GEE to local disk."""

from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ee
from src.gee_utils import init_ee, load_aoi

init_ee()
aoi = load_aoi()
region = aoi.geometry().bounds()

cropland = ee.Image("projects/flood-recovery-sindh/assets/esa_cropland_mask")

task = ee.batch.Export.image.toDrive(
    image=cropland,
    description="esa_cropland_mask_download",
    folder="flood-recovery-exports",
    fileNamePrefix="esa_cropland_mask",
    region=region,
    scale=10,
    crs="EPSG:32642",
    maxPixels=1e13,
)
task.start()
print(f"Submitted: {task.id}")
print("Will land in Google Drive → flood-recovery-exports/")
