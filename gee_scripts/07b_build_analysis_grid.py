"""
Day 6 — Build the cropland analysis grid.

- 300 m square cells in UTM 42N (EPSG:32642), origin-aligned
- Only cells where >=50% of area is ESA cropland (class 40)
- Each cell tagged with its district
"""

from __future__ import annotations
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from shapely.geometry import box
from tqdm import tqdm

REPO = Path(__file__).resolve().parents[1]
AOI_PATH = REPO / "config" / "aoi_districts.geojson"
RASTER_DIR = REPO / "data" / "raw" / "esa_worldcover"
OUT_PATH = REPO / "data" / "processed" / "analysis_grid.gpkg"

CELL_SIZE_M = 300
CROPLAND_THRESHOLD = 0.50  # >= 50% of cell area must be cropland


def load_cropland_raster() -> (
    tuple[np.ndarray, rasterio.transform.Affine, rasterio.CRS]
):
    """Load ESA cropland mask from one or more downloaded tiles, merged."""
    tifs = sorted(RASTER_DIR.glob("esa_cropland_mask*.tif"))
    if not tifs:
        raise FileNotFoundError(
            f"No cropland tiles in {RASTER_DIR}. Download from Drive first."
        )
    print(f"Loading {len(tifs)} cropland tile(s)...")
    if len(tifs) == 1:
        with rasterio.open(tifs[0]) as src:
            arr = src.read(1)
            return arr, src.transform, src.crs
    # Multi-tile: use rasterio.merge
    from rasterio.merge import merge

    srcs = [rasterio.open(t) for t in tifs]
    arr, transform = merge(srcs)
    crs = srcs[0].crs
    for s in srcs:
        s.close()
    return arr[0], transform, crs


def build_grid(
    aoi_proj: gpd.GeoDataFrame, cell_size: int = CELL_SIZE_M
) -> gpd.GeoDataFrame:
    """Dense grid of square cells covering the AOI bbox, origin-aligned."""
    minx, miny, maxx, maxy = aoi_proj.total_bounds
    # Snap to the cell grid.
    minx = (int(minx) // cell_size) * cell_size
    miny = (int(miny) // cell_size) * cell_size
    maxx = (int(maxx) // cell_size + 1) * cell_size
    maxy = (int(maxy) // cell_size + 1) * cell_size

    xs = np.arange(minx, maxx, cell_size)
    ys = np.arange(miny, maxy, cell_size)
    print(f"Grid span: {len(xs)} × {len(ys)} = {len(xs)*len(ys):,} candidate cells")

    cells = [box(x, y, x + cell_size, y + cell_size) for x in xs for y in ys]
    grid = gpd.GeoDataFrame({"geometry": cells}, crs=aoi_proj.crs)
    return grid


def tag_district(grid: gpd.GeoDataFrame, aoi: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Spatial join each cell to its district (by centroid)."""
    print("Tagging cells with district (centroid join)...")
    centroids = grid.copy()
    centroids["geometry"] = centroids.geometry.centroid
    joined = gpd.sjoin(
        centroids, aoi[["district", "geometry"]], how="inner", predicate="within"
    )
    grid = grid.loc[joined.index].copy()
    grid["district"] = joined["district"].values
    print(f"  Inside AOI: {len(grid):,} cells")
    return grid


def compute_cropland_frac(
    grid: gpd.GeoDataFrame,
    cropland_arr: np.ndarray,
    transform,
    cell_size: int = CELL_SIZE_M,
) -> gpd.GeoDataFrame:
    """
    Fraction of each cell's area classified as cropland.

    Memory-safe: rasterizes cell IDs row-chunk by row-chunk and accumulates
    sums + counts per cell_id in a dict, instead of building a 1.4-billion-row
    dataframe.
    """
    print("Computing cropland fraction per cell (chunked for memory)...")
    grid = grid.reset_index(drop=True)
    grid["cell_id"] = np.arange(1, len(grid) + 1)

    H, W = cropland_arr.shape
    CHUNK_ROWS = 2000  # ~80M pixels per chunk, ~300 MB working set

    # Accumulators: per cell_id, running sum and count.
    sums = np.zeros(len(grid) + 1, dtype=np.float64)
    counts = np.zeros(len(grid) + 1, dtype=np.int64)

    shapes = list(zip(grid.geometry, grid["cell_id"]))

    from rasterio.transform import Affine

    n_chunks = (H + CHUNK_ROWS - 1) // CHUNK_ROWS
    for i in tqdm(range(n_chunks), desc="  chunks"):
        r0 = i * CHUNK_ROWS
        r1 = min(r0 + CHUNK_ROWS, H)

        # Transform for this chunk: shift origin down by r0 rows.
        chunk_transform = transform * Affine.translation(0, r0)

        chunk_ids = rasterize(
            shapes,
            out_shape=(r1 - r0, W),
            transform=chunk_transform,
            fill=0,
            dtype="int32",
        )
        chunk_crop = cropland_arr[r0:r1]

        flat_ids = chunk_ids.ravel()
        flat_crop = chunk_crop.ravel().astype(np.float64)
        valid = flat_ids > 0
        if not valid.any():
            continue

        # Accumulate via np.add.at (handles repeated indices).
        np.add.at(sums, flat_ids[valid], flat_crop[valid])
        np.add.at(counts, flat_ids[valid], 1)

    frac = np.divide(
        sums[1:],
        counts[1:],
        out=np.zeros(len(grid), dtype=np.float64),
        where=counts[1:] > 0,
    )
    grid["cropland_frac"] = frac
    print(f"  Mean cropland fraction: {grid['cropland_frac'].mean():.3f}")
    return grid


def main() -> None:
    aoi = gpd.read_file(AOI_PATH)

    cropland_arr, transform, crop_crs = load_cropland_raster()
    print(f"Cropland raster: {cropland_arr.shape}, CRS {crop_crs}")

    # Project AOI into the cropland raster's CRS (should be EPSG:32642).
    aoi_proj = aoi.to_crs(crop_crs)

    grid = build_grid(aoi_proj)
    grid = tag_district(grid, aoi_proj)
    grid = compute_cropland_frac(grid, cropland_arr, transform)

    before = len(grid)
    grid = grid[grid["cropland_frac"] >= CROPLAND_THRESHOLD].copy()
    print(
        f"\nAfter cropland threshold ({CROPLAND_THRESHOLD}): {len(grid):,} / {before:,} cells"
    )

    # Save.
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    grid[["cell_id", "district", "cropland_frac", "geometry"]].to_file(
        OUT_PATH, driver="GPKG"
    )
    print(f"\nSaved → {OUT_PATH}")

    # Per-district summary.
    print("\nCells per district:")
    for dist, n in grid.groupby("district").size().sort_values(ascending=False).items():
        print(f"  {dist:25s} {n:>7,}")


if __name__ == "__main__":
    main()
