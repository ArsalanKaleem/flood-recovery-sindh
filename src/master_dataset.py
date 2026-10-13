"""
Day 10 — Build the master per-cell dataset.

Reads every data stream produced in Days 6-9 (plus the covariates downloaded
today), merges on cell_id, computes distance-to-road and distance-to-drainage
locally, and writes one parquet with ~555k rows × ~22 columns.

Every Day 11+ analysis reads from this file.
"""

from __future__ import annotations
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from tqdm import tqdm

REPO = Path(__file__).resolve().parents[1]
RAW = REPO / "data" / "raw"
PROC = REPO / "data" / "processed"

OUT_PATH = PROC / "master_dataset.parquet"


def load_csvs(folder: Path, prefix: str) -> pd.DataFrame:
    """Load and concat all CSVs under folder matching prefix_*.csv."""
    files = sorted(folder.glob(f"{prefix}*.csv"))
    if not files:
        raise FileNotFoundError(f"No {prefix}*.csv in {folder}")
    frames = [pd.read_csv(f) for f in files]
    return pd.concat(frames, ignore_index=True)


def compute_distance_to(
    cells_proj: gpd.GeoDataFrame, lines_path: Path, col_name: str
) -> pd.Series:
    """
    Nearest-line distance from each cell centroid to any feature in lines_path.
    Returns a Series aligned to cells_proj.index with values in meters.
    Uses geopandas sjoin_nearest, which is fast for this size with a spatial index.
    """
    print(f"  Computing {col_name}...")
    lines = gpd.read_file(lines_path).to_crs(cells_proj.crs)
    lines = lines[lines.geometry.type.isin(["LineString", "MultiLineString"])]

    centroids = cells_proj.copy()
    centroids["geometry"] = centroids.geometry.centroid

    joined = gpd.sjoin_nearest(
        centroids[["cell_id", "geometry"]],
        lines[["geometry"]],
        how="left",
        distance_col=col_name,
    )
    # sjoin_nearest can return duplicates for cells equidistant from 2 lines; dedupe.
    joined = joined.drop_duplicates(subset="cell_id", keep="first")
    return joined.set_index("cell_id")[col_name]


def main() -> None:
    print("=== Loading analysis grid ===")
    grid = gpd.read_file(PROC / "analysis_grid.gpkg")
    print(f"  {len(grid):,} cells")
    master = pd.DataFrame(
        {
            "cell_id": grid["cell_id"].values,
            "district": grid["district"].values,
            "cropland_frac": grid["cropland_frac"].values,
        }
    )

    # --- Primary outcomes (Day 8) ---
    print("\n=== Loading recovery metrics ===")
    rec = pd.read_parquet(PROC / "recovery_metrics.parquet")
    print(f"  {len(rec):,} rows")
    master = master.merge(
        rec[
            [
                "cell_id",
                "ttr_months",
                "ttr_censored",
                "cvd",
                "cvd_months",
                "ccr_pre",
                "ccr_post",
                "ccr_delta",
            ]
        ],
        on="cell_id",
        how="left",
    )

    # --- Soil moisture persistence (Day 9b) ---
    print("\n=== Loading SM persistence ===")
    sm = pd.read_parquet(PROC / "sm_persistence.parquet")
    print(f"  {len(sm):,} rows")
    master = master.merge(sm, on="cell_id", how="left")

    # --- Flood duration (Day 9) ---
    print("\n=== Loading flood duration ===")
    fd = load_csvs(RAW / "flood_duration", "flood_dur_")
    print(f"  {len(fd):,} rows")
    master = master.merge(fd, on="cell_id", how="left")

    # --- LST anomaly (Day 9) ---
    print("\n=== Loading LST anomaly ===")
    lst = load_csvs(RAW / "lst_anomaly", "lst_anom_")
    print(f"  {len(lst):,} rows")
    master = master.merge(lst, on="cell_id", how="left")

    # --- Terrain (Day 10) ---
    print("\n=== Loading terrain (elevation, slope, TWI) ===")
    terrain = load_csvs(RAW / "terrain", "terrain_")
    print(f"  {len(terrain):,} rows")
    master = master.merge(terrain, on="cell_id", how="left")

    # --- Rainfall anomaly (Day 10) ---
    print("\n=== Loading rainfall anomaly ===")
    rain = load_csvs(RAW / "rainfall", "rainfall_")
    print(f"  {len(rain):,} rows")
    master = master.merge(rain, on="cell_id", how="left")

    # --- NDWI/SBI persistence (Day 10) ---
    print("\n=== Loading NDWI persistence ===")
    ndwi = load_csvs(RAW / "ndwi_persistence", "ndwi_persist_")
    print(f"  {len(ndwi):,} rows")
    master = master.merge(ndwi, on="cell_id", how="left")

    print("\n=== Loading SBI persistence ===")
    sbi = load_csvs(RAW / "sbi_persistence", "sbi_persist_")
    print(f"  {len(sbi):,} rows")
    master = master.merge(sbi, on="cell_id", how="left")

    # --- WorldPop (Day 10) ---
    print("\n=== Loading WorldPop ===")
    pop = load_csvs(RAW / "worldpop", "worldpop_")
    print(f"  {len(pop):,} rows")
    master = master.merge(pop, on="cell_id", how="left")

    # --- Distance covariates (local, from OSM + HydroRIVERS) ---
    print("\n=== Computing distance-to-road and distance-to-drainage ===")
    # Project grid to UTM 42N for metric distances.
    cells_proj = grid.to_crs("EPSG:32642")

    road_dist = compute_distance_to(
        cells_proj, RAW / "osm" / "osm_roads.gpkg", "dist_to_road_m"
    )
    master = master.merge(
        road_dist.rename("dist_to_road_m"),
        left_on="cell_id",
        right_index=True,
        how="left",
    )

    # Combine OSM waterways + HydroRIVERS into one drainage layer, compute distance.
    print("  Merging OSM waterways + HydroRIVERS for drainage...")
    water_osm = gpd.read_file(RAW / "osm" / "osm_waterways.gpkg").to_crs("EPSG:32642")
    water_hydro = gpd.read_file(RAW / "osm" / "hydrorivers.gpkg").to_crs("EPSG:32642")
    water_combined = gpd.GeoDataFrame(
        pd.concat(
            [water_osm[["geometry"]], water_hydro[["geometry"]]], ignore_index=True
        ),
        crs="EPSG:32642",
    )
    tmp_path = PROC / "_tmp_drainage.gpkg"
    water_combined.to_file(tmp_path, driver="GPKG")
    drain_dist = compute_distance_to(cells_proj, tmp_path, "dist_to_drainage_m")
    master = master.merge(
        drain_dist.rename("dist_to_drainage_m"),
        left_on="cell_id",
        right_index=True,
        how="left",
    )
    tmp_path.unlink()

    # --- Final cleanup and save ---
    print("\n=== Finalizing ===")
    # Column order for readability.
    cols = [
        "cell_id",
        "district",
        "cropland_frac",
        # Outcomes
        "ttr_months",
        "ttr_censored",
        "cvd",
        "cvd_months",
        "ccr_pre",
        "ccr_post",
        "ccr_delta",
        # Treatment
        "flood_duration",
        # Secondary indicators
        "sm_persist_months",
        "sm_mean_anom_post",
        "lst_anom_K",
        "ndwi_persist_months",
        "sbi_persist_months",
        # Environmental covariates
        "elevation_m",
        "slope_deg",
        "twi_approx",
        "rainfall_anomaly_mm",
        # Context
        "population",
        "dist_to_road_m",
        "dist_to_drainage_m",
    ]
    master = master[cols]

    print(f"Master dataset: {len(master):,} rows × {len(master.columns)} columns")
    print(f"\nMissing values per column:")
    for c in master.columns:
        n = master[c].isna().sum()
        if n:
            print(f"  {c:25s} {n:>7,} ({n/len(master)*100:.1f}%)")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    master.to_parquet(OUT_PATH, index=False)
    print(f"\nSaved → {OUT_PATH} ({OUT_PATH.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
