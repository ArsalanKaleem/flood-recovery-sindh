"""
Day 17 — Spatial robustness check on Model A residuals.

Compute Moran's I on the residuals of the TTR mixed-effects model to
test for residual spatial autocorrelation after accounting for the
district random intercept and matching.

If I > 0.1 and significant, we flag spatial dependence and either
switch to SARAR (expensive) or acknowledge that district-level random
intercepts + matching have absorbed most but not all spatial structure.
"""

from __future__ import annotations
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from libpysal.weights import KNN
from esda.moran import Moran

REPO = Path(__file__).resolve().parents[1]
MASTER = REPO / "data" / "processed" / "master_dataset.parquet"
GRID = REPO / "data" / "processed" / "analysis_grid.gpkg"
TAB = REPO / "results" / "tables"

K_NEIGHBORS = 8  # standard choice for grid-like data


def main() -> None:
    print("Loading data and refitting Model A to extract residuals...")
    master = pd.read_parquet(MASTER)
    df = master[master["matched_pair_id"].notna()].copy()

    # Replicate Model A transformations.
    df["log_dist_drainage"] = np.log(df["dist_to_struct_drainage_m"].clip(lower=1))
    df["log_dist_road"] = np.log(df["dist_to_road_m"].clip(lower=1))
    df["log_pop"] = np.log(df["population"].clip(lower=0) + 1)
    cols_to_std = [
        "flood_duration",
        "sm_persist_months",
        "sbi_persist_months",
        "ndwi_persist_months",
        "lst_anom_K",
        "elevation_m",
        "slope_deg",
        "log_dist_drainage",
        "log_dist_road",
        "rainfall_anomaly_mm",
        "log_pop",
    ]
    for c in cols_to_std:
        df[f"z_{c}"] = (df[c] - df[c].mean()) / df[c].std()
    reg_cols = [f"z_{c}" for c in cols_to_std] + ["ttr_months", "district"]
    df_clean = df.dropna(subset=reg_cols).copy()

    formula = (
        "ttr_months ~ z_flood_duration + z_sm_persist_months + z_sbi_persist_months "
        "+ z_ndwi_persist_months + z_lst_anom_K + z_elevation_m + z_slope_deg "
        "+ z_log_dist_drainage + z_log_dist_road "
        "+ z_rainfall_anomaly_mm + z_log_pop"
    )
    model = smf.mixedlm(formula, df_clean, groups=df_clean["district"])
    result = model.fit(method="lbfgs")
    df_clean["residual"] = result.resid.values
    print(f"  Model refit; {len(df_clean):,} residuals extracted.")

    print(f"\nLoading analysis grid for cell geometry...")
    grid = gpd.read_file(GRID, columns=["cell_id", "geometry"])
    grid["centroid"] = grid.geometry.centroid
    centroids = gpd.GeoDataFrame(
        grid[["cell_id"]], geometry=grid["centroid"], crs=grid.crs
    ).to_crs("EPSG:32642")

    merged = centroids.merge(df_clean[["cell_id", "residual"]], on="cell_id")
    print(f"  Cells with residuals + geometry: {len(merged):,}")

    # Downsample for KNN tractability: Moran's I on 174k points with K=8
    # requires a 174k × 174k sparse weights matrix, which is fine, but the
    # neighbor computation itself is slow. Random sample for stability.
    SAMPLE_SIZE = 25_000
    if len(merged) > SAMPLE_SIZE:
        print(f"\nRandom-sampling {SAMPLE_SIZE:,} cells for Moran's I computation...")
        sample = merged.sample(SAMPLE_SIZE, random_state=42)
    else:
        sample = merged

    print(f"\nBuilding K={K_NEIGHBORS} nearest-neighbor weights...")
    coords = np.column_stack([sample.geometry.x.values, sample.geometry.y.values])
    w = KNN.from_array(coords, k=K_NEIGHBORS)
    w.transform = "r"  # row-standardize

    print(f"\nComputing Moran's I on residuals...")
    mi = Moran(sample["residual"].values, w, permutations=99)
    print(f"\n  Moran's I = {mi.I:.4f}")
    print(f"  Expected I under null = {mi.EI:.6f}")
    print(f"  z-score (standard-normal) = {mi.z_norm:.2f}")
    print(f"  p-value (one-sided, permutation) = {mi.p_sim:.4f}")

    # Interpret.
    if mi.I < 0.05:
        verdict = "NEGLIGIBLE spatial autocorrelation (I < 0.05)"
    elif mi.I < 0.15:
        verdict = (
            "WEAK spatial autocorrelation (0.05 ≤ I < 0.15); district FE sufficient"
        )
    elif mi.I < 0.3:
        verdict = "MODERATE spatial autocorrelation (0.15 ≤ I < 0.3); consider SARAR"
    else:
        verdict = "STRONG spatial autocorrelation (I ≥ 0.3); SARAR warranted"
    print(f"\n  Verdict: {verdict}")

    out = pd.DataFrame(
        [
            {
                "sample_size": len(sample),
                "k_neighbors": K_NEIGHBORS,
                "moran_I": mi.I,
                "expected_I": mi.EI,
                "z_score": mi.z_norm,
                "p_permutation": mi.p_sim,
                "verdict": verdict,
            }
        ]
    )
    out.to_csv(TAB / "day17_moran.csv", index=False)
    print(f"\n  Saved → day17_moran.csv")


if __name__ == "__main__":
    main()
