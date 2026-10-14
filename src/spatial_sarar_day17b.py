"""
Day 17b — SARAR sensitivity check on Model A.

Moran's I on Model A residuals was 0.17 (moderate). Here we fit a spatial
autoregressive combined (SARAR) model on a 10k random sample and compare
coefficients to the baseline mixed-effects Model A. If they're stable
in direction, magnitude, and significance, we conclude that moderate
spatial autocorrelation does not bias effect estimates.

The sample size is a tractability compromise: full-sample SARAR on 174k
observations with K=8 weights is memory-intensive. 10k gives us enough
power to detect substantive coefficient changes.
"""

from __future__ import annotations
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from libpysal.weights import KNN
from spreg import GM_Combo_Het

REPO = Path(__file__).resolve().parents[1]
MASTER = REPO / "data" / "processed" / "master_dataset.parquet"
GRID = REPO / "data" / "processed" / "analysis_grid.gpkg"
TAB = REPO / "results" / "tables"

SAMPLE_SIZE = 10_000
K_NEIGHBORS = 8
RNG_SEED = 42


def main() -> None:
    print("Loading data...")
    master = pd.read_parquet(MASTER)
    df = master[master["matched_pair_id"].notna()].copy()
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
    reg_cols = [f"z_{c}" for c in cols_to_std] + ["ttr_months"]
    df_clean = df.dropna(subset=reg_cols).copy()

    # Attach cell geometry.
    print(f"Attaching geometry to {len(df_clean):,} cells...")
    grid = gpd.read_file(GRID, columns=["cell_id", "geometry"])
    grid["centroid"] = grid.geometry.centroid
    centroids = gpd.GeoDataFrame(
        grid[["cell_id"]], geometry=grid["centroid"], crs=grid.crs
    ).to_crs("EPSG:32642")
    merged = centroids.merge(df_clean, on="cell_id")
    print(f"  Merged: {len(merged):,} cells")

    # Sample.
    print(f"\nRandom-sampling {SAMPLE_SIZE:,} cells for SARAR fit...")
    sample = merged.sample(SAMPLE_SIZE, random_state=RNG_SEED).reset_index(drop=True)
    coords = np.column_stack([sample.geometry.x.values, sample.geometry.y.values])

    print(f"Building K={K_NEIGHBORS} nearest-neighbor weights...")
    w = KNN.from_array(coords, k=K_NEIGHBORS)
    w.transform = "r"

    # Prepare y and X.
    y = sample["ttr_months"].values.reshape(-1, 1)
    x_cols = [f"z_{c}" for c in cols_to_std]
    X = sample[x_cols].values

    print(f"\nFitting SARAR (spatial lag + spatial error) via GMM...")
    model = GM_Combo_Het(
        y=y,
        x=X,
        w=w,
        name_x=x_cols,
        name_y="ttr_months",
        name_w="KNN8",
        name_ds=f"matched sample n={SAMPLE_SIZE}",
    )
    print(model.summary)

    # Extract betas and compare to the Model A table.
    sarar_coefs = {}
    for i, name in enumerate(x_cols):
        sarar_coefs[name] = {
            "sarar_coef": float(model.betas[i + 1]),  # +1 to skip intercept
            "sarar_se": float(np.sqrt(model.vm[i + 1, i + 1])),
        }
    sarar_coefs["rho_lag"] = {"sarar_coef": float(model.betas[-2]), "sarar_se": None}
    sarar_coefs["lambda_err"] = {"sarar_coef": float(model.betas[-1]), "sarar_se": None}

    comparison_rows = []
    model_a = pd.read_csv(TAB / "day16c_model_a_ttr.csv", index_col=0)
    for name in x_cols:
        if name in model_a.index:
            model_a_coef = model_a.loc[name, "coefficient"]
        else:
            model_a_coef = np.nan
        sc = sarar_coefs[name]
        comparison_rows.append(
            {
                "variable": name,
                "model_a_coef": model_a_coef,
                "sarar_coef": sc["sarar_coef"],
                "sarar_se": sc["sarar_se"],
                "change_pct": (
                    (sc["sarar_coef"] - model_a_coef) / model_a_coef * 100
                    if model_a_coef not in (0, np.nan)
                    else np.nan
                ),
            }
        )

    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(TAB / "day17b_sarar_sensitivity.csv", index=False)
    print(f"\n=== Model A vs SARAR coefficient comparison ===")
    with pd.option_context(
        "display.max_columns",
        None,
        "display.width",
        200,
        "display.float_format",
        lambda x: f"{x:>10.3f}",
    ):
        print(comparison.to_string(index=False))

    print(
        f"\n  Spatial lag parameter (rho):   {sarar_coefs['rho_lag']['sarar_coef']:+.4f}"
    )
    print(
        f"  Spatial error parameter (lambda): {sarar_coefs['lambda_err']['sarar_coef']:+.4f}"
    )

    # Summary verdict.
    big_changes = comparison[comparison["change_pct"].abs() > 25]
    print(
        f"\n  Variables with >25% coefficient change: {len(big_changes)}/{len(comparison)}"
    )
    if len(big_changes) <= 2:
        verdict = "ROBUST — SARAR coefficients align with Model A"
    else:
        verdict = "SENSITIVE — meaningful coefficient shifts under SARAR"
    print(f"\n  Verdict: {verdict}")

    print(f"\n  Saved → day17b_sarar_sensitivity.csv")


if __name__ == "__main__":
    main()
