"""
Day 18b — Recovery trajectory archetype discovery via k-means clustering.

For each matched cell, extract its NDVI anomaly trajectory across the 11
seasons used in Day 13 (Kharif 2021 reference through Kharif 2026).
Standardize, cluster with k-means, interpret archetypes, and map them
geographically.

Selection of k by elbow + silhouette; typically 4-6 archetypes emerge.
Each archetype gets interpreted by its mean trajectory shape and
distribution across treated/control + flood_duration terciles.

Outputs:
  - data/processed/trajectory_clusters.parquet (cell_id → cluster_id)
  - results/tables/day18b_cluster_summary.csv (archetype characteristics)
  - results/figures/day18b_archetype_trajectories.png (mean shapes)
  - results/figures/day18b_archetype_map.png (spatial distribution)
"""
from __future__ import annotations
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from src.viz import set_paper_style

REPO     = Path(__file__).resolve().parents[1]
MASTER   = REPO / "data" / "processed" / "master_dataset.parquet"
PANEL    = REPO / "data" / "processed" / "did_panel.parquet"
GRID     = REPO / "data" / "processed" / "analysis_grid.gpkg"
OUT_PARQUET = REPO / "data" / "processed" / "trajectory_clusters.parquet"
TAB      = REPO / "results" / "tables"
FIG_DIR  = REPO / "results" / "figures"

K_RANGE = [3, 4, 5, 6, 7]  # evaluate these k values
RNG_SEED = 42

# Season order matching Day 13.
SEASON_ORDER = [
    "Kharif_2021",   # k=-2 (reference)
    "Rabi_2021-22",  # k=-1
    "Kharif_2022",   # k=0 shock
    "Rabi_2022-23",  # k=+1
    "Kharif_2023",   # k=+2
    "Rabi_2023-24",  # k=+3
    "Kharif_2024",   # k=+4
    "Rabi_2024-25",  # k=+5
    "Kharif_2025",   # k=+6
    "Rabi_2025-26",  # k=+7
    "Kharif_2026",   # k=+8
]


def build_trajectory_matrix() -> tuple[np.ndarray, pd.DataFrame]:
    """
    Build an N × 11 matrix of NDVI anomaly trajectories, one row per
    matched cell. Returns (matrix, metadata_df) aligned by row.
    """
    print("Loading did_panel.parquet (output of Day 13)...")
    panel = pd.read_parquet(PANEL)
    print(f"  {len(panel):,} cell-season rows")

    print("\nPivoting to cell × season matrix...")
    wide = panel.pivot_table(
        index="cell_id",
        columns="season",
        values="ndvi_anom",
        aggfunc="first",
    )
    # Reorder columns to match SEASON_ORDER.
    wide = wide.reindex(columns=SEASON_ORDER)
    print(f"  Pivoted: {wide.shape[0]:,} cells × {wide.shape[1]} seasons")

    # Keep only cells with complete trajectories.
    before = len(wide)
    wide = wide.dropna()
    print(f"  After dropping incomplete trajectories: {len(wide):,} "
          f"(removed {before - len(wide):,})")

    # Attach treatment and district from master.
    master = pd.read_parquet(MASTER)
    meta = master[master["cell_id"].isin(wide.index)][
        ["cell_id", "district", "treated_t3", "flood_duration"]
    ].drop_duplicates("cell_id").set_index("cell_id")
    meta = meta.reindex(wide.index)

    return wide.to_numpy(), meta.reset_index(), wide.columns.tolist()


def select_k(X: np.ndarray) -> int:
    """Evaluate k values by silhouette score (on sample) and inertia (elbow)."""
    # Silhouette on sample for speed.
    sample_size = min(10_000, len(X))
    rng = np.random.default_rng(RNG_SEED)
    sample_idx = rng.choice(len(X), sample_size, replace=False)
    X_sample = X[sample_idx]

    print("\n=== k selection ===")
    results = []
    for k in K_RANGE:
        km = KMeans(n_clusters=k, n_init=10, random_state=RNG_SEED)
        km.fit(X_sample)
        sil = silhouette_score(X_sample, km.labels_, sample_size=5000,
                                 random_state=RNG_SEED)
        inertia = km.inertia_
        results.append({"k": k, "silhouette": sil, "inertia": inertia})
        print(f"  k={k}: silhouette = {sil:.4f}, inertia = {inertia:.0f}")

    results_df = pd.DataFrame(results)

    # Pick k with highest silhouette.
    best_k = int(results_df.loc[results_df["silhouette"].idxmax(), "k"])
    print(f"\n  Selected k = {best_k} (highest silhouette)")
    return best_k, results_df


def fit_kmeans(X: np.ndarray, k: int) -> KMeans:
    """Fit final k-means on full data."""
    print(f"\nFitting final k-means on {len(X):,} cells with k={k}...")
    km = KMeans(n_clusters=k, n_init=20, random_state=RNG_SEED,
                 verbose=0)
    km.fit(X)
    print(f"  Inertia: {km.inertia_:.0f}")
    return km


def characterize_archetypes(X: np.ndarray, labels: np.ndarray,
                              meta: pd.DataFrame,
                              season_labels: list[str]) -> pd.DataFrame:
    """For each cluster: size, treated %, mean flood duration, mean trajectory."""
    rows = []
    for c in sorted(np.unique(labels)):
        mask = labels == c
        traj_mean = X[mask].mean(axis=0)
        traj_std  = X[mask].std(axis=0)
        row = {
            "cluster":        c,
            "n":              int(mask.sum()),
            "pct_total":      float(mask.mean() * 100),
            "pct_treated":    float(meta.loc[mask, "treated_t3"].mean() * 100),
            "mean_flood_dur": float(meta.loc[mask, "flood_duration"].mean()),
            "top_district":   meta.loc[mask, "district"].value_counts().index[0],
        }
        for i, s in enumerate(season_labels):
            row[f"mean_{s}"] = float(traj_mean[i])
            row[f"std_{s}"]  = float(traj_std[i])
        rows.append(row)
    return pd.DataFrame(rows)


def interpret_archetype(summary_row: pd.Series,
                          season_labels: list[str]) -> str:
    """Produce a human-readable label for each cluster from its trajectory shape."""
    traj = [summary_row[f"mean_{s}"] for s in season_labels]
    shock = traj[2]   # Kharif 2022 (k=0)
    final = traj[-1]  # Kharif 2026 (k=8)
    best_post = max(traj[3:])  # best season post-shock

    if abs(shock) < 0.03 and abs(final) < 0.03:
        return "Stable / minimal disruption"
    if shock < -0.1 and final > -0.05:
        return "V-shaped recovery (full)"
    if shock < -0.1 and -0.1 < final < -0.03:
        return "U-shaped recovery (partial)"
    if shock < -0.1 and final < -0.1:
        return "Persistent deficit (incomplete)"
    if shock > 0.03 and final > 0.03:
        return "Above-trend (apparent greening or Rabi over-recovery)"
    if shock < 0 and best_post > 0.05:
        return "Overshoot recovery"
    return "Mixed signal"


def plot_archetype_trajectories(summary: pd.DataFrame,
                                  season_labels: list[str],
                                  out_path: Path) -> None:
    set_paper_style()
    event_times = [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8]
    fig, ax = plt.subplots(figsize=(10, 6))

    cmap = plt.get_cmap("tab10")
    for i, (_, row) in enumerate(summary.iterrows()):
        traj = [row[f"mean_{s}"] for s in season_labels]
        label = (f"Cluster {row['cluster']}: {row['label']} "
                 f"(n={row['n']:,}, {row['pct_treated']:.0f}% treated)")
        ax.plot(event_times, traj, "o-", color=cmap(i), label=label, lw=2)

    ax.axvline(0, color="r", lw=1, alpha=0.4, ls="--", label="Flood (Kharif 2022)")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("Event time (seasons from flood)")
    ax.set_ylabel("Mean NDVI anomaly")
    ax.set_title("Recovery trajectory archetypes")
    ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close()


def plot_archetype_map(cluster_df: pd.DataFrame,
                         summary: pd.DataFrame,
                         out_path: Path) -> None:
    """Spatial map of each cell's archetype assignment."""
    set_paper_style()
    grid = gpd.read_file(GRID, columns=["cell_id", "geometry"])
    merged = grid.merge(cluster_df[["cell_id", "cluster"]], on="cell_id")
    print(f"\nMapping {len(merged):,} cells with archetype assignments...")

    fig, ax = plt.subplots(figsize=(10, 10))
    cmap = plt.get_cmap("tab10")

    for i, (_, srow) in enumerate(summary.iterrows()):
        c = srow["cluster"]
        subset = merged[merged["cluster"] == c]
        subset.plot(ax=ax, color=cmap(i), markersize=0.5, alpha=0.5,
                     label=f"C{c}: {srow['label']}")

    ax.set_title("Spatial distribution of recovery archetypes")
    ax.set_axis_off()
    ax.legend(fontsize=8, loc="lower left", markerscale=10)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close()


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)

    # 1. Build trajectory matrix.
    X_raw, meta, season_labels = build_trajectory_matrix()

    # 2. Standardize (not strictly needed since all values are same unit,
    #    but helps k-means avoid being dominated by any outlier season).
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)

    # 3. Pick k.
    k, k_summary = select_k(X)
    k_summary.to_csv(TAB / "day18b_k_selection.csv", index=False)

    # 4. Fit final.
    km = fit_kmeans(X, k)

    # 5. Characterize archetypes on ORIGINAL (not standardized) trajectory
    #    means for interpretability.
    labels = km.labels_
    summary = characterize_archetypes(X_raw, labels, meta, season_labels)
    summary["label"] = summary.apply(
        lambda r: interpret_archetype(r, season_labels), axis=1
    )
    summary = summary[["cluster", "label", "n", "pct_total", "pct_treated",
                        "mean_flood_dur", "top_district"]
                       + [f"mean_{s}" for s in season_labels]]
    summary.to_csv(TAB / "day18b_cluster_summary.csv", index=False)
    print("\n=== Archetype summary ===")
    with pd.option_context("display.max_columns", None, "display.width", 200,
                            "display.float_format", lambda x: f"{x:>8.3f}"):
        print(summary[["cluster", "label", "n", "pct_treated",
                         "mean_flood_dur", "top_district"]].to_string(index=False))

    # 6. Save per-cell assignments.
    cluster_df = pd.DataFrame({
        "cell_id": meta["cell_id"].values,
        "cluster": labels,
    })
    cluster_df.to_parquet(OUT_PARQUET, index=False)
    print(f"\n  Saved per-cell → {OUT_PARQUET}")

    # 7. Figures.
    traj_path = FIG_DIR / "day18b_archetype_trajectories.png"
    plot_archetype_trajectories(summary, season_labels, traj_path)
    print(f"\n  Saved → {traj_path.name}")

    map_path = FIG_DIR / "day18b_archetype_map.png"
    plot_archetype_map(cluster_df, summary, map_path)
    print(f"  Saved → {map_path.name}")


if __name__ == "__main__":
    main()