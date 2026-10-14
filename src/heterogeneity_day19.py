"""
Day 19 — District heterogeneity + policy breakdown.

Three outputs:
  1. Per-district event-study: refit Day 13 model separately for each
     district. Reveals which districts had the deepest shock, fastest
     recovery, strongest persistence. 22 curves + summary table.
  2. Archetype × district cross-tabulation: % of each district's cells
     in each of Day 18b's three archetypes. Highlights policy priorities.
  3. Policy-relevant characterization of the "persistent deficit" archetype:
     where are they, what are their flood/soil characteristics, who lives
     there? Discussion section inputs.
"""

from __future__ import annotations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS
from tqdm import tqdm

from src.viz import set_paper_style

REPO = Path(__file__).resolve().parents[1]
MASTER = REPO / "data" / "processed" / "master_dataset.parquet"
PANEL = REPO / "data" / "processed" / "did_panel.parquet"
CLUSTERS = REPO / "data" / "processed" / "trajectory_clusters.parquet"
TAB = REPO / "results" / "tables"
FIG_DIR = REPO / "results" / "figures"

PRIMARY_TREAT = "treated_t3"

# Cluster labels from Day 18b.
CLUSTER_LABELS = {
    0: "Persistent deficit",
    1: "Overshoot/intensification",
    2: "Resilient recovery",
}


# -----------------------------------------------------------------------------
# Per-district event studies
# -----------------------------------------------------------------------------
def safe_name(k: int) -> str:
    return f"et_m{abs(k)}" if k < 0 else f"et_p{k}"


def fit_eventstudy_for_district(panel: pd.DataFrame, district: str, ref_time: int = -2):
    """Minimal event-study on one district's matched cells."""
    sub = panel[panel["district"] == district].copy()
    if sub["cell_id"].nunique() < 100:
        return None  # too small to fit

    df = sub.dropna(subset=["ndvi_anom", "event_time"]).copy()
    df["event_time"] = df["event_time"].astype(int)
    event_times = sorted(t for t in df["event_time"].unique() if t != ref_time)

    for k in event_times:
        df[safe_name(k)] = ((df[PRIMARY_TREAT] == 1) & (df["event_time"] == k)).astype(
            int
        )

    df["cell_id_idx"] = df["cell_id"].astype(str)
    df["time_idx"] = df["event_time"].astype(int)
    df_i = df.set_index(["cell_id_idx", "time_idx"])

    rhs = [safe_name(k) for k in event_times]
    formula = f"ndvi_anom ~ 1 + " + " + ".join(rhs) + " + EntityEffects + TimeEffects"

    try:
        model = PanelOLS.from_formula(formula, data=df_i)
        result = model.fit(cov_type="robust")  # no clustering: 1 district only
    except Exception:
        return None

    rows = []
    for k in event_times:
        name = safe_name(k)
        if name in result.params.index:
            rows.append(
                {
                    "event_time": k,
                    "coef": result.params[name],
                    "ci_low": result.conf_int().loc[name, "lower"],
                    "ci_high": result.conf_int().loc[name, "upper"],
                }
            )
    rows.append({"event_time": ref_time, "coef": 0.0, "ci_low": 0.0, "ci_high": 0.0})
    return pd.DataFrame(rows).sort_values("event_time").reset_index(drop=True)


def run_per_district_eventstudies(panel: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Fit event-study on each of 22 districts."""
    print("\n=== Per-district event studies ===")
    districts = sorted(panel["district"].dropna().unique())
    results = {}
    for d in tqdm(districts, desc="  districts"):
        res = fit_eventstudy_for_district(panel, d)
        if res is not None:
            results[d] = res
    print(f"  Fit succeeded for {len(results)} of {len(districts)} districts.")
    return results


def summarize_per_district(curves: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Key metrics per district: shock magnitude, final gap, time-to-50%-recovery."""
    rows = []
    for d, curve in curves.items():
        shock = curve.loc[curve["event_time"] == 0, "coef"].iloc[0]
        final = curve.loc[curve["event_time"] == 8, "coef"]
        final = final.iloc[0] if len(final) else np.nan

        # Time-to-50% recovery: first season where coef >= 0.5 * shock (less negative)
        half = shock * 0.5
        ttr50 = np.nan
        post = curve[curve["event_time"] > 0].sort_values("event_time")
        for _, r in post.iterrows():
            if r["coef"] >= half:
                ttr50 = int(r["event_time"])
                break

        rows.append(
            {
                "district": d,
                "shock_k0": shock,
                "final_k8": final,
                "ttr50_seasons": ttr50,
                "recovery_fraction": (1 - final / shock) if shock < 0 else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values("shock_k0")


def plot_district_lattice(
    curves: dict[str, pd.DataFrame], summary: pd.DataFrame, out_path: Path
) -> None:
    set_paper_style()
    districts = summary.sort_values("shock_k0")["district"].tolist()
    n = len(districts)
    ncols = 4
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(14, 2.6 * nrows), sharex=True, sharey=True
    )
    for ax, d in zip(axes.flat, districts):
        c = curves[d]
        ax.plot(c["event_time"], c["coef"], "o-", color="#1f77b4", lw=1.5, markersize=4)
        ax.fill_between(
            c["event_time"], c["ci_low"], c["ci_high"], alpha=0.2, color="#1f77b4"
        )
        ax.axhline(0, color="k", lw=0.4)
        ax.axvline(0, color="r", lw=0.8, alpha=0.4, ls="--")
        shock = summary.loc[summary["district"] == d, "shock_k0"].iloc[0]
        ax.set_title(f"{d} (shock {shock:+.2f})", fontsize=9)
        ax.set_ylim(-0.4, 0.1)
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    fig.supxlabel("Event time (seasons from shock)")
    fig.supylabel("NDVI anomaly (treated − control)")
    plt.suptitle(
        "Per-district event-study curves (sorted by shock magnitude)",
        fontsize=12,
        y=1.00,
    )
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close()


# -----------------------------------------------------------------------------
# Archetype × district cross-tab
# -----------------------------------------------------------------------------
def archetype_by_district(master: pd.DataFrame, clusters: pd.DataFrame) -> pd.DataFrame:
    """For each district, % of cells in each archetype."""
    print("\n=== Archetype × district cross-tabulation ===")
    merged = master.merge(clusters, on="cell_id", how="inner")
    pivot = merged.groupby(["district", "cluster"]).size().unstack(fill_value=0)
    # Convert to percentages per district row.
    pct = pivot.div(pivot.sum(axis=1), axis=0) * 100
    pct.columns = [CLUSTER_LABELS[c] for c in pct.columns]
    pct["n_total"] = pivot.sum(axis=1)
    pct = pct.sort_values("Persistent deficit", ascending=False)
    return pct


def plot_archetype_heatmap(pct: pd.DataFrame, out_path: Path) -> None:
    set_paper_style()
    fig, ax = plt.subplots(figsize=(8, 7))
    data = pct[list(CLUSTER_LABELS.values())].values
    im = ax.imshow(data, aspect="auto", cmap="RdYlGn_r", vmin=0, vmax=70)
    ax.set_yticks(range(len(pct)))
    ax.set_yticklabels(pct.index, fontsize=8)
    ax.set_xticks(range(len(CLUSTER_LABELS)))
    ax.set_xticklabels(
        list(CLUSTER_LABELS.values()), rotation=25, ha="right", fontsize=9
    )
    # Annotate cells.
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(
                j,
                i,
                f"{data[i,j]:.0f}",
                ha="center",
                va="center",
                fontsize=7,
                color="black",
            )
    plt.colorbar(im, ax=ax, label="% of district's matched cells")
    plt.title("Archetype composition by district", fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close()


# -----------------------------------------------------------------------------
# Persistent-deficit policy characterization
# -----------------------------------------------------------------------------
def characterize_persistent_deficit(
    master: pd.DataFrame, clusters: pd.DataFrame
) -> pd.DataFrame:
    """Policy-relevant characterization of the Persistent Deficit cluster."""
    print("\n=== Persistent deficit cluster: policy characterization ===")
    merged = master.merge(clusters, on="cell_id", how="inner")
    pd_cells = merged[merged["cluster"] == 0]

    # District breakdown.
    district_breakdown = (
        pd_cells.groupby("district")
        .agg(
            n_cells=("cell_id", "count"),
            mean_flood_duration=("flood_duration", "mean"),
            mean_dist_drainage=("dist_to_struct_drainage_m", "mean"),
            mean_elevation=("elevation_m", "mean"),
            mean_population=("population", "mean"),
        )
        .sort_values("n_cells", ascending=False)
    )
    district_breakdown["pct_of_persistent_deficit"] = (
        district_breakdown["n_cells"] / len(pd_cells) * 100
    )
    return district_breakdown


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    master = pd.read_parquet(MASTER)
    panel = pd.read_parquet(PANEL)
    clusters = pd.read_parquet(CLUSTERS)
    print(
        f"  Master: {len(master):,}  Panel: {len(panel):,}  Clusters: {len(clusters):,}"
    )

    # 1. Per-district event studies.
    curves = run_per_district_eventstudies(panel)
    summary = summarize_per_district(curves)
    summary.to_csv(TAB / "day19_per_district_summary.csv", index=False)
    print(f"\n  Saved → day19_per_district_summary.csv")

    print("\n  Districts with deepest shock (top 5):")
    print(
        summary.head(5)[
            ["district", "shock_k0", "final_k8", "ttr50_seasons", "recovery_fraction"]
        ].to_string(index=False)
    )
    print("\n  Districts with shallowest shock (bottom 5):")
    print(
        summary.tail(5)[
            ["district", "shock_k0", "final_k8", "ttr50_seasons", "recovery_fraction"]
        ].to_string(index=False)
    )

    # Save all 22 curves as a long table.
    long_rows = []
    for d, c in curves.items():
        c = c.copy()
        c["district"] = d
        long_rows.append(c)
    pd.concat(long_rows, ignore_index=True).to_csv(
        TAB / "day19_per_district_curves.csv", index=False
    )

    plot_district_lattice(curves, summary, FIG_DIR / "day19_district_lattice.png")
    print(f"  Saved → day19_district_lattice.png")

    # 2. Archetype × district cross-tab.
    pct = archetype_by_district(master, clusters)
    pct.to_csv(TAB / "day19_archetype_by_district.csv")
    print(f"  Saved → day19_archetype_by_district.csv")

    print("\n  Districts with highest persistent-deficit share (top 5):")
    print(
        pct.head(5)[
            [
                "Persistent deficit",
                "Overshoot/intensification",
                "Resilient recovery",
                "n_total",
            ]
        ].to_string()
    )

    plot_archetype_heatmap(pct, FIG_DIR / "day19_archetype_heatmap.png")
    print(f"  Saved → day19_archetype_heatmap.png")

    # 3. Persistent deficit policy characterization.
    pd_chars = characterize_persistent_deficit(master, clusters)
    pd_chars.to_csv(TAB / "day19_persistent_deficit_characterization.csv")
    print(f"\n  Saved → day19_persistent_deficit_characterization.csv")

    print("\n  Top 10 districts by persistent-deficit cell count:")
    print(
        pd_chars.head(10)[
            [
                "n_cells",
                "pct_of_persistent_deficit",
                "mean_flood_duration",
                "mean_dist_drainage",
                "mean_elevation",
            ]
        ].to_string()
    )


if __name__ == "__main__":
    main()
