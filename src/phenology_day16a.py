"""
Day 16a — Per-cell phenological decomposition of each growing season.

For every (cell, season) pair, compute four metrics from the monthly
NDVI trajectory:
  amplitude          — max NDVI minus min NDVI within the season
  peak_month         — month index (1-6) of the within-season NDVI maximum
  integrated_ndvi    — sum of monthly NDVI across the season (area under curve)
  los_months         — months where NDVI exceeds the cell's annual mean

Then compare treated vs matched control cells season-by-season.

Apparent-greening test:
  If treated cells have similar integrated_ndvi but LOWER amplitude
  or shifted peak_month, their vegetation is not crop-cycle-shaped —
  that's apparent greening within cropland. If treated cells have
  lower integrated_ndvi AND lower amplitude, that's real crop deficit.
  Both outcomes support the paper's main story but through different
  mechanisms.
"""

from __future__ import annotations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.viz import set_paper_style

REPO = Path(__file__).resolve().parents[1]
MASTER = REPO / "data" / "processed" / "master_dataset.parquet"
TRAJ_DIR = REPO / "data" / "processed" / "trajectories"
OUT_PATH = REPO / "data" / "processed" / "phenology.parquet"
FIG_DIR = REPO / "results" / "figures"
TAB_DIR = REPO / "results" / "tables"

PRIMARY_TREAT = "treated_t3"

# Kharif = May-Oct, Rabi = Nov-Apr. For a season label YYYY, the start
# calendar month is 5 (Kharif) or 11 (Rabi, straddling into Y+1).
KHARIF_MONTHS = [5, 6, 7, 8, 9, 10]  # 6 months
RABI_MONTHS = [11, 12, 1, 2, 3, 4]  # 6 months


def _per_season_metrics(group: pd.DataFrame) -> pd.Series:
    """Compute 4 phenological metrics from monthly NDVI series within one season."""
    ndvi = group["ndvi"].dropna()
    if len(ndvi) < 3:
        return pd.Series(
            {
                "amplitude": np.nan,
                "peak_month_idx": np.nan,
                "integrated_ndvi": np.nan,
                "los_months": np.nan,
            }
        )
    amplitude = ndvi.max() - ndvi.min()
    peak_month_idx = int(np.argmax(ndvi.to_numpy())) + 1  # 1..6 within season
    integrated = ndvi.sum()
    mean_ndvi = ndvi.mean()
    los = int((ndvi > mean_ndvi).sum())
    return pd.Series(
        {
            "amplitude": amplitude,
            "peak_month_idx": peak_month_idx,
            "integrated_ndvi": integrated,
            "los_months": los,
        }
    )


def _label_season(period: pd.Period) -> str | None:
    """Return season label (Kharif_YYYY or Rabi_YYYY-YY) or None if out of range."""
    y, m = period.year, period.month
    if m in KHARIF_MONTHS:
        return f"Kharif_{y}"
    if m == 11 or m == 12:
        return f"Rabi_{y}-{str(y+1)[-2:]}"
    if m in (1, 2, 3, 4):
        return f"Rabi_{y-1}-{str(y)[-2:]}"
    return None


def process_district(parquet_path: Path) -> pd.DataFrame:
    """Compute per-(cell, season) phenology for one district."""
    df = pd.read_parquet(parquet_path, columns=["cell_id", "month", "ndvi"])
    df["month"] = pd.PeriodIndex(df["month"], freq="M")
    df["season"] = df["month"].apply(_label_season)
    df = df.dropna(subset=["season"])

    # Sort within (cell, season) by month so peak_month_idx is meaningful.
    df = df.sort_values(["cell_id", "season", "month"])

    out = (
        df.groupby(["cell_id", "season"], as_index=False)
        .apply(_per_season_metrics, include_groups=False)
        .reset_index()
    )
    return out


def build_phenology() -> pd.DataFrame:
    files = sorted(TRAJ_DIR.glob("trajectory_*.parquet"))
    frames = []
    for f in tqdm(files, desc="districts"):
        frames.append(process_district(f))
    return pd.concat(frames, ignore_index=True)


def compare_treated_vs_control(
    phen: pd.DataFrame, master: pd.DataFrame
) -> pd.DataFrame:
    """
    For each season, compare mean amplitude / integrated_ndvi / etc.
    between treated and matched control cells.
    """
    matched = master[master["matched_pair_id"].notna()][
        ["cell_id", PRIMARY_TREAT, "matched_pair_id"]
    ]
    merged = phen.merge(matched, on="cell_id", how="inner")

    results = []
    for season in sorted(merged["season"].unique()):
        s = merged[merged["season"] == season]
        t = s[s[PRIMARY_TREAT] == 1]
        c = s[s[PRIMARY_TREAT] == 0]
        for metric in ["amplitude", "peak_month_idx", "integrated_ndvi", "los_months"]:
            mt = t[metric].mean()
            mc = c[metric].mean()
            # Simple t-stat (not clustered — this is diagnostic, not inference).
            diff = mt - mc
            se = np.sqrt(t[metric].var() / len(t) + c[metric].var() / len(c))
            results.append(
                {
                    "season": season,
                    "metric": metric,
                    "mean_treated": mt,
                    "mean_control": mc,
                    "diff": diff,
                    "diff_se": se,
                    "t_stat": diff / se if se > 0 else np.nan,
                }
            )
    return pd.DataFrame(results)


def plot_phenology_comparison(comparison: pd.DataFrame) -> Path:
    set_paper_style()
    metrics = ["amplitude", "integrated_ndvi", "peak_month_idx", "los_months"]
    titles = [
        "Within-season amplitude\n(max NDVI − min NDVI)",
        "Integrated greenness\n(sum of monthly NDVI)",
        "Peak month index\n(1 = first month of season)",
        "Length of growing season\n(months above cell annual mean)",
    ]

    seasons = sorted(comparison["season"].unique())
    season_order = sorted(
        seasons, key=lambda s: (s.split("_")[1], 0 if s.startswith("Kharif") else 1)
    )

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    for ax, metric, title in zip(axes.flat, metrics, titles):
        sub = (
            comparison[comparison["metric"] == metric]
            .set_index("season")
            .loc[season_order]
        )
        x = np.arange(len(season_order))
        ax.plot(
            x,
            sub["mean_treated"],
            "o-",
            color="#d62728",
            label="Treated (flooded)",
            lw=2,
        )
        ax.plot(
            x, sub["mean_control"], "o-", color="#1f77b4", label="Matched control", lw=2
        )
        ax.set_xticks(x)
        ax.set_xticklabels(season_order, rotation=45, ha="right", fontsize=7)
        ax.set_title(title, fontsize=10)
        ax.axvline(
            season_order.index("Kharif_2022"), color="r", lw=1, alpha=0.4, ls="--"
        )
        ax.legend(fontsize=8)
    plt.suptitle(
        "Phenological decomposition: resolving apparent greening within cropland",
        fontsize=11,
    )
    plt.tight_layout()
    out = FIG_DIR / "day16a_phenology.png"
    plt.savefig(out, bbox_inches="tight", dpi=150)
    plt.close()
    return out


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TAB_DIR.mkdir(parents=True, exist_ok=True)

    print("Computing phenological decomposition per (cell, season)...")
    phen = build_phenology()
    phen.to_parquet(OUT_PATH, index=False)
    print(f"  Saved {len(phen):,} rows → {OUT_PATH}")

    print("\nComparing treated vs matched control...")
    master = pd.read_parquet(MASTER)
    comparison = compare_treated_vs_control(phen, master)
    comparison.to_csv(TAB_DIR / "day16a_phenology_comparison.csv", index=False)
    print(f"  Saved comparison table → day16a_phenology_comparison.csv")

    print("\n=== Phenology summary (apparent-greening test) ===")
    # For each post-flood season, show the key diagnostic.
    post_seasons = [
        s
        for s in sorted(comparison["season"].unique())
        if s.split("_")[1] >= "2022" and s != "Kharif_2021"
    ]
    for season in post_seasons:
        sub = comparison[comparison["season"] == season].set_index("metric")
        amp = sub.loc["amplitude", "diff"]
        intg = sub.loc["integrated_ndvi", "diff"]
        verdict = ""
        if intg < -0.05 and amp < -0.02:
            verdict = "real crop deficit (both lower)"
        elif intg >= -0.05 and amp < -0.02:
            verdict = "apparent greening (amp lower, integrated similar)"
        elif intg < -0.05 and amp >= -0.02:
            verdict = "mixed"
        else:
            verdict = "indistinguishable"
        print(
            f"  {season:15s} integrated_diff={intg:+.3f}  amp_diff={amp:+.3f}  → {verdict}"
        )

    plot_path = plot_phenology_comparison(comparison)
    print(f"\nSaved plot → {plot_path}")


if __name__ == "__main__":
    main()
