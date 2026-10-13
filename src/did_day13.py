"""
Day 13 — Event-study DiD on the matched sample.

Panel:   cell × season  (seasons = Kharif May-Oct, Rabi Nov-Apr)
Sample:  matched_pair_id non-null (174,894 cells = 87,447 pairs)
Outcome: NDVI anomaly (primary), EVI anomaly (secondary)
Treat:   treated_t3
Spec:    y_{i,t} = alpha_i + lambda_t + sum_k beta_k * (treated_i * 1[event_time=k])
         where k = −2 is the reference season (Kharif 2021).

Fixed effects: cell (alpha_i), season-year (lambda_t)
SEs: clustered by district
Output: results/day13_did.csv (coefficient table), results/day13_eventstudy.png,
        plus runtime diagnostics on sample integrity.
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
TRAJ_DIR = REPO / "data" / "processed" / "trajectories"
RESULTS = REPO / "results"
FIG_DIR = RESULTS / "figures"
TABLE_DIR = RESULTS / "tables"

PRIMARY_TREAT = "treated_t3"

# Season boundaries: Kharif = May-Oct, Rabi = Nov-Apr.
# "Season label" for a given month:
#   May 2021 - Oct 2021 → Kharif_2021
#   Nov 2021 - Apr 2022 → Rabi_2021-22
#   May 2022 - Oct 2022 → Kharif_2022   (SHOCK)
#   ...
REFERENCE_SEASON = "Kharif_2021"  # event_time = -2
SHOCK_SEASON = "Kharif_2022"  # event_time =  0


def month_to_season(period: pd.Period) -> str:
    """Convert YYYY-MM Period to a season label."""
    y, m = period.year, period.month
    if 5 <= m <= 10:
        return f"Kharif_{y}"
    # Rabi straddles calendar years; label by starting calendar year.
    start_y = y if m >= 11 else y - 1
    return f"Rabi_{start_y}-{str(start_y + 1)[-2:]}"


def event_time(season_label: str) -> int:
    """Integer season offset relative to the SHOCK season (Kharif_2022 = 0)."""
    # Enumerate the sequence of seasons we expect in the data.
    all_seasons = []
    y = 2021
    while y <= 2026:
        all_seasons.append(f"Kharif_{y}")
        all_seasons.append(f"Rabi_{y}-{str(y+1)[-2:]}")
        y += 1
    try:
        return all_seasons.index(season_label) - all_seasons.index(SHOCK_SEASON)
    except ValueError:
        return np.nan


def load_matched_trajectories() -> pd.DataFrame:
    """
    Load trajectory data only for matched cells; return cell × season means
    of NDVI and EVI anomalies.
    """
    master = pd.read_parquet(MASTER)
    matched = master[master["matched_pair_id"].notna()].copy()
    print(
        f"Matched sample: {len(matched):,} cells "
        f"({matched['matched_pair_id'].nunique():,} pairs)"
    )

    matched_cells = set(matched["cell_id"].tolist())

    print("\nLoading trajectory parquets, filtering to matched cells...")
    frames = []
    for f in tqdm(sorted(TRAJ_DIR.glob("trajectory_*.parquet")), desc="  districts"):
        df = pd.read_parquet(f, columns=["cell_id", "month", "ndvi_anom", "evi_anom"])
        df = df[df["cell_id"].isin(matched_cells)]
        if len(df):
            frames.append(df)
    traj = pd.concat(frames, ignore_index=True)
    traj["month"] = pd.PeriodIndex(traj["month"], freq="M")
    print(f"  Loaded {len(traj):,} cell-month rows for matched cells.")

    # Add season label and event time.
    traj["season"] = traj["month"].apply(month_to_season)
    traj["event_time"] = traj["season"].apply(event_time)

    # Aggregate month → season.
    season_df = traj.groupby(["cell_id", "season", "event_time"], as_index=False).agg(
        ndvi_anom=("ndvi_anom", "mean"), evi_anom=("evi_anom", "mean")
    )

    # Winsorize EVI anomaly: a tiny number of cells have EVI blown up by
    # near-zero denominators in cloud-affected pixels. Clip at the 0.1/99.9
    # percentiles (symmetric, ~0.6 either side) so one bad cell doesn't drive
    # the regression. We checked: 99.9% of values fall between -0.58 and +0.66.
    evi_lo, evi_hi = season_df["evi_anom"].quantile([0.001, 0.999])
    n_clipped_evi = (
        (season_df["evi_anom"] < evi_lo) | (season_df["evi_anom"] > evi_hi)
    ).sum()
    season_df["evi_anom"] = season_df["evi_anom"].clip(evi_lo, evi_hi)
    print(
        f"  Winsorized EVI at [{evi_lo:.3f}, {evi_hi:.3f}]: "
        f"{n_clipped_evi:,} rows clipped "
        f"({n_clipped_evi / len(season_df) * 100:.3f}%)"
    )

    # Same for NDVI, out of caution — though diagnostics showed no issues there.
    ndvi_lo, ndvi_hi = season_df["ndvi_anom"].quantile([0.001, 0.999])
    n_clipped_ndvi = (
        (season_df["ndvi_anom"] < ndvi_lo) | (season_df["ndvi_anom"] > ndvi_hi)
    ).sum()
    season_df["ndvi_anom"] = season_df["ndvi_anom"].clip(ndvi_lo, ndvi_hi)
    print(
        f"  Winsorized NDVI at [{ndvi_lo:.3f}, {ndvi_hi:.3f}]: "
        f"{n_clipped_ndvi:,} rows clipped "
        f"({n_clipped_ndvi / len(season_df) * 100:.3f}%)"
    )

    # Merge treatment, matched_pair_id, district.
    season_df = season_df.merge(
        matched[["cell_id", PRIMARY_TREAT, "matched_pair_id", "district"]],
        on="cell_id",
        how="left",
    )
    return season_df


def verify_pair_integrity(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Confirm each matched_pair_id has exactly one treated and one control
    cell in reference season Kharif 2021. Drop pairs failing this check.
    """
    print(f"\n=== Pair integrity check ===")
    ref = panel[panel["season"] == REFERENCE_SEASON]
    print(f"  Rows in {REFERENCE_SEASON}: {len(ref):,}")

    pair_check = ref.groupby("matched_pair_id")[PRIMARY_TREAT].agg(
        n_total="count", n_treated="sum"
    )
    valid_pairs = pair_check[
        (pair_check["n_total"] == 2) & (pair_check["n_treated"] == 1)
    ].index
    bad_pairs = set(pair_check.index) - set(valid_pairs)
    print(f"  Valid pairs (1 treated + 1 control in ref season): {len(valid_pairs):,}")
    print(f"  Dropped pairs: {len(bad_pairs):,}")

    panel = panel[panel["matched_pair_id"].isin(valid_pairs)].copy()
    print(f"  Rows after filter: {len(panel):,}")
    return panel


def fit_event_study(panel: pd.DataFrame, outcome: str):
    """
    Fit event-study DiD:
      y_{i,t} = alpha_i + lambda_t + sum_{k != -2} beta_k * (treated_i * 1[event_time=k])
    SEs clustered by district.
    """
    print(f"\n=== Event-study DiD on {outcome} ===")
    df = panel.dropna(subset=[outcome, "event_time"]).copy()
    df["event_time"] = df["event_time"].astype(int)

    # Reference season event_time = -2. Omit from dummies.
    event_times = sorted([t for t in df["event_time"].unique() if t != -2])

    # Build interaction dummies with formula-safe names.
    # k = -1 → et_m1, k = 0 → et_p0, k = 1 → et_p1, etc.
    def safe_name(k: int) -> str:
        return f"et_m{abs(k)}" if k < 0 else f"et_p{k}"

    for k in event_times:
        df[safe_name(k)] = ((df[PRIMARY_TREAT] == 1) & (df["event_time"] == k)).astype(
            int
        )

    # Need season as a numeric/sortable index. Use event_time directly as time index.
    df["cell_id_idx"] = df["cell_id"].astype(str)
    df["time_idx"] = df["event_time"].astype(int)
    df_indexed = df.set_index(["cell_id_idx", "time_idx"])

    rhs = [safe_name(k) for k in event_times]
    formula = f"{outcome} ~ 1 + " + " + ".join(rhs) + " + EntityEffects + TimeEffects"
    print(f"  Formula: {formula}")
    print(f"  Sample: {len(df_indexed):,} cell-season observations")

    # Cluster SEs by district — align index to the panel.
    district_cluster = df.set_index(["cell_id_idx", "time_idx"])["district"]
    model = PanelOLS.from_formula(formula, data=df_indexed)
    result = model.fit(cov_type="clustered", clusters=district_cluster)

    coefs = []
    for k in event_times:
        name = safe_name(k)
        if name in result.params.index:
            coefs.append(
                {
                    "event_time": k,
                    "coef": result.params[name],
                    "se": result.std_errors[name],
                    "ci_low": result.conf_int().loc[name, "lower"],
                    "ci_high": result.conf_int().loc[name, "upper"],
                    "pvalue": result.pvalues[name],
                }
            )
    # Reference season gets coef = 0 by construction.
    coefs.append(
        {
            "event_time": -2,
            "coef": 0.0,
            "se": 0.0,
            "ci_low": 0.0,
            "ci_high": 0.0,
            "pvalue": np.nan,
        }
    )

    coef_df = pd.DataFrame(coefs).sort_values("event_time").reset_index(drop=True)
    print(f"\n  Pre-trend check (event_time=-1 coefficient):")
    pre = coef_df[coef_df["event_time"] == -1]
    if len(pre):
        p_pre = pre["pvalue"].iloc[0]
        c_pre = pre["coef"].iloc[0]
        se_pre = pre["se"].iloc[0]
        verdict = "PASSES (p > 0.05)" if p_pre > 0.05 else "WARNING: pre-trend p ≤ 0.05"
        print(f"    coef = {c_pre:+.4f} (SE {se_pre:.4f}), p = {p_pre:.3f} → {verdict}")

    return coef_df, result


def plot_event_study(ndvi_coefs: pd.DataFrame, evi_coefs: pd.DataFrame) -> Path:
    """Save event-study plot: coefficients with 95% CI by event time."""
    set_paper_style()
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    for ax, df, label, color in [
        (axes[0], ndvi_coefs, "NDVI anomaly", "#1f77b4"),
        (axes[1], evi_coefs, "EVI anomaly", "#ff7f0e"),
    ]:
        ax.axhline(0, color="k", lw=0.5)
        ax.axvline(
            0, color="r", lw=1.5, alpha=0.5, linestyle="--", label="Flood (Kharif 2022)"
        )
        ax.errorbar(
            df["event_time"],
            df["coef"],
            yerr=[df["coef"] - df["ci_low"], df["ci_high"] - df["coef"]],
            fmt="o",
            color=color,
            capsize=3,
            markersize=5,
            label=label,
        )
        ax.set_ylabel(f"{label}\n(treated − control)")
        ax.legend(loc="lower right", fontsize=8)
        ax.set_xticks(sorted(df["event_time"].unique()))

    axes[1].set_xlabel("Event time (seasons from flood)")
    plt.suptitle(
        "Event-study DiD: cropland vegetation response to the 2022 flood",
        y=1.00,
        fontsize=11,
    )
    plt.tight_layout()

    out = FIG_DIR / "day13_eventstudy.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    return out


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    panel = load_matched_trajectories()
    panel = verify_pair_integrity(panel)

    print(f"\n=== Season composition ===")
    print(panel.groupby("season").size().rename("n_rows").to_string())

    # Save panel for Day 14+ (robustness).
    panel.to_parquet(REPO / "data" / "processed" / "did_panel.parquet", index=False)

    # Fit NDVI (primary).
    ndvi_coefs, ndvi_result = fit_event_study(panel, "ndvi_anom")
    ndvi_coefs.to_csv(TABLE_DIR / "day13_eventstudy_ndvi.csv", index=False)
    print(f"\n  Saved: {TABLE_DIR / 'day13_eventstudy_ndvi.csv'}")

    # Fit EVI (secondary).
    evi_coefs, evi_result = fit_event_study(panel, "evi_anom")
    evi_coefs.to_csv(TABLE_DIR / "day13_eventstudy_evi.csv", index=False)
    print(f"\n  Saved: {TABLE_DIR / 'day13_eventstudy_evi.csv'}")

    # Plot.
    plot_path = plot_event_study(ndvi_coefs, evi_coefs)
    print(f"\n  Saved: {plot_path}")

    print(f"\n=== NDVI event-study coefficients ===")
    with pd.option_context(
        "display.max_columns",
        None,
        "display.width",
        200,
        "display.float_format",
        lambda x: f"{x:>10.4f}",
    ):
        print(ndvi_coefs.to_string(index=False))


if __name__ == "__main__":
    main()
