"""
Day 14 — Robustness checks on the Day 13 event-study DiD.

14a Placebo test:        pretend flood struck in Kharif 2021 → coefficients should be ~0
14b Caliper sensitivity: re-match at 0.5× and 2× caliper, refit → same recovery pattern expected
14c Dose-response:       split treated into flood-duration terciles, refit → monotonic scaling expected
14d Leave-one-out:       drop each district, refit → no single district should drive result

Day 13 is PRIMARY. Day 14 is sensitivity reporting. We save all outputs
so the paper can report "across all checks, the pattern is stable" or
flag specific exceptions honestly.
"""

from __future__ import annotations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS
from scipy.spatial import cKDTree
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from src.viz import set_paper_style

REPO = Path(__file__).resolve().parents[1]
MASTER = REPO / "data" / "processed" / "master_dataset.parquet"
PANEL = REPO / "data" / "processed" / "did_panel.parquet"
TRAJ_DIR = REPO / "data" / "processed" / "trajectories"
FIG_DIR = REPO / "results" / "figures"
TABLE_DIR = REPO / "results" / "tables"

PRIMARY_TREAT = "treated_t3"
MATCH_COVARIATES = [
    "elevation_m",
    "slope_deg",
    "twi_approx",
    "dist_to_drainage_m",
    "cropland_frac",
    "preflood_ndvi_level",
    "preflood_ndvi_trend",
]


# -----------------------------------------------------------------------------
# Shared machinery
# -----------------------------------------------------------------------------
def event_time_from_season(season: str, shock_season: str) -> int:
    all_seasons = []
    y = 2021
    while y <= 2026:
        all_seasons.append(f"Kharif_{y}")
        all_seasons.append(f"Rabi_{y}-{str(y+1)[-2:]}")
        y += 1
    try:
        return all_seasons.index(season) - all_seasons.index(shock_season)
    except ValueError:
        return np.nan


def safe_name(k: int) -> str:
    return f"et_m{abs(k)}" if k < 0 else f"et_p{k}"


def fit_eventstudy(
    panel: pd.DataFrame, outcome: str, treat_col: str, ref_time: int = -2
) -> pd.DataFrame:
    """Minimal event-study fit; returns coefficient dataframe."""
    df = panel.dropna(subset=[outcome, "event_time"]).copy()
    df["event_time"] = df["event_time"].astype(int)
    event_times = sorted(t for t in df["event_time"].unique() if t != ref_time)

    for k in event_times:
        df[safe_name(k)] = ((df[treat_col] == 1) & (df["event_time"] == k)).astype(int)

    df["cell_id_idx"] = df["cell_id"].astype(str)
    df["time_idx"] = df["event_time"].astype(int)
    df_indexed = df.set_index(["cell_id_idx", "time_idx"])

    rhs = [safe_name(k) for k in event_times]
    formula = f"{outcome} ~ 1 + " + " + ".join(rhs) + " + EntityEffects + TimeEffects"

    cluster = df.set_index(["cell_id_idx", "time_idx"])["district"]
    model = PanelOLS.from_formula(formula, data=df_indexed)
    result = model.fit(cov_type="clustered", clusters=cluster)

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
                    "pvalue": result.pvalues[name],
                }
            )
    rows.append(
        {
            "event_time": ref_time,
            "coef": 0.0,
            "ci_low": 0.0,
            "ci_high": 0.0,
            "pvalue": np.nan,
        }
    )
    return pd.DataFrame(rows).sort_values("event_time").reset_index(drop=True)


# -----------------------------------------------------------------------------
# 14a Placebo
# -----------------------------------------------------------------------------
def run_placebo(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Fake shock season = Kharif_2021. Only use PRE-REAL-FLOOD data
    (Kharif 2021 and Rabi 2021-22). Reference = Kharif 2021 is the fake
    shock, so we use the one pre-shock season (Rabi 2020-21 — unavailable),
    forcing us to flip: Kharif 2021 is fake shock k=0, Rabi 2021-22 is
    fake post k=1. There's no fake-pre season. We instead report whether
    the k=1 (fake post) coefficient is distinguishable from zero.
    """
    print("\n=== 14a Placebo test ===")
    pre = panel[panel["season"].isin(["Kharif_2021", "Rabi_2021-22"])].copy()
    pre["event_time"] = pre["season"].apply(lambda s: 0 if s == "Kharif_2021" else 1)
    print(f"  Fake-shock sample: {len(pre):,} rows")

    df = pre.dropna(subset=["ndvi_anom"]).copy()
    df["treat_x_post"] = ((df[PRIMARY_TREAT] == 1) & (df["event_time"] == 1)).astype(
        int
    )

    df["cell_id_idx"] = df["cell_id"].astype(str)
    df["time_idx"] = df["event_time"].astype(int)
    df_i = df.set_index(["cell_id_idx", "time_idx"])
    cluster = df.set_index(["cell_id_idx", "time_idx"])["district"]

    model = PanelOLS.from_formula(
        "ndvi_anom ~ 1 + treat_x_post + EntityEffects + TimeEffects",
        data=df_i,
    )
    result = model.fit(cov_type="clustered", clusters=cluster)
    coef = result.params["treat_x_post"]
    se = result.std_errors["treat_x_post"]
    p = result.pvalues["treat_x_post"]
    print(f"  Placebo DiD coefficient: {coef:+.4f} (SE {se:.4f}), p = {p:.3f}")
    verdict = "PASSES (p > 0.05)" if p > 0.05 else "FAILS (p ≤ 0.05)"
    print(f"  Interpretation: {verdict}")
    return pd.DataFrame([{"test": "placebo", "coef": coef, "se": se, "pvalue": p}])


# -----------------------------------------------------------------------------
# 14b Caliper sensitivity
# -----------------------------------------------------------------------------
def rematch_at_caliper(master: pd.DataFrame, caliper_mult: float) -> pd.Series:
    """Re-match on a different caliper multiplier of SD(logit propensity)."""
    sub = master.dropna(subset=MATCH_COVARIATES + [PRIMARY_TREAT, "propensity"]).copy()
    p = np.clip(sub["propensity"].to_numpy(), 1e-6, 1 - 1e-6)
    sub["logit_p"] = np.log(p / (1 - p))
    caliper = caliper_mult * 0.2 * sub["logit_p"].std()  # 0.2 * SD is the base

    treated = sub[sub[PRIMARY_TREAT] == 1].sample(frac=1, random_state=42)
    control = sub[sub[PRIMARY_TREAT] == 0].copy()
    ctrl_logit = control["logit_p"].to_numpy().reshape(-1, 1)
    tree = cKDTree(ctrl_logit)
    used = np.zeros(len(control), dtype=bool)

    pair_id = pd.Series(np.nan, index=master.index)
    pair_counter = 0
    master_pos = {idx: i for i, idx in enumerate(master.index)}

    for t_idx, t_row in treated.iterrows():
        tlog = t_row["logit_p"]
        k = 10
        found = -1
        while k <= len(control):
            dists, idxs = tree.query([[tlog]], k=min(k, len(control)))
            for d, i in zip(dists[0], idxs[0]):
                if used[i]:
                    continue
                if d > caliper:
                    break
                found = i
                break
            if found >= 0:
                break
            if dists[0][-1] > caliper:
                break
            k *= 2

        if found >= 0:
            used[found] = True
            c_idx = control.index[found]
            pair_counter += 1
            pair_id.iat[master_pos[t_idx]] = pair_counter
            pair_id.iat[master_pos[c_idx]] = pair_counter

    return pair_id, pair_counter, caliper


def run_caliper_sensitivity(
    master: pd.DataFrame, base_panel_full: pd.DataFrame
) -> dict[str, pd.DataFrame]:
    """Re-match at 0.5× and 2× base caliper, re-fit event studies."""
    print("\n=== 14b Caliper sensitivity ===")
    curves = {}
    for mult, label in [(1.0, "baseline"), (0.5, "0.5x"), (2.0, "2x")]:
        print(f"\n  Caliper {label}:")
        pair_id, n_pairs, cal = rematch_at_caliper(master, mult)
        print(f"    caliper = {cal:.4f}, matched pairs = {n_pairs:,}")

        matched_cells = set(master.loc[pair_id.notna(), "cell_id"])
        panel = base_panel_full[base_panel_full["cell_id"].isin(matched_cells)].copy()
        coefs = fit_eventstudy(panel, "ndvi_anom", PRIMARY_TREAT)
        coefs["caliper"] = label
        coefs["n_pairs"] = n_pairs
        curves[label] = coefs
    return curves


# -----------------------------------------------------------------------------
# 14c Dose-response by flood_duration tercile
# -----------------------------------------------------------------------------
def run_dose_response(
    master: pd.DataFrame, base_panel: pd.DataFrame
) -> dict[str, pd.DataFrame]:
    """Split treated cells into flood_duration terciles; refit each vs. matched controls."""
    print("\n=== 14c Dose-response (flood_duration terciles) ===")

    matched = master[master["matched_pair_id"].notna()].copy()
    treated = matched[matched[PRIMARY_TREAT] == 1].copy()
    terciles = treated["flood_duration"].quantile([1 / 3, 2 / 3])
    print(
        f"  Flood duration tercile cutoffs: {terciles.iloc[0]:.1f}, {terciles.iloc[1]:.1f}"
    )

    treated["dose"] = pd.cut(
        treated["flood_duration"],
        bins=[-np.inf, terciles.iloc[0], terciles.iloc[1], np.inf],
        labels=["low", "mid", "high"],
    )

    # For each treated cell's dose, keep its matched control via matched_pair_id.
    dose_map = treated.set_index("matched_pair_id")["dose"].to_dict()
    matched["dose"] = matched["matched_pair_id"].map(dose_map)

    curves = {}
    for dose_label in ["low", "mid", "high"]:
        subset_cells = set(matched.loc[matched["dose"] == dose_label, "cell_id"])
        n_pairs = matched.loc[
            (matched["dose"] == dose_label) & (matched[PRIMARY_TREAT] == 1),
            "matched_pair_id",
        ].nunique()
        print(f"  {dose_label}: {n_pairs:,} pairs")

        panel_sub = base_panel[base_panel["cell_id"].isin(subset_cells)].copy()
        coefs = fit_eventstudy(panel_sub, "ndvi_anom", PRIMARY_TREAT)
        coefs["dose"] = dose_label
        coefs["n_pairs"] = n_pairs
        curves[dose_label] = coefs
    return curves


# -----------------------------------------------------------------------------
# 14d Leave-one-district-out
# -----------------------------------------------------------------------------
def run_leave_one_out(base_panel: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Fit the main event study 22 times, dropping each district in turn."""
    print("\n=== 14d Leave-one-district-out ===")
    districts = sorted(base_panel["district"].dropna().unique().tolist())
    curves = {}
    for d in tqdm(districts, desc="  districts dropped"):
        sub = base_panel[base_panel["district"] != d].copy()
        try:
            coefs = fit_eventstudy(sub, "ndvi_anom", PRIMARY_TREAT)
            coefs["dropped"] = d
            curves[d] = coefs
        except Exception as e:
            print(f"    Dropped {d}: fit failed ({e})")
    return curves


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------
def plot_all(
    primary: pd.DataFrame,
    placebo: pd.DataFrame,
    caliper_curves: dict,
    dose_curves: dict,
    loo_curves: dict,
) -> Path:
    set_paper_style()
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharey=True)

    # Load Day 13 NDVI for the "primary" trace shown alongside each check.
    main = pd.read_csv(TABLE_DIR / "day13_eventstudy_ndvi.csv")

    # (0,0) Caliper sensitivity
    ax = axes[0, 0]
    for label, coefs in caliper_curves.items():
        ax.errorbar(
            coefs["event_time"],
            coefs["coef"],
            yerr=[coefs["coef"] - coefs["ci_low"], coefs["ci_high"] - coefs["coef"]],
            label=f"caliper {label} (n={coefs['n_pairs'].iloc[0]:,})",
            capsize=2,
            alpha=0.8,
            markersize=4,
            marker="o",
        )
    ax.set_title("14b Caliper sensitivity")
    ax.axvline(0, color="r", lw=1, alpha=0.4, ls="--")
    ax.axhline(0, color="k", lw=0.5)
    ax.legend(fontsize=7)

    # (0,1) Dose-response
    ax = axes[0, 1]
    for label, coefs in dose_curves.items():
        ax.errorbar(
            coefs["event_time"],
            coefs["coef"],
            yerr=[coefs["coef"] - coefs["ci_low"], coefs["ci_high"] - coefs["coef"]],
            label=f"{label} dose (n={coefs['n_pairs'].iloc[0]:,})",
            capsize=2,
            alpha=0.8,
            markersize=4,
            marker="o",
        )
    ax.set_title("14c Dose-response (flood duration terciles)")
    ax.axvline(0, color="r", lw=1, alpha=0.4, ls="--")
    ax.axhline(0, color="k", lw=0.5)
    ax.legend(fontsize=7)

    # (1,0) Leave-one-out (plot all 22 curves lightly, overlay primary)
    ax = axes[1, 0]
    for d, coefs in loo_curves.items():
        ax.plot(coefs["event_time"], coefs["coef"], color="gray", alpha=0.3, lw=0.8)
    ax.plot(
        main["event_time"],
        main["coef"],
        color="#1f77b4",
        lw=2,
        label="Primary (Day 13)",
    )
    ax.set_title("14d Leave-one-district-out (gray = each LOO fit)")
    ax.axvline(0, color="r", lw=1, alpha=0.4, ls="--")
    ax.axhline(0, color="k", lw=0.5)
    ax.legend(fontsize=7)

    # (1,1) Placebo (single DiD coefficient, bar chart alongside primary shock)
    ax = axes[1, 1]
    primary_shock = main.loc[main["event_time"] == 0, "coef"].iloc[0]
    labels = ["Primary (Kharif 2022)", "Placebo (Kharif 2021)"]
    coefs = [primary_shock, placebo["coef"].iloc[0]]
    errs = [0, placebo["se"].iloc[0] * 1.96]
    colors = ["#1f77b4", "gray"]
    ax.bar(labels, coefs, yerr=errs, color=colors, capsize=8)
    ax.set_title("14a Placebo vs primary shock coefficient")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_ylabel("DiD coef on NDVI anomaly")

    for row in axes:
        for a in row:
            a.set_xlabel("Event time (seasons from shock)")
            a.set_ylabel("NDVI anomaly (treated − control)")

    plt.suptitle(
        "Day 14 — Robustness checks on the Day 13 event-study DiD", y=1.00, fontsize=12
    )
    plt.tight_layout()
    out = FIG_DIR / "day14_robustness.png"
    plt.savefig(out, bbox_inches="tight", dpi=150)
    plt.close()
    return out


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    master = pd.read_parquet(MASTER)
    panel = pd.read_parquet(PANEL)
    panel["event_time"] = panel["event_time"].astype("Int64")

    # Panel used by caliper rematches needs the full matched trajectory data
    # at the cell × season level, not the Day 13-matched subset. Reload more broadly.
    # But in practice we can filter base_panel to any subset of cells, so we
    # load once and filter as needed.

    placebo = run_placebo(panel)
    placebo.to_csv(TABLE_DIR / "day14_placebo.csv", index=False)

    caliper_curves = run_caliper_sensitivity(master, panel)
    for label, df in caliper_curves.items():
        df.to_csv(TABLE_DIR / f"day14_caliper_{label}.csv", index=False)

    dose_curves = run_dose_response(master, panel)
    for label, df in dose_curves.items():
        df.to_csv(TABLE_DIR / f"day14_dose_{label}.csv", index=False)

    loo_curves = run_leave_one_out(panel)
    # Save a wide-format summary of LOO.
    loo_wide = pd.concat(
        [df.assign(dropped=d) for d, df in loo_curves.items()],
        ignore_index=True,
    )
    loo_wide.to_csv(TABLE_DIR / "day14_loo.csv", index=False)

    primary = pd.read_csv(TABLE_DIR / "day13_eventstudy_ndvi.csv")
    plot_path = plot_all(primary, placebo, caliper_curves, dose_curves, loo_curves)
    print(f"\nSaved composite plot: {plot_path}")

    print("\n=== Day 14 summary ===")
    print(
        f"  Placebo p-value: {placebo['pvalue'].iloc[0]:.3f} "
        f"({'passes' if placebo['pvalue'].iloc[0] > 0.05 else 'FAILS'})"
    )

    for label, df in caliper_curves.items():
        shock = df.loc[df["event_time"] == 0, "coef"].iloc[0]
        print(f"  Caliper {label}: shock k=0 coef = {shock:+.4f}")

    for label, df in dose_curves.items():
        shock = df.loc[df["event_time"] == 0, "coef"].iloc[0]
        print(f"  Dose {label}: shock k=0 coef = {shock:+.4f}")

    # LOO: find the min/max shock coefficient across all 22 LOO fits.
    loo_shocks = [
        df.loc[df["event_time"] == 0, "coef"].iloc[0] for df in loo_curves.values()
    ]
    print(
        f"  LOO shock range: {min(loo_shocks):+.4f} to {max(loo_shocks):+.4f} "
        f"(primary was {primary.loc[primary['event_time']==0, 'coef'].iloc[0]:+.4f})"
    )


if __name__ == "__main__":
    main()
