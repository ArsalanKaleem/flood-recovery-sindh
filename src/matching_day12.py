"""
Day 12 — Propensity-score matching for the flooded-vs-non-flooded comparison.

Primary treatment: treated_t3 (flood_duration >= 3 SAR detections).

Matching covariates:
  elevation_m, slope_deg, twi_approx, dist_to_drainage_m, cropland_frac,
  preflood_ndvi_level, preflood_ndvi_trend

Procedure:
  1. Pull per-cell pre-flood NDVI level (mean) and trend (OLS slope) from the
     trajectory parquets. Jan 2021 - Jul 2022 window.
  2. Fit a logistic propensity model on all 7 covariates.
  3. 1:1 nearest-neighbor matching on the logit of the propensity score,
     without replacement, with a caliper of 0.2 * SD(logit propensity).
  4. Honest reporting: how many treated matched, how many not, remaining
     imbalance on the matched sample.

Writes two new columns to master_dataset.parquet:
  propensity        - logistic model's predicted P(treated) for every cell
  matched_pair_id   - integer pair ID for matched cells (NaN for unmatched)
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

REPO = Path(__file__).resolve().parents[1]
MASTER_PATH = REPO / "data" / "processed" / "master_dataset.parquet"
TRAJ_DIR = REPO / "data" / "processed" / "trajectories"

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

BASELINE_END = pd.Period("2022-07", freq="M")
REF_PERIOD = pd.Period("2021-01", freq="M")


def compute_preflood_ndvi_stats() -> pd.DataFrame:
    """
    Per-cell mean NDVI and OLS trend over the Jan 2021 - Jul 2022 window.
    Returns one row per cell_id with columns: preflood_ndvi_level, preflood_ndvi_trend.
    """
    print("Computing pre-flood NDVI level + trend per cell...")
    files = sorted(TRAJ_DIR.glob("trajectory_*.parquet"))
    all_stats = []
    for f in tqdm(files, desc="  districts"):
        df = pd.read_parquet(f, columns=["cell_id", "month", "ndvi"])
        df["month"] = pd.PeriodIndex(df["month"], freq="M")
        pre = df[df["month"] <= BASELINE_END].copy()
        if pre.empty:
            continue

        # t = months since Jan 2021 for the trend regression.
        pre["t"] = (pre["month"].dt.year - REF_PERIOD.year) * 12 + (
            pre["month"].dt.month - REF_PERIOD.month
        )

        # Vectorized per-cell OLS slope: use groupby aggregation.
        g = pre.groupby("cell_id")
        stats = g["ndvi"].agg(preflood_ndvi_level="mean")

        # For slope, compute analytically: slope = cov(t, y) / var(t) per group.
        def slope(group):
            if len(group) < 2:
                return np.nan
            t = group["t"].to_numpy(dtype=np.float64)
            y = group["ndvi"].to_numpy(dtype=np.float64)
            m = ~np.isnan(y)
            if m.sum() < 2:
                return np.nan
            t, y = t[m], y[m]
            tc = t - t.mean()
            yc = y - y.mean()
            denom = (tc * tc).sum()
            return (tc * yc).sum() / denom if denom > 0 else np.nan

        slopes = g.apply(slope, include_groups=False).rename("preflood_ndvi_trend")
        stats = stats.join(slopes).reset_index()
        all_stats.append(stats)

    return pd.concat(all_stats, ignore_index=True)


def fit_propensity(
    df: pd.DataFrame, covariates: list[str], treat_col: str
) -> tuple[pd.Series, LogisticRegression]:
    """Fit logistic propensity and return scores + model."""
    print(f"\nFitting propensity model on {len(covariates)} covariates...")
    sub = df.dropna(subset=covariates + [treat_col]).copy()
    X = sub[covariates].to_numpy()
    y = sub[treat_col].astype(int).to_numpy()

    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    model = LogisticRegression(max_iter=2000, solver="lbfgs").fit(Xs, y)

    print(f"  Training rows: {len(sub):,}")
    print(f"  Treated prevalence: {y.mean():.3f}")
    coeffs = dict(zip(covariates, model.coef_[0]))
    print(f"  Coefficients (standardized):")
    for name, c in sorted(coeffs.items(), key=lambda kv: -abs(kv[1])):
        print(f"    {name:25s} {c:+.3f}")

    # Predict on all cells with full covariates.
    all_sub = df.dropna(subset=covariates).copy()
    Xa = scaler.transform(all_sub[covariates].to_numpy())
    propensity = pd.Series(
        model.predict_proba(Xa)[:, 1], index=all_sub.index, name="propensity"
    )
    return propensity, model


def one_to_one_match(
    df: pd.DataFrame, treat_col: str, prop_col: str, caliper: float
) -> pd.Series:
    """
    1:1 greedy nearest-neighbor matching on logit(propensity), no replacement.

    Greedy approach: sort treated by propensity, for each treated find
    nearest unused control within caliper. Uses a KDTree over control
    propensities for efficiency.

    Returns a Series indexed to df, with matched_pair_id (NaN if unmatched).
    """
    print(f"\nMatching with caliper = {caliper:.4f} (logit units)...")

    sub = df.dropna(subset=[treat_col, prop_col]).copy()
    # Logit transform; clip to avoid -inf / +inf on exact 0 or 1.
    p = np.clip(sub[prop_col].to_numpy(), 1e-6, 1 - 1e-6)
    sub["logit_p"] = np.log(p / (1 - p))

    treated = sub[sub[treat_col] == 1].copy()
    control = sub[sub[treat_col] == 0].copy()
    print(f"  Treated candidates: {len(treated):,}")
    print(f"  Control candidates: {len(control):,}")

    # Shuffle treated so no systematic bias in matching order.
    treated = treated.sample(frac=1, random_state=42)

    # KDTree over control logit-propensities. We'll mark used controls
    # in a boolean mask and query with higher k if the nearest is already used.
    ctrl_logit = control["logit_p"].to_numpy().reshape(-1, 1)
    tree = cKDTree(ctrl_logit)

    pair_id = np.full(len(df), np.nan)
    pair_counter = 0
    used_control = np.zeros(len(control), dtype=bool)
    control_pos = {idx: i for i, idx in enumerate(control.index)}

    k_initial = 10  # start by looking at 10 nearest controls per treated

    for t_idx, t_row in tqdm(treated.iterrows(), total=len(treated), desc="  matching"):
        tlog = t_row["logit_p"]
        # Query successive k's until we find an unused control or exhaust them.
        k = k_initial
        found = -1
        while k <= len(control):
            dists, idxs = tree.query([[tlog]], k=min(k, len(control)))
            for d, i in zip(dists[0], idxs[0]):
                if used_control[i]:
                    continue
                if d > caliper:
                    break  # distances sorted ascending; all further exceed caliper too
                found = i
                break
            if found >= 0:
                break
            if dists[0][-1] > caliper:
                break  # even the farthest among k is beyond caliper
            k *= 2

        if found >= 0:
            used_control[found] = True
            c_idx = control.index[found]
            pair_counter += 1
            pair_id[df.index.get_loc(t_idx)] = pair_counter
            pair_id[df.index.get_loc(c_idx)] = pair_counter

    print(f"  Matched pairs: {pair_counter:,} out of {len(treated):,} treated")
    print(
        f"  Unmatched treated: {len(treated) - pair_counter:,} "
        f"({(1 - pair_counter/len(treated))*100:.1f}%)"
    )
    return pd.Series(pair_id, index=df.index, name="matched_pair_id")


def balance_table(
    df: pd.DataFrame, treat_col: str, covariates: list[str]
) -> pd.DataFrame:
    treated = df[df[treat_col] == 1]
    control = df[df[treat_col] == 0]
    rows = []
    for c in covariates:
        mt, mc = treated[c].mean(), control[c].mean()
        vt, vc = treated[c].var(ddof=1), control[c].var(ddof=1)
        pooled_sd = np.sqrt((vt + vc) / 2) if (vt + vc) > 0 else np.nan
        smd = (mt - mc) / pooled_sd if pooled_sd and not np.isnan(pooled_sd) else np.nan
        rows.append(
            {
                "covariate": c,
                "mean_treated": mt,
                "mean_control": mc,
                "smd": smd,
                "balanced": abs(smd) < 0.1 if not np.isnan(smd) else False,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    df = pd.read_parquet(MASTER_PATH)
    print(f"Loaded master dataset: {len(df):,} cells")

    # Pre-flood NDVI stats.
    pre_ndvi = compute_preflood_ndvi_stats()
    print(f"  Computed stats for {len(pre_ndvi):,} cells")
    df = df.merge(pre_ndvi, on="cell_id", how="left")

    # Fit propensity on all cells with complete covariates.
    propensity, _ = fit_propensity(df, MATCH_COVARIATES, PRIMARY_TREAT)
    df["propensity"] = np.nan
    df.loc[propensity.index, "propensity"] = propensity.values

    # Caliper from SD of logit propensity among cells with a valid score.
    valid_p = df.loc[df["propensity"].notna(), "propensity"].clip(1e-6, 1 - 1e-6)
    logit_p = np.log(valid_p / (1 - valid_p))
    caliper = 0.2 * logit_p.std()

    # Match.
    df["matched_pair_id"] = one_to_one_match(df, PRIMARY_TREAT, "propensity", caliper)

    # Balance diagnostics.
    print(f"\n=== Balance BEFORE matching (full sample) ===")
    before = balance_table(df, PRIMARY_TREAT, MATCH_COVARIATES)
    with pd.option_context(
        "display.max_columns",
        None,
        "display.width",
        200,
        "display.float_format",
        lambda x: f"{x:>10.3f}",
    ):
        print(before.to_string(index=False))

    print(f"\n=== Balance AFTER matching (matched pairs only) ===")
    matched = df[df["matched_pair_id"].notna()]
    print(f"  Matched sample: {len(matched):,} cells ({len(matched)//2:,} pairs)")
    after = balance_table(matched, PRIMARY_TREAT, MATCH_COVARIATES)
    with pd.option_context(
        "display.max_columns",
        None,
        "display.width",
        200,
        "display.float_format",
        lambda x: f"{x:>10.3f}",
    ):
        print(after.to_string(index=False))

    n_balanced_before = before["balanced"].sum()
    n_balanced_after = after["balanced"].sum()
    print(
        f"\n  Covariates balanced (|SMD|<0.1): before {n_balanced_before}/{len(before)}, "
        f"after {n_balanced_after}/{len(after)}"
    )

    # Per-district matched pair counts.
    print(f"\n=== Matched pairs by district ===")
    matched_counts = (
        df[df["matched_pair_id"].notna()].groupby("district").size() // 2
    ).sort_values(ascending=False)
    print(matched_counts.to_string())

    # Save.
    df.to_parquet(MASTER_PATH, index=False)
    print(f"\nSaved updated master → {MASTER_PATH}")
    print(
        f"  New columns: preflood_ndvi_level, preflood_ndvi_trend, "
        f"propensity, matched_pair_id"
    )


if __name__ == "__main__":
    main()
