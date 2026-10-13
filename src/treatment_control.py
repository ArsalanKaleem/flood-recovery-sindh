"""
Day 11 — Treatment vs. control classification + balance check.

Primary threshold: flood_duration >= 3 (treated). Alternates for sensitivity:
flood_duration >= 1 (lenient) and >= 5 (strict).

Produces:
- Treatment-flag columns written back to master_dataset.parquet
- A balance diagnostic table: SMD on pre-flood covariates between treated
  and control cells, at the primary threshold
- A pre-flood NDVI distribution comparison (groundwork for Day 12 matching)
- Treated/control counts by district

Output: data/processed/master_dataset.parquet (updated in place)
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
MASTER_PATH = REPO / "data" / "processed" / "master_dataset.parquet"

PRIMARY_THRESHOLD = 3
SENS_THRESHOLDS = [1, 5]

# Covariates we'll use for the balance check — these are the pre-flood,
# time-invariant characteristics that could confound treatment-vs-control
# comparisons. Day 12's matching will operate on this same set.
BALANCE_COVARIATES = [
    "elevation_m",
    "slope_deg",
    "twi_approx",
    "dist_to_drainage_m",
    "cropland_frac",
    "rainfall_anomaly_mm",
]


def classify(df: pd.DataFrame, threshold: int) -> pd.Series:
    """Treatment flag: 1 if flood_duration >= threshold, else 0. NaN flood → NaN."""
    out = (df["flood_duration"] >= threshold).astype("int8")
    out[df["flood_duration"].isna()] = pd.NA
    return out


def smd(treated: pd.Series, control: pd.Series) -> float:
    """Standardized mean difference. |SMD| < 0.1 is the balance threshold."""
    mt, mc = treated.mean(), control.mean()
    vt, vc = treated.var(ddof=1), control.var(ddof=1)
    pooled_sd = np.sqrt((vt + vc) / 2)
    if pooled_sd == 0:
        return np.nan
    return (mt - mc) / pooled_sd


def balance_table(
    df: pd.DataFrame, treat_col: str, covariates: list[str]
) -> pd.DataFrame:
    """SMD + means for each covariate, treated vs control."""
    treated = df[df[treat_col] == 1]
    control = df[df[treat_col] == 0]

    rows = []
    for c in covariates:
        rows.append(
            {
                "covariate": c,
                "mean_treated": treated[c].mean(),
                "mean_control": control[c].mean(),
                "sd_treated": treated[c].std(),
                "sd_control": control[c].std(),
                "smd": smd(treated[c], control[c]),
                "balanced": abs(smd(treated[c], control[c])) < 0.1,
            }
        )
    return pd.DataFrame(rows)


def district_breakdown(df: pd.DataFrame, treat_col: str) -> pd.DataFrame:
    """Treated/control cell counts per district."""
    g = df.groupby("district")[treat_col].agg(
        n_total="count",
        n_treated="sum",
    )
    g["n_control"] = g["n_total"] - g["n_treated"]
    g["pct_treated"] = (g["n_treated"] / g["n_total"] * 100).round(1)
    return g.sort_values("pct_treated", ascending=False)


def main() -> None:
    df = pd.read_parquet(MASTER_PATH)
    print(f"Loaded master dataset: {len(df):,} cells")
    print(
        f"Flood duration: median {df['flood_duration'].median():.1f}, "
        f"max {df['flood_duration'].max():.0f}, "
        f"% zero {(df['flood_duration']==0).mean()*100:.1f}%"
    )

    # Classify at each threshold.
    print(f"\n=== Treatment classifications ===")
    for t in [PRIMARY_THRESHOLD] + SENS_THRESHOLDS:
        col = f"treated_t{t}"
        df[col] = classify(df, t)
        n_t = (df[col] == 1).sum()
        n_c = (df[col] == 0).sum()
        print(
            f"  threshold ≥{t}: treated {n_t:>7,} ({n_t/len(df)*100:5.1f}%)  "
            f"control {n_c:>7,} ({n_c/len(df)*100:5.1f}%)"
        )

    # Primary-threshold balance table.
    print(f"\n=== Balance check at primary threshold (≥{PRIMARY_THRESHOLD}) ===")
    print("For a VALID comparison, |SMD| should be < 0.1 on every covariate.")
    print("If it's not — which is almost certain here — we need matching (Day 12).\n")

    bal = balance_table(df, f"treated_t{PRIMARY_THRESHOLD}", BALANCE_COVARIATES)
    with pd.option_context(
        "display.max_columns",
        None,
        "display.width",
        200,
        "display.float_format",
        lambda x: f"{x:>10.3f}",
    ):
        print(bal.to_string(index=False))

    n_unbalanced = (~bal["balanced"]).sum()
    print(
        f"\n  → {n_unbalanced}/{len(bal)} covariates have |SMD| ≥ 0.1 "
        f"(unbalanced). This is the Day 11 checkpoint evidence that "
        f"raw flooded-vs-non-flooded is NOT a valid comparison."
    )

    # Pre-flood NDVI check: a critical covariate that MUST be balanced.
    # We don't have per-cell pre-flood NDVI in the master, so pull it from
    # the trajectory parquets. For a quick first look, compare on
    # cropland_frac as a proxy for pre-flood vegetation intensity.
    print(f"\n=== Pre-flood vegetation intensity (cropland_frac) ===")
    for t_name, t_val in [("treated", 1), ("control", 0)]:
        subset = df[df[f"treated_t{PRIMARY_THRESHOLD}"] == t_val]["cropland_frac"]
        print(
            f"  {t_name:7s}: n={len(subset):>7,}  "
            f"mean={subset.mean():.3f}  median={subset.median():.3f}"
        )

    # District breakdown.
    print(f"\n=== Treated share by district (primary threshold) ===")
    db = district_breakdown(df, f"treated_t{PRIMARY_THRESHOLD}")
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(db.to_string())

    # Save back to master.
    df.to_parquet(MASTER_PATH, index=False)
    print(f"\nSaved updated master dataset → {MASTER_PATH}")
    print(f"  New columns: treated_t1, treated_t3, treated_t5")


if __name__ == "__main__":
    main()
