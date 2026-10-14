"""
Day 16c — Two mixed-effects regressions.

Model A: What explains time-to-recovery per cell?
  TTR ~ flood_duration + waterlog + salinity + heat + terrain +
        drainage + rainfall + population + (1 | district)

Model B: Does flood exposure interact with growing season?
  integrated_ndvi ~ flood_duration * is_kharif + season_fe + cell_fe

Model A uses TTR as the single-cell outcome (555k rows, matched sample).
Model B uses season-level panel data (phenology.parquet × matched sample).
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from linearmodels.panel import PanelOLS

REPO = Path(__file__).resolve().parents[1]
MASTER = REPO / "data" / "processed" / "master_dataset.parquet"
PHEN = REPO / "data" / "processed" / "phenology.parquet"
TAB = REPO / "results" / "tables"


def model_a_ttr() -> pd.DataFrame:
    """
    Mixed-effects: TTR as a function of flood & environmental covariates,
    random intercept by district.
    """
    print("=" * 60)
    print("MODEL A: TTR ~ flood + env covariates + (1|district)")
    print("=" * 60)

    df = pd.read_parquet(MASTER)
    print(f"Full master: {len(df):,}")

    # Restrict to matched sample for valid causal interpretation.
    df = df[df["matched_pair_id"].notna()].copy()
    print(f"Matched sample: {len(df):,}")

    # Transform. Log-transform skewed vars (distance, population).
    df["log_dist_drainage"] = np.log(df["dist_to_struct_drainage_m"].clip(lower=1))
    df["log_dist_road"] = np.log(df["dist_to_road_m"].clip(lower=1))
    df["log_pop"] = np.log(df["population"].clip(lower=0) + 1)

    # Standardize continuous predictors so coefficients are comparable.
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

    # Drop any rows with NaN in the regression columns.
    reg_cols = [f"z_{c}" for c in cols_to_std] + ["ttr_months", "district"]
    df_clean = df.dropna(subset=reg_cols).copy()
    print(f"After dropping NaN: {len(df_clean):,}")

    formula = (
        "ttr_months ~ z_flood_duration + z_sm_persist_months + z_sbi_persist_months "
        "+ z_ndwi_persist_months + z_lst_anom_K + z_elevation_m + z_slope_deg "
        "+ z_log_dist_drainage + z_log_dist_road "
        "+ z_rainfall_anomaly_mm + z_log_pop"
    )

    print(f"\nFitting mixed-effects model...")
    model = smf.mixedlm(formula, df_clean, groups=df_clean["district"])
    result = model.fit(method="lbfgs")

    print(f"\n{result.summary()}")

    # Extract coefficients to a clean table.
    coef_df = pd.DataFrame(
        {
            "coefficient": result.params,
            "std_error": result.bse,
            "pvalue": result.pvalues,
            "ci_low": result.conf_int()[0],
            "ci_high": result.conf_int()[1],
        }
    )
    coef_df["significant"] = coef_df["pvalue"] < 0.01
    coef_df.to_csv(TAB / "day16c_model_a_ttr.csv")
    print(f"\n  Saved → day16c_model_a_ttr.csv")

    print(f"\n  Standardized effect sizes (|coef|, significant only):")
    sig_coefs = (
        coef_df.loc[
            coef_df["significant"] & coef_df.index.str.startswith("z_"), "coefficient"
        ]
        .abs()
        .sort_values(ascending=False)
    )
    for name, mag in sig_coefs.items():
        var = name.replace("z_", "")
        direction = "+" if coef_df.loc[name, "coefficient"] > 0 else "-"
        print(f"    {direction} {var:25s} |beta| = {mag:.3f}")

    return coef_df


def model_b_asymmetry() -> pd.DataFrame:
    """
    Season-level panel: does flood_duration suppress integrated NDVI
    MORE in Kharif than Rabi? Interaction model with cell FE.
    """
    print("\n" + "=" * 60)
    print("MODEL B: integrated_ndvi ~ flood × Kharif + season_fe + cell_fe")
    print("=" * 60)

    master = pd.read_parquet(MASTER)
    phen = pd.read_parquet(PHEN)
    print(f"Phenology rows: {len(phen):,}")

    # Restrict to matched cells.
    matched = master[master["matched_pair_id"].notna()][
        ["cell_id", "flood_duration", "district"]
    ]
    panel = phen.merge(matched, on="cell_id", how="inner")
    print(f"Matched panel: {len(panel):,}")

    # Keep only post-flood seasons (shock + after).
    post_seasons = [
        "Kharif_2022",
        "Rabi_2022-23",
        "Kharif_2023",
        "Rabi_2023-24",
        "Kharif_2024",
        "Rabi_2024-25",
        "Kharif_2025",
        "Rabi_2025-26",
        "Kharif_2026",
    ]
    panel = panel[panel["season"].isin(post_seasons)].copy()
    panel["is_kharif"] = panel["season"].str.startswith("Kharif").astype(int)
    panel["z_flood"] = (
        panel["flood_duration"] - panel["flood_duration"].mean()
    ) / panel["flood_duration"].std()

    # Panel FE regression: cell + season fixed effects, interaction term.
    # Convert season to integer code for a numeric time index.
    season_order = [
        "Kharif_2022",
        "Rabi_2022-23",
        "Kharif_2023",
        "Rabi_2023-24",
        "Kharif_2024",
        "Rabi_2024-25",
        "Kharif_2025",
        "Rabi_2025-26",
        "Kharif_2026",
    ]
    season_to_int = {s: i for i, s in enumerate(season_order)}
    panel["season_idx"] = panel["season"].map(season_to_int).astype(int)

    panel["cell_id_idx"] = panel["cell_id"].astype(str)
    panel = panel.dropna(subset=["integrated_ndvi", "z_flood", "is_kharif"])
    df_i = panel.set_index(["cell_id_idx", "season_idx"])

    # Interaction. We drop main effects of z_flood (absorbed by cell FE)
    # and is_kharif (absorbed by season FE); only the interaction identifies.
    df_i["flood_x_kharif"] = df_i["z_flood"] * df_i["is_kharif"]

    print(f"Panel size: {len(df_i):,}")
    print(f"Fitting PanelOLS with entity + time effects, clustered SEs by district...")

    cluster = panel.set_index(["cell_id_idx", "season_idx"])["district"]
    model = PanelOLS.from_formula(
        "integrated_ndvi ~ 1 + flood_x_kharif + EntityEffects + TimeEffects",
        data=df_i,
    )
    result = model.fit(cov_type="clustered", clusters=cluster)
    print(f"\n{result.summary}")

    coef = result.params["flood_x_kharif"]
    se = result.std_errors["flood_x_kharif"]
    p = result.pvalues["flood_x_kharif"]

    out = pd.DataFrame(
        [
            {
                "coefficient": "flood_x_kharif",
                "estimate": coef,
                "std_error": se,
                "pvalue": p,
                "ci_low": result.conf_int().loc["flood_x_kharif", "lower"],
                "ci_high": result.conf_int().loc["flood_x_kharif", "upper"],
            }
        ]
    )
    out.to_csv(TAB / "day16c_model_b_asymmetry.csv", index=False)
    print(f"\n  Saved → day16c_model_b_asymmetry.csv")

    verdict = (
        "CONFIRMED"
        if (p < 0.01 and coef < 0)
        else ("NOT SIGNIFICANT" if p >= 0.01 else "WRONG SIGN")
    )
    print(
        f"\n  Interaction: flood_duration × Kharif = {coef:+.4f} (SE {se:.4f}), "
        f"p = {p:.2g} → {verdict}"
    )
    print(
        f"  Interpretation: each 1-SD increase in flood_duration causes an "
        f"ADDITIONAL {coef:+.3f} integrated NDVI drop in Kharif vs Rabi."
    )

    return out


def main() -> None:
    TAB.mkdir(parents=True, exist_ok=True)
    model_a_ttr()
    model_b_asymmetry()


if __name__ == "__main__":
    main()
