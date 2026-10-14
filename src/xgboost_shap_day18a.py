"""
Day 18a — XGBoost + SHAP cross-check on TTR.


Trains XGBoost to predict time-to-recovery from the Model A covariates.
Compares SHAP feature importance to Model A beta ranking.
Produces SHAP summary and dependence plots as additional paper figures.

Three things we want to see:
  1. SHAP top-5 features ≈ Model A top-5 → cross-method robustness
  2. SHAP dependence plots reveal any nonlinear thresholds Model A missed
  3. Interaction effects visible in SHAP interactions
"""

from __future__ import annotations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from src.viz import set_paper_style

REPO = Path(__file__).resolve().parents[1]
MASTER = REPO / "data" / "processed" / "master_dataset.parquet"
TAB = REPO / "results" / "tables"
FIG_DIR = REPO / "results" / "figures"

FEATURES = [
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
TARGET = "ttr_months"

SHAP_SAMPLE_SIZE = 5000
RNG_SEED = 42


def prepare_data() -> tuple[pd.DataFrame, pd.Series]:
    """Load master, restrict to matched sample, build feature matrix."""
    df = pd.read_parquet(MASTER)
    df = df[df["matched_pair_id"].notna()].copy()
    df["log_dist_drainage"] = np.log(df["dist_to_struct_drainage_m"].clip(lower=1))
    df["log_dist_road"] = np.log(df["dist_to_road_m"].clip(lower=1))
    df["log_pop"] = np.log(df["population"].clip(lower=0) + 1)
    df = df.dropna(subset=FEATURES + [TARGET])
    X = df[FEATURES]
    y = df[TARGET]
    print(f"Matched sample: {len(df):,} cells (features complete)")
    return X, y


def fit_xgboost(X: pd.DataFrame, y: pd.Series) -> xgb.XGBRegressor:
    """Train XGBoost with early stopping."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RNG_SEED
    )
    print(f"Train: {len(X_train):,}  Test: {len(X_test):,}")

    model = xgb.XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RNG_SEED,
        early_stopping_rounds=20,
        eval_metric="mae",
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    pred = model.predict(X_test)
    r2 = r2_score(y_test, pred)
    mae = mean_absolute_error(y_test, pred)
    print(f"\n  Test R² = {r2:.3f}")
    print(f"  Test MAE = {mae:.2f} months")
    return model, X_test, y_test


def compute_shap(model: xgb.XGBRegressor, X: pd.DataFrame) -> np.ndarray:
    """Compute SHAP values on a random sample."""
    print(f"\nComputing SHAP values on {SHAP_SAMPLE_SIZE:,} cells...")
    sample = X.sample(SHAP_SAMPLE_SIZE, random_state=RNG_SEED)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample)
    return shap_values, sample


def feature_importance_ranking(
    shap_values: np.ndarray, feature_names: list[str]
) -> pd.DataFrame:
    """Mean absolute SHAP per feature → importance ranking."""
    mean_abs = np.abs(shap_values).mean(axis=0)
    df = (
        pd.DataFrame(
            {
                "feature": feature_names,
                "mean_abs_shap": mean_abs,
            }
        )
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    df["rank"] = range(1, len(df) + 1)
    return df


def compare_to_model_a(shap_ranking: pd.DataFrame) -> pd.DataFrame:
    """Side-by-side comparison of SHAP importance vs Model A |β|."""
    model_a = pd.read_csv(TAB / "day16c_model_a_ttr.csv", index_col=0)
    # Map SHAP feature names (unprefixed) to Model A names (z_prefix).
    rows = []
    for _, r in shap_ranking.iterrows():
        feat = r["feature"]
        model_a_name = f"z_{feat}"
        if model_a_name in model_a.index:
            beta = model_a.loc[model_a_name, "coefficient"]
            rows.append(
                {
                    "feature": feat,
                    "shap_rank": int(r["rank"]),
                    "shap_importance": r["mean_abs_shap"],
                    "model_a_beta": beta,
                    "model_a_abs_beta": abs(beta),
                }
            )
    comp = pd.DataFrame(rows)
    comp["model_a_rank"] = comp["model_a_abs_beta"].rank(ascending=False).astype(int)
    comp["rank_diff"] = comp["shap_rank"] - comp["model_a_rank"]
    return comp


def plot_shap_summary(shap_values: np.ndarray, X: pd.DataFrame, out_path: Path) -> None:
    set_paper_style()
    plt.figure(figsize=(8, 6))
    shap.summary_plot(
        shap_values, X, feature_names=X.columns.tolist(), show=False, max_display=11
    )
    plt.title("SHAP values on TTR (months)", fontsize=11, pad=10)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close()


def plot_shap_dependence(
    shap_values: np.ndarray, X: pd.DataFrame, features: list[str], out_path: Path
) -> None:
    """Dependence plots for top features."""
    set_paper_style()
    n = len(features)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]
    for ax, feat in zip(axes, features):
        idx = list(X.columns).index(feat)
        ax.scatter(X[feat].values, shap_values[:, idx], s=3, alpha=0.3, c="#1f77b4")
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xlabel(feat)
        ax.set_ylabel("SHAP value")
        ax.set_title(f"SHAP dependence: {feat}", fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close()


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    X, y = prepare_data()

    print("\nFitting XGBoost...")
    model, X_test, y_test = fit_xgboost(X, y)

    shap_values, X_shap = compute_shap(model, X_test)

    print("\n=== SHAP feature importance ranking ===")
    importance = feature_importance_ranking(shap_values, FEATURES)
    print(importance.to_string(index=False))

    print("\n=== SHAP vs Model A beta ranking ===")
    comp = compare_to_model_a(importance)
    with pd.option_context("display.float_format", lambda x: f"{x:>10.3f}"):
        print(
            comp[
                [
                    "feature",
                    "shap_rank",
                    "model_a_rank",
                    "rank_diff",
                    "shap_importance",
                    "model_a_beta",
                ]
            ].to_string(index=False)
        )

    comp.to_csv(TAB / "day18a_shap_vs_model_a.csv", index=False)

    # Agreement check: Kendall's tau on the two rankings.
    from scipy.stats import kendalltau

    tau, p = kendalltau(comp["shap_rank"], comp["model_a_rank"])
    print(f"\n  Rank concordance (Kendall's tau): {tau:+.3f} (p = {p:.3f})")
    print(
        f"  Interpretation: {'STRONG agreement' if tau > 0.6 else 'MODERATE agreement' if tau > 0.3 else 'WEAK agreement'}"
    )

    print("\nMaking SHAP figures...")
    summary_path = FIG_DIR / "day18a_shap_summary.png"
    plot_shap_summary(shap_values, X_shap, summary_path)
    print(f"  Saved → {summary_path.name}")

    top3 = importance.head(3)["feature"].tolist()
    dep_path = FIG_DIR / "day18a_shap_dependence_top3.png"
    plot_shap_dependence(shap_values, X_shap, top3, dep_path)
    print(f"  Saved → {dep_path.name}")

    print("\nDone.")


if __name__ == "__main__":
    main()
