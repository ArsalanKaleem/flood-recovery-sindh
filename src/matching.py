"""
Propensity-score and nearest-neighbor matching between flooded (treatment) and
non-flooded (candidate control) cropland cells.

Design notes (from Day 12 of the working plan):
- Match on pre-flood NDVI trend/level, elevation, slope, soil/texture proxy,
  and distance to river or drainage channel.
- After matching, check standardized mean differences (SMD) on every covariate.
  Target |SMD| < 0.1 before trusting the DiD in Day 13.
- Sensitivity to matching bandwidth is tested on Day 14 as a robustness check.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


MATCHING_COVARIATES: list[str] = [
    "pre_flood_ndvi_mean",
    "pre_flood_ndvi_trend",
    "elevation_m",
    "slope_deg",
    "soil_texture_proxy",
    "dist_to_drainage_m",
]


def fit_propensity(df: pd.DataFrame, treat_col: str = "flooded",
                   covariates: Sequence[str] = MATCHING_COVARIATES) -> pd.Series:
    """
    Fit a logistic propensity model for P(treat=1 | covariates).
    Returns propensity scores as a pd.Series aligned to df.index.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    X = StandardScaler().fit_transform(df[list(covariates)].values)
    y = df[treat_col].astype(int).values
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    return pd.Series(model.predict_proba(X)[:, 1], index=df.index, name="propensity")


def nearest_neighbor_match(df: pd.DataFrame, propensity: pd.Series,
                           treat_col: str = "flooded", caliper: float = 0.05) -> pd.DataFrame:
    """
    1:1 nearest-neighbor matching on propensity, without replacement,
    with a caliper (max absolute distance).

    Returns a dataframe with columns: [treatment_id, control_id, distance].
    """
    raise NotImplementedError("Implement on Day 12 — see notebooks/03_matching_counterfactual.ipynb")


def standardized_mean_difference(df: pd.DataFrame, treat_col: str,
                                 covariates: Sequence[str] = MATCHING_COVARIATES) -> pd.DataFrame:
    """
    SMD for each covariate between treated and control groups.
    |SMD| < 0.1 is the conventional balance threshold.
    """
    treated = df[df[treat_col] == 1]
    control = df[df[treat_col] == 0]
    records = []
    for c in covariates:
        mt, mc = treated[c].mean(), control[c].mean()
        vt, vc = treated[c].var(ddof=1), control[c].var(ddof=1)
        pooled_sd = np.sqrt((vt + vc) / 2)
        smd = (mt - mc) / pooled_sd if pooled_sd > 0 else np.nan
        records.append({"covariate": c, "mean_treated": mt, "mean_control": mc, "smd": smd})
    return pd.DataFrame(records)
