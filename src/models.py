"""
Statistical models used in Phase 4.

- Difference-in-differences (Day 13): flooded vs. matched non-flooded, before/after 2022
- Mixed-effects regression (Day 16): recovery metric ~ flood + climate + soil, random intercept by district
- Spatial diagnostics + spatial lag/error (Day 17): correct for autocorrelation in residuals
"""

from __future__ import annotations

from typing import Sequence

import pandas as pd


DEFAULT_EXPLANATORY: list[str] = [
    "flood_duration_days",
    "waterlog_persistence",
    "salinity_proxy",
    "rainfall_anomaly_2022",
    "elevation_m",
    "slope_deg",
    "dist_to_drainage_m",
]


def fit_did(panel: pd.DataFrame, y: str, treat: str = "flooded",
            post: str = "post_2022", entity: str = "cell_id",
            time: str = "period") -> object:
    """
    Two-way fixed-effects DiD on a cell-period panel.

    panel must have columns: entity, time, y, treat, post, (plus any controls).
    """
    from linearmodels.panel import PanelOLS

    p = panel.set_index([entity, time])
    p["treat_post"] = p[treat].astype(int) * p[post].astype(int)
    model = PanelOLS.from_formula(
        f"{y} ~ 1 + treat_post + EntityEffects + TimeEffects",
        data=p,
    )
    return model.fit(cov_type="clustered", cluster_entity=True)


def fit_mixed_effects(df: pd.DataFrame, y: str,
                      fixed: Sequence[str] = DEFAULT_EXPLANATORY,
                      group: str = "district") -> object:
    """Random-intercept mixed-effects model by district."""
    import statsmodels.formula.api as smf

    formula = f"{y} ~ " + " + ".join(fixed)
    return smf.mixedlm(formula, df, groups=df[group]).fit(method="lbfgs")


def morans_i(residuals: pd.Series, weights) -> object:
    """Global Moran's I on model residuals; expects a libpysal weights object."""
    from esda.moran import Moran

    return Moran(residuals.values, weights)
