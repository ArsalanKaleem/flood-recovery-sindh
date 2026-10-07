"""
Day 8 — Primary recovery metrics per cell.

Reads processed per-cell-per-month trajectories (Day 7), computes three
per-cell outcomes, saves one row per cell for Phase 3+ analysis.

Metrics:
  ttr_months    — months until NDVI anomaly returns within 0.05 of baseline
                   and stays there ≥3 consecutive months (right-censored at 50)
  ttr_censored   — 1 if the cell never recovered in the observation window
  cvd            — cumulative vegetation deficit: sum of negative NDVI
                   anomalies from Aug 2022 onward (unweighted)
  cvd_months     — number of post-flood months with negative anomalies
                   (denominator context for CVD)
  ccr_pre        — # of growing-season peaks pre-flood (Jan 2021 – Jul 2022)
  ccr_post       — # of growing-season peaks post-flood (Nov 2022 – Oct 2024)
  ccr_delta      — ccr_post − ccr_pre (negative = lost cycles)

Parameters chosen per design discussion:
  threshold = 0.05 NDVI anomaly units
  sustained_months = 3 (one Kharif or Rabi growing season)
  censoring = right-censored at 50 months (full observation window)
  peak prominence = 0.1 NDVI units (noise floor)
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from tqdm import tqdm

from src.trajectories import PROC_DIR as TRAJ_DIR

REPO = Path(__file__).resolve().parents[1]
OUT_PATH = REPO / "data" / "processed" / "recovery_metrics.parquet"

# Analysis windows.
# Analysis windows.
FLOOD_PERIOD = pd.Period("2022-08", freq="M")
PRE_START = pd.Period("2021-10", freq="M")  # 10 months before flood
PRE_END = pd.Period("2022-07", freq="M")
POST_CCR_START = pd.Period("2023-08", freq="M")  # 12 months after flood (post-recovery)
POST_CCR_END = pd.Period(
    "2024-05", freq="M"
)  # 10 months, matches pre window# 19 months, matches pre window# inclusive

# Metric parameters.
THRESHOLD = 0.05  # NDVI anomaly recovery band
SUSTAINED = 3  # consecutive months above threshold
CENSOR_AT = 50  # months; = full observation window Aug 2022 → Sep 2026
PEAK_PROM = 0.10  # NDVI units for find_peaks prominence


def _compute_ttr(anom_post: np.ndarray) -> tuple[int, int]:
    """
    Months from flood (index 0 = Aug 2022) to first month where
    anomaly >= -THRESHOLD and stays there for SUSTAINED consecutive months.
    Returns (ttr_months, censored_flag).
    """
    if len(anom_post) == 0:
        return CENSOR_AT, 1

    # Boolean mask: True where anomaly is "recovered" (within threshold).
    recovered = anom_post >= -THRESHOLD

    # Look for the first index i where recovered[i:i+SUSTAINED] all True.
    for i in range(len(recovered) - SUSTAINED + 1):
        if recovered[i : i + SUSTAINED].all():
            return int(i), 0
    return CENSOR_AT, 1


def _compute_cvd(anom_post: np.ndarray) -> tuple[float, int]:
    """Cumulative deficit (sum of negative anomalies) and count of deficit months."""
    neg = anom_post[anom_post < 0]
    return float(-neg.sum()), int(len(neg))


def _count_peaks(ndvi_series: np.ndarray) -> int:
    """
    Number of growing-season peaks via scipy.signal.find_peaks.
    Returns 0 for all-NaN or too-short series.
    """
    y = pd.Series(ndvi_series).interpolate(limit_direction="both").to_numpy()
    if len(y) < 4 or np.all(np.isnan(y)):
        return 0
    peaks, _ = find_peaks(y, prominence=PEAK_PROM)
    return int(len(peaks))


def _cell_metrics(group: pd.DataFrame) -> pd.Series:
    """Compute all metrics for a single cell's full time series (sorted by month)."""
    group = group.sort_values("month")
    months = pd.PeriodIndex(group["month"])

    # Slice windows.
    post_mask = months >= FLOOD_PERIOD
    anom_post = group.loc[post_mask, "ndvi_anom"].to_numpy()

    pre_mask = (months >= PRE_START) & (months <= PRE_END)
    ndvi_pre = group.loc[pre_mask, "ndvi"].to_numpy()

    post_ccr_mask = (months >= POST_CCR_START) & (months <= POST_CCR_END)
    ndvi_post = group.loc[post_ccr_mask, "ndvi"].to_numpy()

    # Drop NaN anomalies (missing baselines) from TTR/CVD so they don't
    # confuse the "recovered" test.
    anom_post = anom_post[~np.isnan(anom_post)]

    ttr, cens = _compute_ttr(anom_post)
    cvd, cvd_m = _compute_cvd(anom_post)
    ccr_pre = _count_peaks(ndvi_pre)
    ccr_post = _count_peaks(ndvi_post)

    return pd.Series(
        {
            "ttr_months": ttr,
            "ttr_censored": cens,
            "cvd": cvd,
            "cvd_months": cvd_m,
            "ccr_pre": ccr_pre,
            "ccr_post": ccr_post,
            "ccr_delta": ccr_post - ccr_pre,
        }
    )


def process_district(parquet_path: Path) -> pd.DataFrame:
    """Compute recovery metrics for every cell in one district parquet shard."""
    df = pd.read_parquet(parquet_path)
    df["month"] = pd.PeriodIndex(df["month"], freq="M")
    district = df["district"].iloc[0]

    metrics = (
        df.groupby("cell_id", sort=False)
        .apply(_cell_metrics, include_groups=False)
        .reset_index()
    )
    metrics["district"] = district
    return metrics


def main() -> None:
    files = sorted(TRAJ_DIR.glob("trajectory_*.parquet"))
    if not files:
        raise FileNotFoundError(f"No trajectory shards in {TRAJ_DIR}")

    all_metrics = []
    for f in tqdm(files, desc="districts"):
        m = process_district(f)
        all_metrics.append(m)

    out = pd.concat(all_metrics, ignore_index=True)
    cols = [
        "cell_id",
        "district",
        "ttr_months",
        "ttr_censored",
        "cvd",
        "cvd_months",
        "ccr_pre",
        "ccr_post",
        "ccr_delta",
    ]
    out = out[cols]

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT_PATH, index=False)
    print(f"\nSaved {len(out):,} rows → {OUT_PATH}")

    # Summary stats.
    print(f"\n--- Summary across {len(out):,} cells ---")
    print(
        f"TTR (uncensored): median {out[out['ttr_censored']==0]['ttr_months'].median():.1f} months, "
        f"mean {out[out['ttr_censored']==0]['ttr_months'].mean():.1f}"
    )
    print(
        f"TTR censoring rate: {out['ttr_censored'].mean()*100:.1f}% of cells never recovered"
    )
    print(
        f"CVD: median {out['cvd'].median():.2f} NDVI-month, mean {out['cvd'].mean():.2f}"
    )
    print(
        f"CCR pre:  median {out['ccr_pre'].median():.0f}, mean {out['ccr_pre'].mean():.2f}"
    )
    print(
        f"CCR post: median {out['ccr_post'].median():.0f}, mean {out['ccr_post'].mean():.2f}"
    )
    print(
        f"CCR delta: median {out['ccr_delta'].median():.0f}, mean {out['ccr_delta'].mean():.2f}"
    )

    # Per-district TTR median for a first look at heterogeneity.
    print(f"\nMedian TTR by district (uncensored only):")
    district_ttr = (
        out[out["ttr_censored"] == 0]
        .groupby("district")["ttr_months"]
        .median()
        .sort_values()
    )
    for d, v in district_ttr.items():
        print(f"  {d:25s} {v:>5.1f} months")


if __name__ == "__main__":
    main()
