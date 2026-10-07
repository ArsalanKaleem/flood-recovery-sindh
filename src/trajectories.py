"""
Trajectory loading and seasonal anomaly computation.

Called from notebooks/scripts in Phase 2. Produces the master
per-cell-per-month anomaly dataset that Day 8+ work on.
"""

from __future__ import annotations
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
RAW_DIR = REPO / "data" / "raw" / "trajectories"
OUT_PARQUET = REPO / "data" / "processed" / "recovery_trajectories.parquet"

# Pre-flood baseline window: Jan 2021 → Jul 2022 (flood struck in Aug 2022).
BASELINE_END = "2022-07"


def load_all_csvs(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Load every trajectory_*.csv, tag with district, concat into one frame."""
    files = sorted(raw_dir.glob("trajectory_*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSVs in {raw_dir}")
    print(f"Loading {len(files)} district CSVs...")

    frames = []
    for f in files:
        # District name: strip 'trajectory_' prefix and '.csv' suffix; underscores → spaces.
        district = f.stem.replace("trajectory_", "").replace("_", " ")
        df = pd.read_csv(f, usecols=["cell_id", "month", "ndvi", "evi"])
        df["district"] = district
        frames.append(df)
        print(f"  {district:25s} {len(df):>10,} rows")

    df = pd.concat(frames, ignore_index=True)
    print(f"\nTotal: {len(df):,} rows across {df['cell_id'].nunique():,} cells")
    return df


def compute_anomalies(
    df: pd.DataFrame, baseline_end: str = BASELINE_END
) -> pd.DataFrame:
    """
    For each cell, compute NDVI/EVI anomaly = observed − per-cell per-calendar-month
    mean over the pre-flood baseline window.

    Returns the same dataframe with 2 new columns: ndvi_anom, evi_anom.
    """
    df = df.copy()
    # Normalize month to a pandas period for cleaner handling.
    df["month"] = pd.PeriodIndex(df["month"], freq="M")
    df["cal_month"] = df["month"].dt.month  # 1..12

    print(f"\nComputing seasonal baselines (pre-flood window ≤ {baseline_end})...")
    baseline_cutoff = pd.Period(baseline_end, freq="M")
    pre = df[df["month"] <= baseline_cutoff]
    print(f"  Baseline rows: {len(pre):,} ({len(pre)/len(df)*100:.1f}% of total)")

    # Per-cell, per-calendar-month mean of NDVI/EVI during the baseline window.
    base = pre.groupby(["cell_id", "cal_month"], as_index=False).agg(
        ndvi_base=("ndvi", "mean"), evi_base=("evi", "mean")
    )
    print(f"  Per-cell-per-calendar-month baseline table: {len(base):,} rows")

    df = df.merge(base, on=["cell_id", "cal_month"], how="left")
    df["ndvi_anom"] = df["ndvi"] - df["ndvi_base"]
    df["evi_anom"] = df["evi"] - df["evi_base"]

    # Flag cells where we lack any baseline data (shouldn't happen with 19 months pre-flood,
    # but worth checking).
    missing = df["ndvi_base"].isna().sum()
    if missing:
        print(
            f"  WARNING: {missing:,} rows have no baseline (cell lacks pre-flood data)"
        )

    df = df.drop(columns=["cal_month"])
    return df


def save(df: pd.DataFrame, out_path: Path = OUT_PARQUET) -> None:
    """Save as parquet (much faster/smaller than CSV for this shape)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Convert Period back to string for parquet-friendly storage.
    df = df.copy()
    df["month"] = df["month"].astype(str)
    df.to_parquet(out_path, index=False)
    print(f"\nSaved → {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")


def load_processed(path: Path = OUT_PARQUET) -> pd.DataFrame:
    """Load the processed anomaly dataset. Called from Day 8+ scripts."""
    df = pd.read_parquet(path)
    df["month"] = pd.PeriodIndex(df["month"], freq="M")
    return df


def main() -> None:
    df = load_all_csvs()
    df = compute_anomalies(df)
    save(df)

    # Spot-check: pick a cell, print its trajectory.
    first_cell = df["cell_id"].iloc[0]
    sample = df[df["cell_id"] == first_cell].sort_values("month").head(10)
    print(f"\nSample trajectory for cell {first_cell}:")
    print(sample[["month", "ndvi", "ndvi_base", "ndvi_anom"]].to_string(index=False))


if __name__ == "__main__":
    main()
