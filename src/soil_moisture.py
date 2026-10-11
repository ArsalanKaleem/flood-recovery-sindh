"""
Day 9b — Soil moisture persistence proxy.

Reads the per-cell per-month SM trajectories, builds a per-cell per-calendar-month
baseline from the pre-flood window (Jan 2021 – Jul 2022), and computes:

  sm_persist_months = # of months in Aug 2022 – Jul 2023 where SM anomaly
                      exceeded a threshold (positive = waterlogging)

Default threshold: 0.03 m³/m³ (volumetric soil water units). This is roughly
10% above typical dry Sindh baseline — generous enough to catch persistent
wetness but not so loose that normal variation triggers it.
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

REPO = Path(__file__).resolve().parents[1]
SM_DIR = REPO / "data" / "raw" / "sm_trajectories"
OUT_PATH = REPO / "data" / "processed" / "sm_persistence.parquet"

BASELINE_END = pd.Period("2022-07", freq="M")
POST_START = pd.Period("2022-08", freq="M")
POST_END = pd.Period("2023-07", freq="M")  # 12 months after flood
SM_THRESHOLD = 0.03  # m³/m³


def _compute_district_persistence(df: pd.DataFrame) -> pd.DataFrame:
    """Per-cell persistence count on one district's data."""
    df = df.copy()
    df["month"] = pd.PeriodIndex(df["month"], freq="M")
    df["cal_month"] = df["month"].dt.month.astype("int8")

    # Pre-flood per-cell per-calendar-month baseline.
    pre = df[df["month"] <= BASELINE_END]
    base = pre.groupby(["cell_id", "cal_month"], as_index=False).agg(
        sm_base=("soil_moisture", "mean")
    )

    df = df.merge(base, on=["cell_id", "cal_month"], how="left")
    df["sm_anom"] = df["soil_moisture"] - df["sm_base"]

    # Count post-flood months with positive anomaly > threshold.
    post = df[(df["month"] >= POST_START) & (df["month"] <= POST_END)].copy()
    post["waterlogged"] = (post["sm_anom"] > SM_THRESHOLD).astype(int)

    persist = post.groupby("cell_id", as_index=False).agg(
        sm_persist_months=("waterlogged", "sum"), sm_mean_anom_post=("sm_anom", "mean")
    )
    return persist


def main() -> None:
    files = sorted(SM_DIR.glob("sm_traj_*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSVs in {SM_DIR}")

    all_persist = []
    for f in tqdm(files, desc="districts"):
        df = pd.read_csv(f, usecols=["cell_id", "month", "soil_moisture"])
        persist = _compute_district_persistence(df)
        all_persist.append(persist)

    out = pd.concat(all_persist, ignore_index=True)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT_PATH, index=False)
    print(f"\nSaved {len(out):,} cells → {OUT_PATH}")

    print(f"\n--- SM persistence summary ---")
    print(
        f"Cells with ≥1 waterlogged month: {(out['sm_persist_months']>=1).mean()*100:.1f}%"
    )
    print(
        f"Cells with ≥3 waterlogged months: {(out['sm_persist_months']>=3).mean()*100:.1f}%"
    )
    print(f"Median persistence: {out['sm_persist_months'].median():.0f} months")
    print(f"Mean persistence:   {out['sm_persist_months'].mean():.2f} months")
    print(f"Max persistence:    {out['sm_persist_months'].max()} months")


if __name__ == "__main__":
    main()
