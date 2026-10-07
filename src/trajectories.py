"""
Trajectory loading and seasonal anomaly computation (harmonic baseline).

For each cell, fits a seasonal baseline model to the 19-month pre-flood
window (Jan 2021 – Jul 2022):

    NDVI(t) = a + b*t + c*sin(2π·t/12) + d*cos(2π·t/12) + ε

Four parameters per cell: level (a), linear trend (b), annual seasonality
amplitude via (c, d). Anomaly = observed − fitted baseline, computed
across the full 69-month window.

Processed district-by-district to stay within laptop RAM. Writes one
parquet shard per district under data/processed/trajectories/.
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
RAW_DIR = REPO / "data" / "raw" / "trajectories"
PROC_DIR = REPO / "data" / "processed" / "trajectories"

# Pre-flood baseline window: Jan 2021 → Jul 2022 (flood struck Aug 2022).
BASELINE_END = "2022-07"

# Reference month for the linear trend's t=0 origin.
REF_PERIOD = pd.Period("2021-01", freq="M")


def _month_to_t(month_period) -> np.ndarray:
    """Convert month periods to integer t (months since Jan 2021)."""
    pi = pd.PeriodIndex(month_period)
    years = pi.year.to_numpy()
    months = pi.month.to_numpy()
    ref_total = REF_PERIOD.year * 12 + (REF_PERIOD.month - 1)
    row_total = years * 12 + (months - 1)
    return (row_total - ref_total).astype(np.float64)


def _design_matrix(t: np.ndarray) -> np.ndarray:
    """Harmonic design matrix: [1, t, sin(2πt/12), cos(2πt/12)]."""
    omega = 2 * np.pi / 12
    return np.column_stack(
        [
            np.ones_like(t),
            t,
            np.sin(omega * t),
            np.cos(omega * t),
        ]
    )


def _fit_cell_baselines(df: pd.DataFrame, baseline_end: str, band: str) -> pd.Series:
    """
    Fit per-cell harmonic baseline on pre-flood data, return predicted
    baseline for every row in df (indexed to match df).

    Vectorized by solving one normal equation per cell group — fast.
    """
    baseline_cutoff = pd.Period(baseline_end, freq="M")

    df = df.copy()
    df["t"] = _month_to_t(df["month"])

    pre = df[df["month"] <= baseline_cutoff]

    # Fit per cell: group-apply with least squares.
    # For speed, do it as a loop over groups but keep each group tiny.
    coeffs = {}
    for cell_id, g in pre.groupby("cell_id", sort=False):
        y = g[band].to_numpy(dtype=np.float64)
        valid = ~np.isnan(y)
        if valid.sum() < 4:  # need at least 4 obs for 4-param fit
            continue
        X = _design_matrix(g["t"].to_numpy(dtype=np.float64)[valid])
        y = y[valid]
        try:
            # Normal equation: beta = (X'X)^-1 X'y
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
            coeffs[cell_id] = beta
        except np.linalg.LinAlgError:
            continue

    # Predict baseline for every row.
    coeffs_df = pd.DataFrame.from_dict(
        coeffs, orient="index", columns=["b0", "b1", "b2", "b3"]
    )
    merged = df[["cell_id", "t"]].merge(
        coeffs_df, left_on="cell_id", right_index=True, how="left"
    )
    omega = 2 * np.pi / 12
    pred = (
        merged["b0"]
        + merged["b1"] * merged["t"]
        + merged["b2"] * np.sin(omega * merged["t"])
        + merged["b3"] * np.cos(omega * merged["t"])
    )
    return pred.values


def _compute_district_anomalies(df: pd.DataFrame, baseline_end: str) -> pd.DataFrame:
    """Harmonic baseline + anomaly computation for one district."""
    df = df.copy()
    df["month"] = pd.PeriodIndex(df["month"], freq="M")

    print(f"    fitting NDVI baselines...", end="", flush=True)
    df["ndvi_base"] = _fit_cell_baselines(df, baseline_end, "ndvi")
    print(" done")

    print(f"    fitting EVI baselines...", end="", flush=True)
    df["evi_base"] = _fit_cell_baselines(df, baseline_end, "evi")
    print(" done")

    df["ndvi_anom"] = df["ndvi"] - df["ndvi_base"]
    df["evi_anom"] = df["evi"] - df["evi_base"]

    df["month"] = df["month"].astype(str)  # parquet-friendly
    return df


def process_all_districts(
    raw_dir: Path = RAW_DIR,
    out_dir: Path = PROC_DIR,
    baseline_end: str = BASELINE_END,
    resume: bool = True,
) -> None:
    """Loop districts, compute harmonic anomalies per district, save parquet shards."""
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(raw_dir.glob("trajectory_*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSVs in {raw_dir}")

    print(f"Processing {len(files)} districts with harmonic baseline...")
    total_rows = 0
    for i, f in enumerate(files, 1):
        district = f.stem.replace("trajectory_", "").replace("_", " ")
        out_path = out_dir / f"{f.stem}.parquet"

        if resume and out_path.exists():
            n = len(pd.read_parquet(out_path, columns=["cell_id"]))
            print(f"  [{i:2d}/{len(files)}] {district:25s} skipped ({n:,} rows)")
            total_rows += n
            continue

        print(f"  [{i:2d}/{len(files)}] {district:25s} loading...")
        df = pd.read_csv(f, usecols=["cell_id", "month", "ndvi", "evi"])
        df["district"] = district
        n_in = len(df)

        df = _compute_district_anomalies(df, baseline_end)

        n_missing = df["ndvi_base"].isna().sum()
        df.to_parquet(out_path, index=False)
        warn = f"  ({n_missing:,} no-baseline)" if n_missing else ""
        print(f"            {n_in:>10,} rows → {out_path.name}{warn}")
        total_rows += n_in
        del df

    print(f"\nAll districts processed. Total rows: {total_rows:,}")


def load_processed(
    dir_path: Path = PROC_DIR, districts: list[str] | None = None
) -> pd.DataFrame:
    """Load processed anomaly data. Optionally filter to specific districts."""
    files = sorted(dir_path.glob("trajectory_*.parquet"))
    if districts is not None:
        wanted = {d.replace(" ", "_") for d in districts}
        files = [f for f in files if f.stem.replace("trajectory_", "") in wanted]
    if not files:
        raise FileNotFoundError(f"No parquet files in {dir_path}")
    frames = [pd.read_parquet(f) for f in files]
    df = pd.concat(frames, ignore_index=True)
    df["month"] = pd.PeriodIndex(df["month"], freq="M")
    return df


def main() -> None:
    process_all_districts()

    # Spot-check: show a sample trajectory.
    sample = load_processed(districts=["Dadu"])
    first_cell = sample["cell_id"].iloc[0]
    s = sample[sample["cell_id"] == first_cell].sort_values("month").head(12)
    print(f"\nSample trajectory for cell {first_cell} (Dadu):")
    print(s[["month", "ndvi", "ndvi_base", "ndvi_anom"]].to_string(index=False))


if __name__ == "__main__":
    main()
