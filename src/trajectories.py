"""
Trajectory loading and seasonal anomaly computation.

Processes district-by-district to stay within laptop RAM. Writes one
parquet shard per district under data/processed/trajectories/, then
load_processed() stitches them on read.
"""

from __future__ import annotations
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
RAW_DIR = REPO / "data" / "raw" / "trajectories"
PROC_DIR = REPO / "data" / "processed" / "trajectories"

# Pre-flood baseline window: Jan 2021 → Jul 2022 (flood struck in Aug 2022).
BASELINE_END = "2022-07"


def _compute_district_anomalies(df: pd.DataFrame, baseline_end: str) -> pd.DataFrame:
    """Anomaly computation on a single district's data (small enough to fit)."""
    df = df.copy()
    df["month"] = pd.PeriodIndex(df["month"], freq="M")
    df["cal_month"] = df["month"].dt.month.astype("int8")

    baseline_cutoff = pd.Period(baseline_end, freq="M")
    pre = df[df["month"] <= baseline_cutoff]
    base = pre.groupby(["cell_id", "cal_month"], as_index=False).agg(
        ndvi_base=("ndvi", "mean"), evi_base=("evi", "mean")
    )

    df = df.merge(base, on=["cell_id", "cal_month"], how="left")
    df["ndvi_anom"] = df["ndvi"] - df["ndvi_base"]
    df["evi_anom"] = df["evi"] - df["evi_base"]

    df = df.drop(columns=["cal_month"])
    df["month"] = df["month"].astype(str)  # parquet-friendly
    return df


def process_all_districts(
    raw_dir: Path = RAW_DIR,
    out_dir: Path = PROC_DIR,
    baseline_end: str = BASELINE_END,
    resume: bool = True,
) -> None:
    """Loop districts, compute anomalies per district, save one parquet each."""
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(raw_dir.glob("trajectory_*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSVs in {raw_dir}")

    print(f"Processing {len(files)} districts...")
    total_rows = 0
    for i, f in enumerate(files, 1):
        district = f.stem.replace("trajectory_", "").replace("_", " ")
        out_path = out_dir / f"{f.stem}.parquet"

        if resume and out_path.exists():
            n = len(pd.read_parquet(out_path, columns=["cell_id"]))
            print(
                f"  [{i:2d}/{len(files)}] {district:25s} skipped ({n:,} rows already saved)"
            )
            total_rows += n
            continue

        df = pd.read_csv(f, usecols=["cell_id", "month", "ndvi", "evi"])
        df["district"] = district
        n_in = len(df)

        df = _compute_district_anomalies(df, baseline_end)

        n_missing = df["ndvi_base"].isna().sum()
        df.to_parquet(out_path, index=False)
        warn = f"  ({n_missing:,} no-baseline)" if n_missing else ""
        print(
            f"  [{i:2d}/{len(files)}] {district:25s} {n_in:>10,} rows → {out_path.name}{warn}"
        )
        total_rows += n_in

        # Explicit cleanup to keep memory flat across iterations.
        del df

    print(f"\nAll districts processed. Total rows: {total_rows:,}")


def load_processed(
    dir_path: Path = PROC_DIR, districts: list[str] | None = None
) -> pd.DataFrame:
    """
    Load processed anomaly data. Pass districts=['Dadu', 'Larkana'] to filter.
    Reads all shards and concatenates. Returns one DataFrame.
    """
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

    # Spot-check: load one district, show a sample trajectory.
    sample = load_processed(districts=["Dadu"])
    first_cell = sample["cell_id"].iloc[0]
    s = sample[sample["cell_id"] == first_cell].sort_values("month").head(10)
    print(f"\nSample trajectory for cell {first_cell} (Dadu):")
    print(s[["month", "ndvi", "ndvi_base", "ndvi_anom"]].to_string(index=False))


if __name__ == "__main__":
    main()
