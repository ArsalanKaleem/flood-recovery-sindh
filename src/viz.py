"""
Shared plotting functions so every figure in the paper has consistent styling.
"""

from __future__ import annotations

from typing import Optional, Sequence

import matplotlib.pyplot as plt
import pandas as pd


# Project palette — colorblind-safe, distinguishable in grayscale.
DISTRICT_COLORS = {
    "Dadu":              "#1f77b4",
    "Jamshoro":          "#ff7f0e",
    "Larkana":           "#2ca02c",
    "Qambar-Shahdadkot": "#d62728",
    "Jacobabad":         "#9467bd",
    "Shikarpur":         "#8c564b",
    "Sanghar":           "#e377c2",
}


def set_paper_style() -> None:
    """Journal-friendly matplotlib defaults."""
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
    })


def plot_recovery_trajectory(df: pd.DataFrame, x: str = "date", y: str = "ndvi_anomaly",
                              group: str = "district", ax: Optional[plt.Axes] = None) -> plt.Axes:
    """
    Small-multiple-ready line plot of vegetation anomaly over time, grouped by district.
    Vertical guide marks the August 2022 flood onset.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 4))
    for name, sub in df.groupby(group):
        ax.plot(sub[x], sub[y], label=name,
                color=DISTRICT_COLORS.get(name), linewidth=1.5)
    ax.axvline(pd.Timestamp("2022-08-15"), color="k", linestyle="--", alpha=0.5,
               label="Flood onset")
    ax.axhline(0, color="k", linewidth=0.5)
    ax.set_xlabel("Date")
    ax.set_ylabel(y.replace("_", " ").title())
    ax.legend(loc="lower right", fontsize=8)
    return ax
