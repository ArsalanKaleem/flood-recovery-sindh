"""
Day 16b — Structural drainage distance proxy from OSM canals.

LBOD/RBOD-specific shapefiles are not publicly available. We approximate
major structural drainage by filtering the OSM waterways layer to only
canals of substantial length (>= 10 km after merging contiguous segments),
which effectively isolates the engineered drain network from short local
irrigation channels and natural streams.

Then compute distance from each analysis cell centroid to the nearest
major-structural-drainage line, add as a new column to master_dataset.

Methods caveat: this does not distinguish LBOD/RBOD from each other or
from functional canals, so we cannot test infrastructure-failure
hypotheses specifically. It does provide a cell-level measure of
engineered drainage proximity for regression use.
"""

from __future__ import annotations
from pathlib import Path

import geopandas as gpd
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
MASTER = REPO / "data" / "processed" / "master_dataset.parquet"
GRID = REPO / "data" / "processed" / "analysis_grid.gpkg"
OSM_WATER = REPO / "data" / "raw" / "osm" / "osm_waterways.gpkg"

LENGTH_THRESHOLD_M = 10_000  # retain only canals >= 10 km after dissolving


def main() -> None:
    print("Loading OSM waterways...")
    water = gpd.read_file(OSM_WATER).to_crs("EPSG:32642")
    print(f"  Total waterways: {len(water):,}")

    # Keep only canals (OSM fclass).
    canals = water[water["fclass"] == "canal"].copy()
    print(f"  Canals: {len(canals):,}")

    # Dissolve contiguous canal segments into continuous lines, then measure
    # length and keep only major (>= 10km) ones. OSM breaks long canals into
    # many short segments at bridges etc., so we union adjacent ones first.
    canals["length_m"] = canals.geometry.length
    print(f"  Canal length distribution:")
    print(f"    median: {canals['length_m'].median():.0f} m")
    print(f"    95th:   {canals['length_m'].quantile(0.95):.0f} m")
    print(f"    max:    {canals['length_m'].max():.0f} m")

    # Group by name (if available) and compute total length per named canal.
    # Unnamed canals get kept individually.
    if "name" in canals.columns:
        unnamed_fallback = pd.Series(
            ["UNNAMED_" + str(i) for i in canals.index],
            index=canals.index,
        )
        canals["name_norm"] = canals["name"].fillna(unnamed_fallback)
        name_lengths = canals.groupby("name_norm")["length_m"].sum()
        major_names = name_lengths[name_lengths >= LENGTH_THRESHOLD_M].index
        structural = canals[canals["name_norm"].isin(major_names)].copy()
    else:
        structural = canals[canals["length_m"] >= LENGTH_THRESHOLD_M].copy()

    print(
        f"\n  Structural drainage (named canals ≥ {LENGTH_THRESHOLD_M/1000:.0f} km total): {len(structural):,} segments"
    )

    # Also include any single-segment canal ≥10km (unnamed long ones).
    extra = canals[
        (canals["length_m"] >= LENGTH_THRESHOLD_M)
        & (~canals.index.isin(structural.index))
    ]
    if len(extra):
        structural = pd.concat([structural, extra], ignore_index=False)
        print(f"  + {len(extra):,} additional long unnamed segments")

    # Save the structural layer for QGIS inspection.
    out_struct = REPO / "data" / "processed" / "osm_structural_drainage.gpkg"
    structural[["geometry"]].to_file(out_struct, driver="GPKG")
    print(f"  Saved structural drainage layer → {out_struct.name}")

    # Compute distance from each cell centroid.
    print("\nComputing distance-to-structural-drainage for 555k cells...")
    grid = gpd.read_file(GRID).to_crs("EPSG:32642")
    grid["centroid"] = grid.geometry.centroid
    centroids = gpd.GeoDataFrame(
        grid[["cell_id"]], geometry=grid["centroid"], crs="EPSG:32642"
    )

    joined = gpd.sjoin_nearest(
        centroids,
        structural[["geometry"]].reset_index(drop=True),
        how="left",
        distance_col="dist_to_struct_drainage_m",
    )
    joined = joined.drop_duplicates(subset="cell_id", keep="first")

    dist = joined.set_index("cell_id")["dist_to_struct_drainage_m"]

    # Merge into master.
    print("\nMerging into master dataset...")
    master = pd.read_parquet(MASTER)
    master = master.merge(
        dist.rename("dist_to_struct_drainage_m"),
        left_on="cell_id",
        right_index=True,
        how="left",
    )
    print(f"  Rows: {len(master):,}")
    print(
        f"  dist_to_struct_drainage_m: median {master['dist_to_struct_drainage_m'].median():.0f} m, "
        f"95th {master['dist_to_struct_drainage_m'].quantile(0.95):.0f} m"
    )

    master.to_parquet(MASTER, index=False)
    print(f"\n  Saved updated master → {MASTER}")

    # Quick sanity check: treated cells should be closer to drainage than
    # controls (because flooding occurs preferentially near drainage).
    treated_dist = master[master["treated_t3"] == 1][
        "dist_to_struct_drainage_m"
    ].median()
    control_dist = master[master["treated_t3"] == 0][
        "dist_to_struct_drainage_m"
    ].median()
    print(f"\n  Sanity check:")
    print(f"    Treated median distance: {treated_dist:.0f} m")
    print(f"    Control median distance: {control_dist:.0f} m")


if __name__ == "__main__":
    main()
