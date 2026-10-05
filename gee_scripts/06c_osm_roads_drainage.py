"""
Day 5c — Roads + drainage from local shapefiles, clipped to AOI.

Sources (local):
- Roads:      Geofabrik Pakistan OSM extract (gis_osm_roads_free_1.shp)
- OSM water:  Geofabrik Pakistan OSM extract (gis_osm_waterways_free_1.shp)
             — includes Sindh's irrigation canal network
- Rivers:    HydroRIVERS v1.0 global (HydroRIVERS_v10_as.shp)
             — natural rivers; filtered to Pakistan + Indus basin

Outputs saved to data/raw/osm/ as GeoPackage files.
"""

from __future__ import annotations
from pathlib import Path

import geopandas as gpd
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
AOI_PATH = REPO / "config" / "aoi_districts.geojson"
OUT_DIR = REPO / "data" / "raw" / "osm"
OUT_DIR.mkdir(parents=True, exist_ok=True)

GEOFABRIK_DIR = Path(r"C:\Users\Abbasi\Desktop\pakistan-260712-free.shp")
HYDRORIVERS_SHP = Path(
    r"C:\Users\Abbasi\Desktop\HydroRIVERS_v10_as_shp\HydroRIVERS_v10_as_shp\HydroRIVERS_v10_as.shp"
)

# OSM road classes we care about (per Geofabrik `fclass` field).
ROAD_CLASSES = {
    "motorway",
    "trunk",
    "primary",
    "secondary",
    "tertiary",
    "motorway_link",
    "trunk_link",
    "primary_link",
    "secondary_link",
    "tertiary_link",
}

# OSM waterway classes.
WATERWAY_CLASSES = {"river", "stream", "canal", "drain"}


def clip_and_save(
    gdf: gpd.GeoDataFrame, aoi: gpd.GeoDataFrame, out_path: Path, label: str
) -> None:
    """Reproject to AOI CRS, clip, save."""
    if gdf.crs != aoi.crs:
        gdf = gdf.to_crs(aoi.crs)
    print(f"  {label}: {len(gdf)} input features; clipping to AOI...")
    clipped = gpd.clip(gdf, aoi)
    # Keep only line geoms (clipping can create points at polygon boundaries).
    clipped = clipped[clipped.geometry.type.isin(["LineString", "MultiLineString"])]
    print(f"  {label}: {len(clipped)} features inside AOI")
    clipped.to_file(out_path, driver="GPKG")
    print(f"  saved → {out_path.name}")


def main() -> None:
    aoi = gpd.read_file(AOI_PATH).to_crs("EPSG:4326")
    aoi_union = aoi.dissolve()  # one polygon covering the full AOI
    print(f"AOI loaded: {len(aoi)} districts\n")

    # ----- Roads -----
    print("=== OSM Roads ===")
    roads_shp = GEOFABRIK_DIR / "gis_osm_roads_free_1.shp"
    roads = gpd.read_file(roads_shp)
    print(f"  Loaded {len(roads)} total Pakistan roads")
    roads = roads[roads["fclass"].isin(ROAD_CLASSES)]
    print(f"  Filtered to {len(roads)} major roads (motorway through tertiary)")
    clip_and_save(roads, aoi_union, OUT_DIR / "osm_roads.gpkg", "roads")

    # ----- OSM waterways (canals + natural) -----
    print("\n=== OSM Waterways ===")
    water_shp = GEOFABRIK_DIR / "gis_osm_waterways_free_1.shp"
    water = gpd.read_file(water_shp)
    print(f"  Loaded {len(water)} total Pakistan waterways")
    water = water[water["fclass"].isin(WATERWAY_CLASSES)]
    print(f"  Filtered to {len(water)} rivers/canals/streams/drains")
    clip_and_save(water, aoi_union, OUT_DIR / "osm_waterways.gpkg", "waterways")

    # ----- HydroRIVERS (natural rivers only) -----
    print("\n=== HydroRIVERS ===")
    print(f"  Loading {HYDRORIVERS_SHP.name} (large file, ~1.4M rows)...")
    # Read only what's inside the AOI bounding box to speed things up massively.
    bbox = tuple(aoi_union.total_bounds)
    rivers = gpd.read_file(HYDRORIVERS_SHP, bbox=bbox)
    print(f"  {len(rivers)} river reaches in AOI bbox")
    clip_and_save(rivers, aoi_union, OUT_DIR / "hydrorivers.gpkg", "rivers")

    print(f"\nAll outputs in {OUT_DIR}")


if __name__ == "__main__":
    main()
