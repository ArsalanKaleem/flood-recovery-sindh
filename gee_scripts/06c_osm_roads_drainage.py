"""Day 5c — OSM roads + drainage lines over the Sindh AOI (local, osmnx)."""

from __future__ import annotations
from pathlib import Path

import geopandas as gpd
import osmnx as ox

REPO = Path(__file__).resolve().parents[1]
AOI_PATH = REPO / "config" / "aoi_districts.geojson"
OUT_DIR = REPO / "data" / "raw" / "osm"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    aoi = gpd.read_file(AOI_PATH).to_crs("EPSG:4326")
    polygon = aoi.unary_union  # single polygon covering the whole AOI

    print("Fetching OSM roads (motorway/trunk/primary/secondary/tertiary)...")
    roads = ox.features_from_polygon(
        polygon,
        tags={"highway": ["motorway", "trunk", "primary", "secondary", "tertiary"]},
    )
    roads = roads[roads.geometry.type.isin(["LineString", "MultiLineString"])]
    print(f"  roads: {len(roads)} features")
    roads.to_file(OUT_DIR / "osm_roads.gpkg", driver="GPKG")

    print("Fetching OSM waterways (river/stream/canal/drain)...")
    waterways = ox.features_from_polygon(
        polygon,
        tags={"waterway": ["river", "stream", "canal", "drain"]},
    )
    waterways = waterways[
        waterways.geometry.type.isin(["LineString", "MultiLineString"])
    ]
    print(f"  waterways: {len(waterways)} features")
    waterways.to_file(OUT_DIR / "osm_waterways.gpkg", driver="GPKG")

    print(f"\nSaved to {OUT_DIR}")


if __name__ == "__main__":
    main()
