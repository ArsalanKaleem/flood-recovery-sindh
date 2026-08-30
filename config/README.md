# Configuration

## `aoi_districts.geojson` (create on Day 1)

The 7-district AOI. Not yet included — you generate it on Day 1 from one of:

1. **GAUL admin-2** via GEE: filter `FAO/GAUL/2015/level2` for the seven district
   names and export to GeoJSON.
2. **OSM** via `osmnx`:
   ```python
   import osmnx as ox, geopandas as gpd
   names = ["Dadu, Sindh, Pakistan", "Jamshoro, Sindh, Pakistan",
            "Larkana, Sindh, Pakistan", "Qambar Shahdadkot, Sindh, Pakistan",
            "Jacobabad, Sindh, Pakistan", "Shikarpur, Sindh, Pakistan",
            "Sanghar, Sindh, Pakistan"]
   gdf = gpd.GeoDataFrame(
       {"district": [n.split(",")[0] for n in names],
        "geometry": [ox.geocode_to_gdf(n).geometry.iloc[0] for n in names]},
       crs="EPSG:4326")
   gdf.to_file("config/aoi_districts.geojson", driver="GeoJSON")
   ```
3. **Pakistan Bureau of Statistics** district shapefile (if you have one from prior work).

Whichever source, save the file as `config/aoi_districts.geojson` with at
minimum a `district` string property per feature. `src/gee_utils.load_aoi()`
expects that path.
