"""Export the analysis grid as a zipped shapefile for GEE upload."""

from pathlib import Path
import shutil
import zipfile
import geopandas as gpd

REPO = Path(__file__).resolve().parents[1]
GRID_IN = REPO / "data" / "processed" / "analysis_grid.gpkg"
SHP_DIR = REPO / "data" / "processed" / "analysis_grid_shp"
ZIP_OUT = REPO / "data" / "processed" / "analysis_grid.zip"

SHP_DIR.mkdir(parents=True, exist_ok=True)

print(f"Loading {GRID_IN.name}...")
grid = gpd.read_file(GRID_IN).to_crs("EPSG:4326")
print(f"  {len(grid):,} cells")

# Shapefile field names max 10 chars. All our names already fit.
grid = grid[["cell_id", "district", "cropland_frac", "geometry"]].rename(
    columns={"cropland_frac": "crop_frac"}
)

shp_path = SHP_DIR / "analysis_grid.shp"
print(f"Writing shapefile to {SHP_DIR}...")
grid.to_file(shp_path, driver="ESRI Shapefile")

# Zip the four shapefile companion files together.
print(f"Zipping to {ZIP_OUT.name}...")
with zipfile.ZipFile(ZIP_OUT, "w", zipfile.ZIP_DEFLATED) as z:
    for ext in (".shp", ".shx", ".dbf", ".prj", ".cpg"):
        f = shp_path.with_suffix(ext)
        if f.exists():
            z.write(f, arcname=f.name)

# Clean up the loose files, keep only the zip.
shutil.rmtree(SHP_DIR)

size_mb = ZIP_OUT.stat().st_size / 1e6
print(f"\nDone. {ZIP_OUT.name}: {size_mb:.1f} MB")
if size_mb > 300:
    print("WARNING: >300 MB, may hit GEE upload limit. Simplification needed.")
