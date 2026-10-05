# Flood Recovery in Sindh

Multi-year satellite assessment of agricultural recovery after the 2022 floods in Sindh, Pakistan.

## Project overview

This project examines how cropland in Sindh recovered after the catastrophic 2022 flooding and which environmental conditions shaped the speed and completeness of that recovery. The study focuses on seven districts in the flood-affected region: Dadu, Jamshoro, Larkana, Qambar-Shahdadkot, Jacobabad, Shikarpur, and Sanghar.

The core question is not simply whether vegetation declined after the flood. Instead, the project asks:

- How quickly did flooded cropland return to a near-pre-flood vegetation state?
- How much cumulative productivity was lost relative to a non-flooded counterfactual?
- Which areas recovered fully, and which remained impaired by prolonged waterlogging or salinity?
- Which physical and climatic conditions most strongly explain differences in recovery across space?

## Why this is a recovery study, not a damage map

A simple before/after NDVI comparison would capture shock but would not isolate recovery. This repository is designed around a counterfactual framework:

- Flooded cropland is compared against matched non-flooded cropland with similar pre-flood conditions.
- Recovery is tracked over the period 2021 → 2022 → 2023 → 2024 → 2025/26.
- The outcomes of interest are recovery metrics, not just flood extent or initial damage.

Primary outcomes include:

- Time-to-recovery: number of growing seasons required for NDVI/EVI to return within a threshold of the pre-flood baseline
- Cumulative vegetation deficit: area under the anomaly curve from the flood shock until recovery
- Crop-cycle restoration: whether distinct seasonal growth peaks return to pre-flood patterns

These outcomes are then related to flood duration, waterlogging persistence, salinity proxies, rainfall anomalies, elevation, soil texture proxies, and distance to drainage.

## Scientific design

The project combines remote sensing, geospatial analytics, and statistical modeling:

- Sentinel-2 and Landsat time series for vegetation and water signals
- Sentinel-1 SAR for flood and water extent signals
- CHIRPS and ERA5-Land for rainfall and climate context
- SRTM and terrain derivatives for elevation and slope
- Soil and drainage proxies for local geomorphic controls
- Matching methods and difference-in-differences frameworks to estimate causal recovery contrasts

A key methodological step is to create a matched control group so that observed post-flood differences are not simply due to baseline differences in land productivity or geography.

## Repository structure

```text
flood-recovery-sindh/
├── config/                 # AOI files and district boundaries
├── data/
│   ├── raw/                # Unmodified source data and exports
│   ├── interim/            # Derived rasters, composites, and masks
│   └── processed/          # Analysis-ready tables and feature matrices
├── gee_scripts/            # Google Earth Engine Python pipelines
│   ├── 06_salinity_waterlog_proxy.py
│   └── ...
├── notebooks/              # Analysis notebooks by phase
├── src/                    # Reusable Python utilities and model code
│   ├── gee_utils.py
│   ├── indices.py
│   ├── matching.py
│   ├── models.py
│   └── viz.py
├── results/                # Figures, tables, and saved outputs
├── manuscript/             # Drafts, references, and paper materials
├── docs/                   # Internal documentation and data dictionaries
├── environment.yml         # Conda environment specification
├── .env.example            # Example environment file
├── README.md               # Project overview and setup guide
├── .gitignore
└── .env                    # Local environment settings (not committed)
```

## Data sources

The project uses public, open-access data sources, including:

- Sentinel-1/2 and Landsat 8/9 surface reflectance products
- CHIRPS rainfall data
- ERA5-Land climate variables
- SRTM elevation and derived terrain metrics
- ESA WorldCover and related land-cover products
- WorldPop and ancillary population or settlement context
- JRC Global Surface Water and drainage-linked reference layers
- OpenStreetMap and PBS district statistics where relevant

No raw satellite archives are committed to this repository; derived products and analysis-ready tables are stored under `data/` where appropriate.

## Environment setup

```bash
# 1. Create the conda environment
conda env create -f environment.yml
conda activate flood-recovery

# 2. Authenticate Google Earth Engine (one-time setup)
earthengine authenticate

# 3. Configure the GEE project ID
# either via a .env file or by exporting it in the terminal
export GEE_PROJECT_ID=your-project-id-here

# 4. Sanity check
python -c "import ee; ee.Initialize(project='your-project-id'); print('GEE OK')"
```

## Typical workflow

1. Build regional flood and vegetation masks in Google Earth Engine.
2. Create monthly or seasonal vegetation composites and anomaly time series.
3. Derive flood-duration, waterlogging, and salinity persistence proxies.
4. Assemble a cell-level analysis dataset for cropland parcels.
5. Match flooded cells with comparable non-flooded cells.
6. Estimate recovery metrics and model drivers of recovery using DiD and mixed-effects approaches.
7. Validate results through robustness checks and diagnostics.

## Key analysis concepts

### Waterlogging persistence

A proxy for how long anomalously wet or waterlogged conditions persisted after the flood. In the GEE pipeline, this is approximated by the number of post-flood months in which NDWI remained elevated above a baseline threshold.

### Salinity persistence

A proxy for lingering salt stress or bare-soil brightening after inundation. In the project workflow, this is approximated by counting the number of post-flood months in which SBI stayed elevated relative to the same pre-flood baseline window.

### Recovery metrics

The analysis emphasizes metrics that reflect functional agricultural recovery rather than simple vegetation loss, including recovery timing and cumulative deficit.

## Current status

This repository is organized as an active research workflow: geospatial processing scripts, analysis notebooks, and modeling utilities are being developed and refined around the counterfactual recovery study design.

## License

License information is still pending.

## Citation

Citation details will be added once the manuscript is finalized.
