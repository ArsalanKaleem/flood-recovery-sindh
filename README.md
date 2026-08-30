# From Flood Damage to Recovery

**Multi-Year Satellite Assessment of Agricultural Recovery Following the 2022 Sindh Floods**

Districts: Dadu, Jamshoro, Larkana, Qambar-Shahdadkot, Jacobabad, Shikarpur, Sanghar.

## Research question

How fast did cropland vegetation recover after the 2022 Sindh floods, and what environmental and
contextual factors explain the differences in recovery speed across space?

## Why this isn't a damage map

The single largest failure mode for this project is producing a NDVI before/after comparison and
calling it a recovery study. This repository is designed around a different question. The core
comparison is **flooded cropland vs. a matched non-flooded counterfactual**, tracked over
2021 → 2022 → 2023 → 2024 → 2025/26. The outputs of interest are:

- **Time-to-recovery** — growing seasons until NDVI/EVI returns to within a threshold of pre-flood baseline
- **Cumulative vegetation deficit** — area under the anomaly curve, from shock to recovery
- **Crop-cycle restoration** — count of distinct growing-season peaks pre- vs. post-flood

Explained by flood duration, waterlogging persistence, salinity proxy, rainfall anomaly, elevation,
soil texture proxy, and distance to drainage.

## Repository layout

```
flood-recovery-sindh/
├── config/           # AOI boundary (7 districts)
├── data/
│   ├── raw/          # Untouched exports from GEE and other sources
│   ├── interim/      # Derived rasters (flood extent, composites, masks)
│   └── processed/    # Analysis-ready tables (per-cell dataframe)
├── gee_scripts/      # Server-side GEE Python pipelines
├── notebooks/        # Analysis notebooks, numbered by phase
├── src/              # Reusable Python modules
├── results/          # Figures, tables, saved model objects
├── manuscript/       # Paper draft, references, submission package
└── docs/             # Data dictionary and internal notes
```

## Environment setup

```bash
# 1. Create the conda environment
conda env create -f environment.yml
conda activate flood-recovery

# 2. Authenticate Google Earth Engine (one-time)
earthengine authenticate

# 3. Set your GEE project ID (edit .env or export directly)
export GEE_PROJECT_ID=your-project-id-here

# 4. Sanity check
python -c "import ee; ee.Initialize(project='your-project-id'); print('GEE OK')"
```

## Working plan

25 focused days across five phases:

1. **Days 1–5** — Foundations & data acquisition
2. **Days 6–10** — Recovery metric construction
3. **Days 11–15** — Counterfactual & study design
4. **Days 16–20** — Statistical modeling & driver analysis
5. **Days 21–25** — Writing, validation, submission prep

Full plan: see `docs/working_plan.md` or the companion PDF.

## Data availability

All primary inputs are free public archives (Sentinel-1/2, Landsat 8/9, CHIRPS, ERA5-Land, SRTM,
ESA WorldCover, WorldPop, JRC Global Surface Water, OSM, PBS district statistics). No raw
satellite bytes are committed to this repo — see `.gitignore`. Derived analysis-ready tables under
`data/processed/` are versioned.

## License

TBD.

## Citation

TBD once the manuscript is drafted.
