# From Flood Damage to Recovery

**Multi-year satellite assessment of cropland recovery after the 2022 Sindh floods.**

> This repository contains the data pipeline, analysis code, and results for a
> peer-reviewed study of how fast flooded cropland in Sindh, Pakistan recovered
> from the 2022 monsoon floods — and what environmental factors explain the
> differences in recovery speed across space.

## What this study does

Most published remote-sensing papers on the 2022 Sindh floods map flood extent
or estimate crop losses in the shock season. This study asks a different
question: **how quickly did affected cropland bounce back, and did it bounce
back fully?**

The central methodological choice is a **matched counterfactual design** rather
than a before/after NDVI comparison. For each flooded cropland cell, we identify
a non-flooded cropland cell with statistically identical pre-flood
characteristics (elevation, slope, drainage proximity, soil wetness, pre-flood
vegetation level and trend). An event-study difference-in-differences model
then estimates the recovery trajectory of flooded cells relative to their
matched controls, season by season, from one year before the flood through
four years after.

## Headline findings

- Flooded cropland showed a **−0.20 NDVI anomaly** in the shock season (Kharif
  2022) relative to matched controls.
- Partial recovery was observable within one year, bottoming out at **−0.06**
  by Rabi 2023-24.
- Recovery did not continue monotonically. Cropland productivity **re-dampened**
  from Kharif 2024 onwards and has **not returned to pre-flood levels** through
  Kharif 2026 (persistent gap of −0.11 NDVI).
- Dose-response confirms the pattern: cells with longer flood exposure show
  deeper initial shock (−0.13 / −0.19 / −0.28 by flood-duration tercile) and
  slower recovery.
- Results are robust to placebo tests, caliper sensitivity, leave-one-district
  cross-validation, and alternative treatment thresholds.

## Study area

Agricultural Sindh — 22 districts covering the full Indus-fed cropland belt,
excluding Karachi metropolitan sub-districts and the Rann of Kutch salt flats.
~555,000 cropland cells at 300 m resolution.

Districts: Dadu, Jamshoro, Larkana, Jacobabad, Shikarpur, Sanghar, Badin,
Ghotki, Hyderabad, Kashmore, Khairpur, Malir, Matiari, Mirpurkhas, Mithi,
Naushahro Firoz, Nawabshah, Sukkur, Tando Allahyar, Tando M. Khan, Thatta,
Umerkot.

## Data sources

All inputs are free, public:

- **Sentinel-2 SR Harmonized** — monthly NDVI/EVI composites, 2021-01 to
  2026-09 (69 months). Cloud/shadow masking via s2cloudless probability.
- **Sentinel-1 GRD IW** — flood extent and duration, both orbits, Aug-Oct 2022.
  Refined Lee speckle filter + VV threshold.
- **CHIRPS** — rainfall anomaly for the 2022 monsoon vs 2015–2021 baseline.
- **ERA5-Land** — monthly soil moisture (layer 1) and skin temperature.
- **SRTM** — elevation, slope, and TWI approximation.
- **ESA WorldCover v200** — cropland mask (class 40).
- **WorldPop 2020** — population density.
- **JRC Global Surface Water** — permanent water masking.
- **Geofabrik Pakistan OSM extract + HydroRIVERS v1.0** — roads and drainage.
- **GADM 4.1** — administrative boundaries (level 3).

## Methodology

1. **Analysis grid** — 300 m stratified grid over cropland pixels across the 22
   districts; cells with ≥50% cropland fraction kept; 555,309 cells total.
2. **Vegetation trajectories** — per-cell monthly NDVI/EVI anomalies, computed
   relative to a per-cell harmonic baseline (annual sine + linear trend) fit
   on 19 months of pre-flood observations.
3. **Treatment classification** — cells with ≥3 Sentinel-1 water detections
   during Aug-Oct 2022 are "treated" (15.8% of grid); ≥1 and ≥5 used for
   sensitivity.
4. **Propensity matching** — 1:1 nearest-neighbor matching on logit propensity
   with caliper 0.2×SD. 87,447 of 87,948 treated cells matched (99.4%). All 7
   covariates balanced to |SMD| < 0.03 post-matching.
5. **Event-study DiD** — cell × season panel with cell + season-year fixed
   effects, SEs clustered by district. 11 seasons (Kharif 2021 reference
   through Kharif 2026).
6. **Robustness** — placebo (p=0.56), caliper sensitivity (0.5×/baseline/2×
   all coef ≈ −0.20), flood-duration dose-response (monotonic), leave-one-out
   across 22 districts (range −0.19 to −0.20).

## Repository layout

flood-recovery-sindh/
├── config/                  # 22-district AOI (GADM 4.1)
├── data/
│   ├── raw/                 # Downloaded CSVs (gitignored)
│   └── processed/           # Analysis-ready parquet files (gitignored)
├── gee\_scripts/             # Server-side GEE Python pipelines (Days 1-10)
├── src/
│   ├── gee\_utils.py         # GEE auth + export helpers
│   ├── indices.py           # NDVI, EVI, NDWI, SBI, S1 water
│   ├── trajectories.py      # Harmonic baselines + anomalies (Day 7)
│   ├── soil\_moisture.py     # ERA5 SM persistence (Day 9b)
│   ├── recovery\_metrics.py  # TTR, CVD, CCR per cell (Day 8)
│   ├── master\_dataset.py    # Merge everything (Day 10)
│   ├── treatment\_control.py # Treatment classification (Day 11)
│   ├── matching\_day12.py    # Propensity matching (Day 12)
│   ├── did\_day13.py         # Event-study DiD (Day 13)
│   ├── robustness\_day14.py  # Placebo/caliper/dose/LOO (Day 14)
│   └── viz.py               # Shared plotting style
├── results/
│   ├── figures/             # Plots for the manuscript
│   └── tables/              # Coefficient CSVs
├── manuscript/              # Draft, references, submission package
└── docs/
├── data\_dictionary.md
├── resume\_notes.md
└── day15\_gap\_review.md


## Reproducing the pipeline

Full environment and data acquisition takes ~40 compute hours on Google Earth
Engine (noncommercial tier) and ~5 hours on a laptop.

```bash
# 1. Environment
conda env create -f environment.yml
conda activate flood-recovery
earthengine authenticate

# 2. GEE pipelines (Days 1-10). Each submits async tasks; monitor at
#    https://code.earthengine.google.com/tasks
python gee_scripts/01_aoi_setup.py
python gee_scripts/02_sentinel2_composites.py
python gee_scripts/02b_sentinel2_full_batch.py
python gee_scripts/03_sentinel1_flood_mapping.py
python gee_scripts/04_ancillary_layers.py
# ... through 10_extract_covariates.py

# 3. Local analysis (Days 7-14)
python -m src.trajectories
python -m src.recovery_metrics
python -m src.soil_moisture
python -m src.master_dataset
python -m src.treatment_control
python -m src.matching_day12
python -m src.did_day13
python -m src.robustness_day14
```

See `docs/data_dictionary.md` for every variable's source, resolution, and
transformation.

## Status

Phase 3 (counterfactual design) complete. Entering Phase 4 (statistical
modeling of recovery drivers).

## License

TBD — will be set to CC-BY-4.0 at manuscript submission.

## Citation

TBD — this entry will be updated with the DOI once the manuscript is accepted.

## Contact

Repository: https://github.com/ArsalanKaleem/flood-recovery-sindh
