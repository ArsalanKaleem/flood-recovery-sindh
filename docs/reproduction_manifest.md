# Reproduction Manifest

Exact command sequence to recreate every result in `results/`.

## Prerequisites

- Python 3.11 (via conda env `flood-recovery`)
- GEE account with `flood-recovery-sindh` project set up
- ~50 GB free disk space for `data/processed/` outputs
- Approximately 40 compute-hours on GEE (noncommercial tier), plus
  ~5 hours of local Python work

Install:

```bash
conda env create -f environment.yml
conda activate flood-recovery
earthengine authenticate
```

## Phase 1: Foundation data (Days 1–5)

Each GEE pipeline submits async tasks. Monitor at
https://code.earthengine.google.com/tasks

```bash
python gee_scripts/01_aoi_setup.py
python gee_scripts/02_sentinel2_composites.py
python gee_scripts/02b_sentinel2_full_batch.py
python gee_scripts/03_sentinel1_flood_mapping.py
python gee_scripts/04_ancillary_layers.py
python gee_scripts/05_persistence_and_pop.py
```

Download CSVs from Google Drive → `flood-recovery-exports/` to local
`data/raw/trajectories/`, etc.

## Phase 2: Analysis grid + metrics (Days 6–10)

```bash
python gee_scripts/07b_build_analysis_grid.py        # 555k cell grid
python -m src.trajectories                           # harmonic baselines
python -m src.recovery_metrics                       # TTR, CVD, CCR
python gee_scripts/09_extract_secondary.py           # SM, LST, flood_dur
python -m src.soil_moisture                          # SM persistence
python gee_scripts/10_extract_covariates.py          # terrain, rainfall, etc.
python -m src.master_dataset                         # merge → master_dataset.parquet
```

**Output**: `data/processed/master_dataset.parquet` (555,309 × 23 columns)

## Phase 3: Counterfactual design (Days 11–14)

```bash
python -m src.treatment_control                      # treat_t1/t3/t5 columns
python -m src.matching_day12                         # 1:1 PSM, 87,447 pairs
python -m src.did_day13                              # event-study DiD
python -m src.robustness_day14                       # placebo, caliper, dose, LOO
```

**Outputs**:

- Treatment/matched_pair_id columns in master_dataset
- `data/processed/did_panel.parquet`
- Tables `day13_eventstudy_*.csv`, `day14_*.csv`
- Figures `day13_eventstudy.png`, `day14_robustness.png`

## Phase 4: Statistical modeling + ML (Days 15–19)

```bash
python -m src.phenology_day16a                       # per-cell seasonal metrics
python -m src.drainage_proxy_day16b                  # OSM canal distance
python -m src.regression_day16c                      # Model A + Model B
python -m src.spatial_day17                          # Moran's I
python -m src.spatial_sarar_day17b                   # SARAR sensitivity
python -m src.xgboost_shap_day18a                    # XGBoost + SHAP
python -m src.clustering_day18b                      # k-means archetypes
python -m src.heterogeneity_day19                    # district breakdown
```

**Outputs**:

- `phenology.parquet`, `trajectory_clusters.parquet`
- All `day16*` through `day19*` tables and figures

## Known gotchas

- **Day 7 trajectories**: pandas Period arithmetic requires `.year*12 + month`
  instead of direct subtraction. See `_month_to_t` in `trajectories.py`.
- **Day 13 PanelOLS**: time index must be numeric, not string. Use `event_time`
  int, not `season` label. See `did_day13.py`.
- **Day 13 EVI**: winsorize at 0.1/99.9 percentiles — one cell (ID 733246)
  produces EVI value of 2×10¹⁰ due to denominator → 0 in cloud-contaminated
  pixel.
- **Day 16c Model A**: random effects covariance converges singular; coefficients
  unaffected but note the warning.
- **Day 16c Model B**: season must be converted to integer index for PanelOLS.
- **Day 17 SARAR**: `spreg.GM_Combo_Het` returns 2D betas (column vectors);
  index as `model.betas[i][0]`.
- **Day 18b k-means**: on Windows the Intel/LLVM OpenMP warning is cosmetic.

## Cross-reference

Every number in `docs/results_summary.md` traces to a specific CSV in
`results/tables/`. See the "Source:" lines in that doc.
