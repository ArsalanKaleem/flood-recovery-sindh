# Data Dictionary

Every variable, source, native resolution, and date range that enters the master analysis
dataset. Fill in as each layer is exported (Days 1-5) and revised during Phase 2 (Days 6-10).

## Spatial units


| Unit                 | Description                          | Source         | Extent                                                                    |
| -------------------- | ------------------------------------ | -------------- | ------------------------------------------------------------------------- |
| AOI district polygon | 7 Sindh flood-affected districts     | GAUL / OSM     | Dadu, Jamshoro, Larkana, Qambar-Shahdadkot, Jacobabad, Shikarpur, Sanghar |
| Analysis grid cell   | Stratified grid over cropland pixels | Derived, Day 6 | ~30-100 m, tuned for statistical power                                    |

## Vegetation indices


### Phase 1 asset inventory (complete)

All raw/derived layers now live as GEE assets under
`projects/flood-recovery-sindh/assets/`:

- `s2_monthly_composites/` — 69 months (2021-01 → 2026-09), NDVI + EVI, 30 m
- `flood_duration_2022` — single image, S1 both orbits, 30 m
- `chirps_rainfall_anomaly_2022` — single image, mm vs 2015-2021 baseline, 5 km
- `srtm_terrain` — elevation, slope, TWI approx, 30 m
- `esa_cropland_mask` — binary, class 40 cropland, 10 m
- `era5_soil_moisture/` — 68 months, volumetric_soil_water_layer_1, 1 km
- `era5_lst/` — 68 months, skin_temperature, 1 km


| Variable              | Source                   | Native res | Date range         | Notes                               |
| --------------------- | ------------------------ | ---------- | ------------------ | ----------------------------------- |
| NDVI (monthly median) | Sentinel-2 SR HARMONIZED | 10 m       | 2021-01 → present | s2cloudless cloud/shadow mask       |
| EVI (monthly median)  | Sentinel-2 SR HARMONIZED | 10 m       | 2021-01 → present | Applied on reflectance-scaled bands |
| NDVI (gap-fill)       | Landsat 8/9 SR C2 L2     | 30 m       | 2021-01 → present | Used mainly Aug-Oct 2022            |

## Flood variables


| Variable                      | Source            | Native res | Date range         | Notes                                          |
| ----------------------------- | ----------------- | ---------- | ------------------ | ---------------------------------------------- |
| Flood extent (binary)         | Sentinel-1 GRD VV | 10 m       | 2022-08 → 2022-10 | Threshold TBD; validated against JRC GSW       |
| Flood duration (days flagged) | Sentinel-1 GRD VV | 10 m       | 2022-08 → 2022-10 | Sum of per-scene flags, permanent water masked |

## Climate covariates


| Variable                 | Source              | Native res     | Date range         | Notes                         |
| ------------------------ | ------------------- | -------------- | ------------------ | ----------------------------- |
| Daily rainfall (mm)      | CHIRPS DAILY        | 0.05° (~5 km) | 2015-01 → present | For anomaly baseline          |
| Rainfall anomaly 2022    | Derived from CHIRPS | 0.05°         | 2022 monsoon       | mm vs. multi-year mean        |
| Soil moisture (0-7 cm)   | ERA5-Land Monthly   | 0.1° (~11 km) | 2021-01 → present | volumetric_soil_water_layer_1 |
| Land-surface temperature | ERA5-Land Monthly   | 0.1°          | 2021-01 → present | Waterlog / heat-stress proxy  |

## Terrain & soil


| Variable           | Source            | Native res | Date range | Notes                     |
| ------------------ | ----------------- | ---------- | ---------- | ------------------------- |
| Elevation          | SRTM v3           | 30 m       | static     |                           |
| Slope              | Derived from SRTM | 30 m       | static     | ee.Terrain.slope          |
| TWI                | Derived from SRTM | 30 m       | static     | Topographic wetness index |
| Soil texture proxy | TBD (SoilGrids?)  | TBD        | static     | Fill in Day 4             |

## Land cover & masking


| Variable         | Source              | Native res | Date range        | Notes                                       |
| ---------------- | ------------------- | ---------- | ----------------- | ------------------------------------------- |
| Cropland mask v1 | ESA WorldCover v200 | 10 m       | 2021              | Class 40 = cropland                         |
| Cropland mask v2 | Derived             | 10 m       | Day 6             | Intersected with PBS district ag boundaries |
| Permanent water  | JRC GSW v1.4        | 30 m       | occurrence ≥ 90% | Masked out of flood layer                   |

## Socioeconomic context


| Variable                   | Source                      | Native res | Date range       | Notes                      |
| -------------------------- | --------------------------- | ---------- | ---------------- | -------------------------- |
| Population density         | WorldPop                    | 100 m      | latest available |                            |
| Roads / settlements        | OpenStreetMap (osmnx)       | vector     | snapshot         | Distance-to-road covariate |
| District crop area / yield | PBS Agricultural Statistics | district   | annual           | Anchor for validation      |

## Derived recovery outcomes (Phase 2)


| Variable               | Definition                                                             | Notes                      |
| ---------------------- | ---------------------------------------------------------------------- | -------------------------- |
| Time-to-recovery       | Growing seasons until NDVI/EVI returns within θ of pre-flood baseline | θ TBD, sensitivity-tested |
| Cumulative deficit     | ∫ (baseline − observed) dt across shock-to-recovery window           | Signed, positive = deficit |
| Crop-cycle restoration | # of distinct growing-season peaks pre- vs. post-flood                 | Cropping intensity proxy   |

## Recovery drivers (Phase 4 explanatory variables)


| Variable               | Derivation                                    | Notes                               |
| ---------------------- | --------------------------------------------- | ----------------------------------- |
| flood_duration_days    | Day 3 raster, summarized per cell             | Primary treatment intensity         |
| waterlog_persistence   | ERA5 soil moisture anomaly persistence, Day 9 | Days of positive anomaly post-flood |
| salinity_proxy         | NDWI / soil-brightness persistence, Day 5     | Post-flood residual signature       |
| rainfall_anomaly_2022  | CHIRPS anomaly, Day 4                         | mm vs. baseline                     |
| elevation_m, slope_deg | SRTM derivatives, Day 4                       |                                     |
| dist_to_drainage_m     | OSM waterways + rivers                        | Day 5                               |

### Coverage QA (Day 2)

Monthly S2 composite coverage over the 22-district AOI was measured on
completion. Coverage ≥ 95% for 59 of 67 months. Flood-critical window
(Aug–Oct 2022): 83.6%, 99.7%, 99.7%. Landsat 8/9 gap-fill (originally
planned) was determined unnecessary given s2cloudless preservation.
Single gappy month: 2025-07 at 56.1% — handled via seasonal
decomposition during Day 7 time-series construction.
