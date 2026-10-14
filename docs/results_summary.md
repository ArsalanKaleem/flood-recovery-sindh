# Results Summary

Freeze of all quantitative findings from Phases 1–4. Every number here can be
traced back to a specific results table or figure in `results/`.

Last updated: Day 20 freeze.

## Study scope

- **AOI**: 22 agricultural districts of Sindh, Pakistan (GADM 4.1 Level 3)
- **Cropland cells**: 555,309 at 300m resolution (ESA WorldCover ≥50% cropland)
- **Period**: January 2021 – September 2026 (69 months)
- **Treated cells (primary definition, flood_duration ≥ 3)**: 87,948 (15.8%)
- **Matched pairs (Day 12)**: 87,447 (99.4% treated-cell match rate)
- **Matched sample size (both members)**: 174,894 cells

## Day 12 matching diagnostics

- Caliper: 0.1875 logit propensity units (0.2 × SD)
- Covariates matched: elevation, slope, TWI, dist_to_drainage, cropland_frac,
  preflood_ndvi_level, preflood_ndvi_trend
- **All 7 covariates balanced to |SMD| < 0.03** post-matching
  (vs 1/7 balanced pre-matching)

Source: `results/tables/day12_balance_before.csv`, `day12_balance_after.csv`.

## Day 13 primary event-study DiD

Spec: cell × season panel, cell + season fixed effects, SE clustered by district.
Reference season: Kharif 2021 (k = −2).

- **Pre-trend check (k = −1)**: NDVI coef = +0.0033 (p = 0.43), EVI +0.0006 (p = 0.85). **Passes parallel trends.**
- **Shock (k = 0, Kharif 2022)**: NDVI = −0.2015 (SE 0.0096), EVI = −0.0973 (SE 0.0081)
- **Best recovery point (k = 3, Rabi 2023-24)**: NDVI = −0.0576
- **Persistent deficit at k = 8 (Kharif 2026)**: NDVI = −0.1167, EVI = −0.0785

Source: `results/tables/day13_eventstudy_ndvi.csv`, `day13_eventstudy_evi.csv`.
Figure: `day13_eventstudy.png`.

## Day 14 robustness

- **14a Placebo test** (fake shock Kharif 2021): coef +0.0033, p = 0.56. **Passes.**
- **14b Caliper sensitivity**: shock coef at 0.5x = −0.2016, baseline = −0.2015,
  2x = −0.2015. **All within 0.0001.**
- **14c Dose-response terciles** (by flood_duration):
  - Low tercile shock = −0.131
  - Mid tercile shock = −0.193
  - High tercile shock = −0.280
    **Monotonic by exposure intensity.**
- **14d Leave-one-district-out**: shock range across 22 LOO fits = −0.2043 to −0.1931.
  **No single district drives the finding.**

Source: `day14_placebo.csv`, `day14_caliper_*.csv`, `day14_dose_*.csv`, `day14_loo.csv`.
Figure: `day14_robustness.png`.

## Day 16a phenological decomposition

Per-cell, per-season metrics: amplitude, peak_month, integrated_ndvi, LOS.

### Kharif-Rabi recovery asymmetry (integrated NDVI, treated − control):


| Season              | Diff        | Verdict                 |
| ------------------- | ----------- | ----------------------- |
| Kharif 2022 (shock) | −1.383     | mixed                   |
| Rabi 2022-23        | −0.429     | mixed                   |
| Kharif 2023         | **−0.445** | real deficit            |
| Rabi 2023-24        | **+0.185**  | treated exceeds control |
| Kharif 2024         | **−0.604** | real deficit            |
| Rabi 2024-25        | **+0.240**  | treated exceeds control |
| Kharif 2025         | **−0.468** | real deficit            |
| Rabi 2025-26        | **+0.153**  | treated exceeds control |
| Kharif 2026         | **−0.298** | real deficit            |

**Interpretation**: Flooded cropland shows seasonally decoupled recovery:
Kharif (summer, monsoon-rainfed) stays persistently suppressed; Rabi (winter,
canal-irrigated) matches or exceeds matched controls starting one year post-flood.

Source: `day16a_phenology_comparison.csv`. Figure: `day16a_phenology.png`.

## Day 16b drainage proxy

OSM structural canals ≥10 km (409 segments). Median cell distance to major
drainage: 2,477 m. Treated vs control median: 2,587m vs 2,453m (treated slightly
farther, consistent with drainage-failure hypothesis).

## Day 16c Model A — mixed-effects regression on TTR

Spec: `TTR ~ 11 standardized covariates + (1 | district)`.
N = 174,161 matched cells. Converged, group variance = 4.83.

### Standardized coefficients (ranked by |β|, all p < 0.001 unless noted):


| Variable            | β        | Interpretation                                            |
| ------------------- | --------- | --------------------------------------------------------- |
| flood_duration      | **+3.72** | Longer flood → longer TTR                                |
| ndwi_persist_months | **+3.33** | Persistent waterlogging slows recovery                    |
| elevation_m         | −0.74    | Higher ground → faster recovery                          |
| log_dist_drainage   | +0.64     | Farther from drainage → longer TTR                       |
| slope_deg           | +0.57     | —                                                        |
| log_pop             | −0.50    | More populated → faster recovery (human intervention)    |
| rainfall_anomaly_mm | −0.43    | —                                                        |
| lst_anom_K          | −0.40    | Warmer → faster recovery (plant growth)                  |
| sbi_persist_months  | −0.31    | Wrong sign, confounded with flood exposure                |
| sm_persist_months   | −0.18    | Wrong sign, ERA5 scale 1 is redundant with flood_duration |
| log_dist_road       | −0.06    | Marginal (p = 0.011)                                      |

Source: `day16c_model_a_ttr.csv`.

## Day 16c Model B — Kharif-Rabi interaction formalized

Spec: `integrated_ndvi ~ flood_duration × is_kharif + cell_FE + season_FE`.
Post-flood panel only, 1,574,040 observations.

- **Interaction β = −0.321** (SE 0.051), **p = 4.4 × 10⁻¹⁰**
- Each 1-SD increase in flood_duration causes an **additional** 0.321-unit
  integrated NDVI deficit in Kharif compared to Rabi.

**The Kharif-Rabi asymmetry is a formal, highly significant finding.**

Source: `day16c_model_b_asymmetry.csv`.

## Day 17 spatial robustness

- **Moran's I** on Model A residuals = 0.167 (K = 8, N = 25,000). Moderate
  spatial autocorrelation.
- **SARAR (spatial lag + error) sensitivity** on 10,000-cell random sample:
  - Spatial error λ = 0.46 (p < 0.001) — meaningful spatial error correlation
  - Spatial lag ρ = 0.04 (p = 0.44) — no significant spatial lag
  - **Primary coefficients stable**: flood_duration β 3.72 → 3.78 (+1.5%),
    ndwi_persist β 3.33 → 3.39 (+1.9%). **Dominant drivers unaffected by
    spatial correction.**

Source: `day17_moran.csv`, `day17b_sarar_sensitivity.csv`.

## Day 18a XGBoost + SHAP

Spec: XGBoost, max_depth=6, early stopping. Test R² = 0.438, MAE = 3.52 months.

### SHAP mean |value| ranking (top 5):

1. flood_duration: 3.10
2. ndwi_persist_months: 1.95
3. lst_anom_K: 0.83
4. elevation_m: 0.80
5. rainfall_anomaly_mm: 0.38

Rank concordance with Model A |β| (Kendall's τ): **+0.527, p = 0.026**.

### Nonlinear threshold findings:

- **flood_duration SHAP curve saturates near 5 SAR detections** (~60 days inundation)
- **NDWI persistence SHAP curve accelerates past 4 months** (compounding effect)
- LST anomaly shows U-shape (both high and low LST associated with longer TTR)

Source: `day18a_shap_vs_model_a.csv`. Figures: `day18a_shap_summary.png`,
`day18a_shap_dependence_top3.png`.

## Day 18b trajectory archetypes

K-means on 11-season NDVI anomaly vectors. k = 3 by silhouette (0.25).


| Archetype                 | n       | % total | % treated | Mean flood_dur | Shape                                |
| ------------------------- | ------- | ------- | --------- | -------------- | ------------------------------------ |
| Persistent deficit        | 29,365  | 17%     | 82%       | 7.1            | V-shaped, never recovers past −0.14 |
| Overshoot/intensification | 44,964  | 26%     | 32%       | 2.6            | Monotonically climbs to +0.43        |
| Resilient recovery        | 100,565 | 58%     | 49%       | 3.8            | Shallow V, back to +0.05 by k=2      |

**Geographic finding**: persistent deficit concentrates in the central-west Indus
corridor (Larkana, Dadu, Jamshoro, Jacobabad).

Source: `day18b_cluster_summary.csv`. Figures: `day18b_archetype_trajectories.png`,
`day18b_archetype_map.png`.

## Day 19 district heterogeneity

### Archetype composition — districts with highest persistent-deficit share:

1. **Matiari: 55%** (standout — majority of its matched cells are chronically broken)
2. Sukkur: 31%
3. Hyderabad: 28%
4. Naushahro Firoz: 25%
5. Jamshoro: 25%

### Policy concentration — persistent-deficit cells by district:

- **Larkana: 5,700 cells (19% of all persistent deficit)**
- **Dadu: 5,008 cells (17%)**
- Jacobabad: 2,940 cells (10%)
- Naushahro Firoz: 1,565 cells (5%)

**Larkana + Dadu together account for 36% of all persistently-broken cropland
across the 22-district AOI.**

### Persistent-deficit cell characteristics (mean):

- Flood duration: 7–9 SAR detections (worst exposure)
- Distance to drainage: 2.5–8 km (farther than AOI median of 2.5 km)
- Elevation: 26–63 m (low-lying)
- Population density: moderate-to-high

Source: `day19_per_district_summary.csv`, `day19_archetype_by_district.csv`,
`day19_persistent_deficit_characterization.csv`. Figures: `day19_district_lattice.png`,
`day19_archetype_heatmap.png`.

## Headline findings for abstract

1. Flooded cropland shows a **−0.20 NDVI shock** in Kharif 2022, partially
   recovers to −0.06 by Rabi 2023-24, and remains **−0.11 below matched controls
   through 4+ years** post-flood (p < 10⁻⁹).
2. Recovery is **seasonally decoupled**: Kharif remains persistently suppressed
   (−0.30 to −0.60 integrated NDVI) while Rabi exceeds matched controls from
   year 2 onwards. Interaction coefficient: −0.32, p < 10⁻⁹.
3. **17% of matched cropland cells** show chronic 4-year deficit; these
   concentrate in Larkana and Dadu (combined 36% of total) and are
   characterized by low elevation + distance from structural drainage.
4. Linear and nonlinear specifications (mixed-effects, XGBoost+SHAP, SARAR
   spatial correction) all identify **flood duration and NDWI persistence** as
   the dominant TTR drivers; flood_duration shows a nonlinear saturation past
   ~5 SAR detections.
5. The results provide the **pixel-level matched counterfactual within cropland
   masks** that TT-ORU 2026 raised as needed to resolve the apparent-greening
   question at district aggregates. Our within-cropland evidence confirms real
   recovery deficits persist 4+ years post-flood despite potentially
   above-trend aggregate greenness.
