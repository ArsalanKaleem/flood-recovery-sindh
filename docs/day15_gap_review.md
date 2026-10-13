# Day 15 — Mid-Project Gap Review

**Project:** Post-2022 Sindh Flood Cropland Vegetation Recovery
**Phase:** Phase 3 — Counterfactual and Robustness Analysis
**Status:** Design review completed; empirical claims require verification against saved outputs.

## 1. Research question

How fast did cropland vegetation recover after the 2022 Sindh floods, and what environmental and contextual factors explain differences in recovery speed across space?

The project aims to move beyond flood-extent mapping by estimating seasonal vegetation trajectories relative to a matched, non-flooded counterfactual and investigating spatial variation in recovery.

## 2. Current analytical design

The planned primary analysis includes:

* Cropland cells classified using `<span>flood_duration >= 3</span>` as the primary treatment threshold.
* One-to-one propensity score matching using pre-flood covariates and vegetation characteristics.
* A cell-by-season panel of NDVI and EVI anomalies.
* An event-study Difference-in-Differences model with cell and season-year fixed effects.
* Robustness checks involving alternative flood thresholds, matching calipers, and placebo periods.

The design is intended to estimate differences in vegetation trajectories between flooded cells and comparable non-flooded cells. Its causal interpretation depends on the validity of the comparison group, pre-treatment trends, and other identification assumptions.

## 3. Preliminary findings requiring verification

The current project review reports the following estimates:

* An immediate NDVI difference of approximately -0.20.
* Partial recovery within one year, with a reported coefficient of approximately -0.07 by Rabi 2023–24.
* A persistent negative difference of approximately -0.11 in later seasons.
* Potential differences in vegetation response between Kharif and Rabi seasons.

These figures are preliminary reported results, not independently verified conclusions in this review. Before including them as established findings in the paper, verify the underlying model outputs, confidence intervals, reference season, event-time coding, matched sample, and robustness results.

A persistent negative coefficient should be described as a vegetation anomaly gap relative to the matched counterfactual. It does not, by itself, prove that flooding was the sole cause or that recovery will never be complete.

## 4. Planned explanatory variables

The following variables are reported as available in `<span>master_dataset.parquet</span>`:


| Variable                       | Planned role                                                   |
| ------------------------------ | -------------------------------------------------------------- |
| Flood duration                 | Primary treatment definition and exposure intensity            |
| Pre-flood NDVI level and trend | Matching and pre-treatment comparability                       |
| Soil moisture persistence      | Potential explanatory variable; measurement limitations remain |
| LST anomaly                    | Thermal conditions                                             |
| NDWI and SBI persistence       | Surface-water and spectral indicators                          |
| Elevation, slope, and TWI      | Terrain and topographic water accumulation                     |
| Rainfall anomaly               | Meteorological exposure                                        |
| Distance to drainage and roads | Geographic and accessibility context                           |
| ESA cropland                   | Cropland eligibility and sample definition                     |
| WorldPop                       | Population context, if theoretically justified                 |

Variable availability must be distinguished from successful validation and analytical usefulness. In particular, highly correlated flood-duration and soil-moisture measures may create multicollinearity or make their individual coefficients difficult to interpret.

## 5. Soil moisture limitation

Soil moisture remains a known limitation because the current measure represents the surface layer and may overlap substantially with flood duration.

The next analysis should examine its measurement definition, missingness, correlation with flood duration, and incremental explanatory value. If it contributes little independent information or is inadequately measured, exclude it from the primary explanatory model and document the limitation. A better-validated soil-moisture measure can be considered as a sensitivity analysis.

## 6. Novel contribution and literature positioning

The proposed contribution is a seasonal vegetation-recovery trajectory evaluated against a matched counterfactual, rather than a flood-damage or susceptibility map alone.

The originality of this contribution is not yet established merely by the difference in analytical approach. Before making a novelty claim, compare the study with relevant research on the 2022 Pakistan floods, vegetation recovery, remote-sensing-based impact assessment, and matched or quasi-experimental counterfactual designs.

The paper should state precisely what it adds: for example, the study area, temporal coverage, counterfactual design, recovery estimates, or analysis of environmental heterogeneity—only where supported by the final evidence.

## 7. Outstanding checks

Before advancing to the explanatory analysis:

* [ ]  Verify Day 13 event-study estimates against saved model outputs.
* [ ]  Confirm treatment timing and the fully pre-flood reference season.
* [ ]  Review pre-treatment coefficients and their uncertainty.
* [ ]  Verify Day 14 placebo, caliper, and treatment-threshold sensitivity results.
* [ ]  Confirm post-matching balance and the number of independent districts.
* [ ]  Check spatial dependence and the adequacy of the inference method.
* [ ]  Assess collinearity and measurement limitations among explanatory variables.
* [ ]  Collect relevant Pakistan Bureau of Statistics crop statistics for contextual discussion.
* [ ]  Update the literature comparison before asserting novelty.

## 8. Decision before Day 16

Proceed to the explanatory analysis using the pre-specified primary design. Preserve the original specification, document deviations, and report null or contradictory results honestly.

Day 16 should investigate whether environmental and contextual variables explain spatial variation in the observed recovery trajectories, without treating associations as causal mechanisms unless the design supports that interpretation.

**Day 15 conclusion:** The project has a coherent counterfactual-analysis direction. Its contribution and substantive conclusions will depend on verified estimates, credible identification, transparent robustness checks, and a defensible comparison with prior literature.
