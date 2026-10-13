# Day 15 — Mid-Project Gap Review (revised Oct 2026)

## Status after Day 14

Phases 1–3 complete. Primary finding: matched flooded cells show
-0.20 NDVI shock in Kharif 2022, partial recovery to -0.06 by
Rabi 2023-24, then re-dampening to a persistent -0.11 gap through
Kharif 2026 (k=8). All four Day 14 robustness checks pass.

## Research question refinement

External audit (Oct 2026) raised concern about "apparent greening"
following TT-ORU preprint (district-level, Pakistan-wide, 2018-2024).
Review of TT-ORU code (pakistan_flood_vegetation_analysis.R)
confirms:

- TT-ORU operates at district-year level, not pixel-level
- No cropland masking, no matched counterfactual, no pre-trend test
- Headline finding: heavily-flooded districts show above-trend NDVI
  recovery by 2023-24, attributed to apparent (non-crop) greening

Our pixel-level matched-counterfactual design within ESA-masked
cropland provides the test TT-ORU's method cannot perform. Our
finding of persistent -0.11 within-cropland deficit through k=8 is
the within-cropland evidence that district aggregates cannot
provide. We now position our paper explicitly as resolving the
apparent-greening question TT-ORU raised.

## Enhancements planned for Phase 4

To make the apparent-greening resolution airtight:

- Day 16a: per-cell phenological decomposition (amplitude, peak
  timing, integrated greenness) to confirm NDVI anomaly reflects
  crop-cycle disruption not weed hyper-growth
- Day 16b: digitize LBOD/RBOD drainage infrastructure, add
  distance-to-drainage covariates to Day 16c regression

Core Days 1-14 analysis unchanged.

## Variables ready for Phase 4

All in master_dataset.parquet. See docs/data_dictionary.md.

## Open items

- PBS crop statistics still outstanding; needed for Day 22
  Discussion only
- SM persistence may be redundant with flood_duration in regression;
  evaluate on Day 16c, consider ERA5 soil_moisture_layer_3 refetch
  if needed
