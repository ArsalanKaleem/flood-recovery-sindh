# Resume notes — pause for quota reset

Paused at Day 9 complete, Day 10 pending covariate CSVs.

## When you come back

1. Check GEE tasks page: how many of the 110 covariate tasks finished?
   https://code.earthengine.google.com/tasks?project=flood-recovery-sindh
2. Download completed covariate CSVs from Drive → `flood-recovery-exports`.
   Expected files (22 of each prefix):

   - terrain_<District>.csv
   - rainfall_<District>.csv
   - ndwi_persist_<District>.csv
   - sbi_persist_<District>.csv
   - worldpop_<District>.csv
3. Any that failed, resubmit after Nov 1 quota reset.
4. Then run Day 10 integration script (not yet written — Claude will provide).
5. Then proceed to Day 11 (treatment classification).

## Open design decision

AI/ML extension (deferred until Day 18):

- Confirmed: XGBoost + SHAP on Day 18 (replacing bare RF cross-check)
- Pending choice: whether to add Day 18.5 trajectory clustering (archetype map)

## Key files

- Master data lives in `data/processed/` (not yet master_dataset.parquet)
- Recovery metrics: `recovery_metrics.parquet`
- SM persistence: `sm_persistence.parquet`
- Trajectories: `data/processed/trajectories/*.parquet` (22 shards)
