# HANDOFF — current state

_Last updated: 2026-10-11_

## Where we are

Planning is **finalized**. Next concrete step: **scaffold the `elm/` engine and
root caller `run_efc.py`, then run a smoke test** against the ABBY verified case.
No engine code exists yet.

## Verified reference case (use for the smoke test)

- Site: `ABBY`; case name: `ABBY_ctrlvertsoc`; compset: `I20TRCNPRDCTCBC`.
- Config: `/pscratch/sd/t/tianyihu/E3SM_out/SOIL_project/NEON_ctrl/ctrlvertsoc/ABBY/config/ABBY_ctrlvertsoc.cfg`
  - `use_vertsoilc = .true.`, `trans_startyear = 2018`, `nyears_trans = 7`,
    `surffile = /pscratch/.../CTSM_NEON/ABBY/surfdata_1x1_NEON_ABBY_hist_2000_c260408.nc`.
- Run dir: `/pscratch/sd/t/tianyihu/E3SM_out/SOIL_project/NEON_ctrl/ctrlvertsoc/ABBY/ABBY_ctrlvertsoc_I20TRCNPRDCTCBC/run`
  - `H2OSOI.nc` `(time=61320, levgrnd=15, lndgrid=1)` units `mm3/mm3`
  - `TSOI.nc`   `(time=61320, levgrnd=15, lndgrid=1)` units `K`
  - `NPP.nc`    `(time=61320, lndgrid=1)` units `gC/m^2/s`
  - `time`: hourly, in days (step 1/24); `levgrnd` depths in meters from coord.
- NEON csv: `/pscratch/sd/t/tianyihu/CTSM_Hu/neon_surffiles/ABBY_surfaceData.csv`.

## Settled decisions

1. **Data source = run-folder `.nc`** (NOT pkl). Pkl collapses to top layer only.
2. **All `levgrnd` layers** (15 here). Depths read from the file coordinate.
3. **Hourly → daily means**; `TSOI` K→°C; `NPP_daily` daily-mean gC/m^2/s.
4. **CLI (one case per run):** `--site --case-name --config --run-dir`
   (explicit) `--compset` (placeholder, default `I20TRCNPRDCTCBC`)
   `--neon-csv-root --output-root --mode {test|operational} --use-vertsoilc`
   (default true) `--stage {inputs|predict|all}`.
5. **Outputs:** Stage-1 consolidated long-form table (Parquet + optional CSV);
   Stage-2 per-case NetCDF (dims `(time, soil_layer)`, 5 summary stats, optional
   bootstrap dim).
6. **Output location policy:** always ask user; default
   `/pscratch/sd/t/tianyihu/efc-prediction`; layout
   `{root}/{site}/{case-name}/{test|operational}/attempt{N}/`; engine creates
   dirs + picks next free attempt; never write inside repo. (See AGENTS.md.)
7. **Feature engineering** ported from `clm_ref/efc_workflow.py`
   (`_drydown_metrics`) + temperature metrics into `elm/features.py`; no legacy
   dependency.
8. **Stage-2 contract fixed** by `model_artifacts/.../manifest.json` (8 outcomes,
   per-outcome feature lists) and `predict_with_saved_cumulative70_models.py`.

## Open / to confirm at build time

- Exact mapping of `taxis`/`time` (days) → calendar `Date` (no-leap).
- First smoke test must assert: 15 layers, 2018–2024 coverage, units match,
  no NaNs in model features after 365-day warmup, successful NetCDF write.

## Next step

Scaffold `elm/{io,config,surface,features,build,predict,paths}.py` + `run_efc.py`,
then smoke-test on ABBY (output to
`/pscratch/sd/t/tianyihu/efc-prediction/ABBY/ABBY_ctrlvertsoc/test/attempt1/`).
Retry budget 3; if exhausted, ask the user.
