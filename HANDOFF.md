# HANDOFF — current state

_Last updated: 2026-10-11_

## Where we are

The ELM EFC engine is **built, independently reviewed, and smoke-tested** on the
ABBY `ctrlvertsoc` case. End-to-end run (Stage 1 + Stage 2) succeeds and all
output checks pass. Review findings addressed. Ready for broader use (more
sites/cases) and any follow-on work.

## What exists

- `run_efc.py` — root caller (CLI). Args: `--site --case-name --config
  --run-dir --neon-csv-root --output-root --compset(placeholder, default
  I20TRCNPRDCTCBC) --mode{test|operational} --use-vertsoilc(default true)
  --stage{inputs|predict|all} --keep-bootstraps --no-csv`.
- `elm/` engine:
  - `config.py` — tolerant `.cfg` parser (inline comments, quotes, Fortran bools).
  - `paths.py` — `{root}/{site}/{case}/{mode}/attempt{N}` resolver (auto next-N).
  - `io.py` — reads run-dir `H2OSOI.nc/TSOI.nc/NPP.nc`, hourly→daily (noleap),
    depths from `levgrnd`. Validates time epoch==start_year, noleap calendar,
    and a complete contiguous 24/day grid. Loud single-layer guard.
  - `surface.py` — texture from surfdata (`nlevsoi`→`levgrnd` with deepest
    carried forward); C/N from NEON csv by depth horizon; reports below-horizon
    layer count.
  - `features.py` — trailing W/M/Y drydown + temperature metrics; a faithful,
    verified-identical port of `clm_ref`.
  - `build.py` — Stage-1 consolidated long-form table (Parquet + optional CSV) +
    provenance JSON.
  - `predict.py` — applies the 8 bootstrap bundles → per-case NetCDF
    `(time, soil_layer)`, 5 summary stats (+ optional bootstrap dim).

## Verified smoke test (ABBY ctrlvertsoc)

Output: `/pscratch/sd/t/tianyihu/efc-prediction/ABBY/ABBY_ctrlvertsoc/test/attempt{1,2}/`
- Stage-1 table: 32850 rows = 15 layers × 2190 days, coverage 2019-01-01→2024-12-31
  (365-day warmup consumes 2018; source 2018–2024).
- NetCDF: dims `(time=2190, soil_layer=15)`, 40 vars (8 outcomes × 5 stats),
  all finite; coords `time/soil_layer/soil_node_depth_cm`; provenance attrs.
- Provenance: `texture_native_nlevsoi=10`, `texture_extended_to_levgrnd=True`,
  `cn_below_deepest_horizon_layers=6`.
- All modules `py_compile` clean.

## Independent review (sub-agent) — resolved

Verdict was NEEDS_CHANGES; the two MAJOR items and four MINORs were fixed:
- **M1** (dates were synthesized from cfg): `io._parse_time_reference` now reads
  the file's `units`/`calendar`, asserts noleap, Jan-1 epoch, and epoch
  year == `trans_startyear`.
- **M2** (no completeness check): `io._validate_daily_grid` asserts day index
  starts at 0, is contiguous, and has a uniform 24 samples/day.
- m1 case-name mismatch warning; m2 explicit gridcell/level axis in
  `read_texture`; m3 below-horizon C/N count in provenance; m4 fail loudly if no
  table is written.
- Confirmed by reviewer: `features.py` numerically identical to `clm_ref`;
  Stage-2 summary stats match the reference.

## Conventions (unchanged; see AGENTS.md)

- Interpreter: `/global/common/software/m4803/conda/envs/OLMT_pm_TianyiPY311/bin/python`.
- Output policy: ask user; default `/pscratch/sd/t/tianyihu/efc-prediction`;
  `{site}/{case}/{test|operational}/attempt{N}`; never write in repo.
- One case per invocation; `--compset` is a placeholder for future multi-case.

## Possible next steps

- Run additional NEON sites/cases (watch the new loud validations — they will
  catch any case whose time epoch ≠ `trans_startyear` or with partial days).
- Optional: a small unit-test suite (config parsing, noleap dates, drydown
  equality vs clm_ref) to lock the contract.

## Parallelization & Slurm (added 2026-10-11)

- `run_efc.py --jobs N` sets the RandomForest thread count for Stage-2 predict
  (runtime `n_jobs` override on loaded estimators). Verified numerically
  equivalent to serial (max abs diff 3.5e-18 ≈ 1 float64 ULP, from parallel
  mean-summation order). Default 1.
- Timing (ABBY): Stage 1 ~1.4 min / 0.1 GB; full run ~19 min / 2.5 GB, Stage 2
  dominates. On the 2-core login node `--jobs 2` was *slower* (oversubscription)
  — real speedup needs dedicated compute cores → use Slurm.
- `slurm/run_efc_single_case.slurm`: example Perlmutter single-case job
  (`--qos=shared --account=m4803 --constraint=cpu --cpus-per-task=32`,
  `--mode operational --jobs=$SLURM_CPUS_PER_TASK`). Validated with `bash -n`
  and path checks; **not submitted** (operational action for the user). The
  end-to-end Slurm run itself is therefore unverified.
- One case per job; job array for many sites is a future addition.

## First operational run — ABBY_ctrlvertsoc (2026-10-12)

- **Succeeded on attempt 2.** Outputs in
  `/pscratch/sd/t/tianyihu/efc-prediction/ABBY/ABBY_ctrlvertsoc/operational/attempt2/`
  (NetCDF 2190×15, 40 vars all finite; features Parquet/CSV identical to
  attempt1; provenance; Slurm script + logs co-located).
- **Attempt 1 (job 59702073): TIMEOUT** at 30 min, `seff` CPU eff 3.76%. Root
  cause: Stage-2 set `RandomForestRegressor.n_jobs=N` per model, which *slows*
  the 500 tiny-input predicts (verified n_jobs=16 was 2× slower than serial).
- **Fix (committed):** `elm/predict.py` now forces each model `n_jobs=1` and
  parallelizes ACROSS the 500 bootstrap members with
  `Parallel(n_jobs, prefer="threads")`. Numerically EXACT vs serial
  (max abs diff 0.00e+00, allclose). Added BLAS/OpenMP=1 env pinning in the
  Slurm script to prevent nested oversubscription.
- **Attempt 2 (job 59704245): COMPLETED**, wall 19:03, 8 cores, `seff` 8.59%,
  mem 4.63 GB/16 GB.

### Efficiency finding + resource recommendation
- Even with the fix, `/usr/bin/time` shows ~137% CPU (~1.4 cores avg) and wall
  ≈ serial Stage-2 time. The thread-across-models parallelism barely engages —
  scikit-learn RF `.predict` on tiny inputs holds the GIL enough to serialize.
  **This is a performance limitation, not a correctness issue.**
- **Recommended operational resources (now the example-script defaults):**
  `--cpus-per-task=4` (2 physical cores), `--mem=8GB`, `--time=00:40:00`
  (observed 19 min). Over-allocating cores wastes `shared` charge and worsens
  backfill wait for no wall-time gain.
- **Future optimization (not blocking):** for real speedup use process-based
  parallelism across the 15 layers and/or 8 outcomes (loky backend), rather than
  threads across 500 tiny predicts.
- Queue reality on `shared`: waits ranged 1–6 h during this exercise;
  `squeue --start` estimates were unreliable (withdrawn/overshot).

## Operational run convention (added 2026-10-11)

- **Script + Slurm logs live in the attempt output folder**, co-located with the
  scientific outputs: `{root}/{site}/{case}/{test|operational}/attempt{N}/`.
  Pre-create the attempt dir, put the `.slurm` script there, submit from there
  (so `#SBATCH --output/--error` land there), and pass that dir to the engine.
- `run_efc.py --output-dir DIR` (new) overrides the auto-selected attempt so the
  engine writes into the pre-created folder. Verified working (stage inputs
  wrote to an explicit dir). Default behavior (auto attempt{N}) unchanged when
  `--output-dir` is omitted.
- Repo `slurm/run_efc_single_case.slurm` updated to this pattern (submit from the
  attempt dir; `OUTPUT_DIR=$SLURM_SUBMIT_DIR`). AGENTS.md §1 and README §6 updated.
- Retry-budget rule: ask the user each new task; default 3.

## In-flight operational job (ABBY_ctrlvertsoc)

- Attempt dir: `/pscratch/sd/t/tianyihu/efc-prediction/ABBY/ABBY_ctrlvertsoc/operational/attempt1/`
  (contains `run_efc_ABBY_ctrlvertsoc.slurm`, `.jobid`).
- First submission 59688781 was CANCELLED (still PENDING, 0 compute) to adopt the
  attempt-folder convention; resubmitted as **59691362**.
- Status at last check: PENDING, reason progressed Priority→Licenses→Resources
  (normal `shared`-partition queue wait, not a failure; no retry budget consumed).
- Remaining: monitor to terminal, then `seff`/`sacct` efficiency review +
  resource-allocation recommendation + update example script walltime/cpus; then
  audit outputs (NetCDF dims/finite, feature rows, provenance).
