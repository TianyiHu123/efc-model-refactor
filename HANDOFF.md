# HANDOFF — current state pointer

_Last updated: 2026-10-13_

Durable reference lives in **README.md** (pipeline, features §8a, outputs §8b,
key features §8c, plotting §8d) and **AGENTS.md** (operating rules). This file
is only a pointer to current state: what is done, what is active, what is next.

## State: ready for broader use

The ELM EFC engine (Stage 1 inputs + Stage 2 predict) and the plotting package
are implemented, reviewed, and smoke-tested on `ABBY_ctrlvertsoc`.

## Done

- Engine `elm/` + `run_efc.py`: cfg → run-dir `.nc` → features → Stage-1 table →
  Stage-2 NetCDF. Verified on ABBY (2190×15, 40 vars finite).
- First operational Slurm run succeeded (ABBY_ctrlvertsoc). Resource guidance in
  README §6b; example script `slurm/run_efc_single_case.slurm`.
- Plotting `elm/plotting.py` + CLI `--plot` / `--plot-only` / `--plot-vars` /
  `--plot-dir` (README §8d). Reviewed (APPROVED) and smoke-tested: lignin plotted
  from the existing operational attempt into
  `/pscratch/sd/t/tianyihu/efc-prediction/ABBY/ABBY_ctrlvertsoc/plot`.

## Active

- None. (Plotting changes not yet committed at last update.)

## Next / open

- Confirm EFC target **units** from the training source (AOGS_EFC Submission3)
  and replace the "unitless (placeholder)" label in `elm/plotting.py`.
- Run additional NEON sites/cases (watch the loud `io.py` validations).
- Optional: Slurm job array for many sites; small unit-test suite to lock the
  feature contract.

## Conventions

- Interpreter: `/global/common/software/m4803/conda/envs/OLMT_pm_TianyiPY311/bin/python`.
- Output policy (ask user; default `/pscratch/sd/t/tianyihu/efc-prediction`;
  `{site}/{case}/{test|operational}/attempt{N}`; never write in repo): README §6,
  AGENTS.md §1.
- One case per invocation; `--compset` is a placeholder for future multi-case.
