# AGENTS.md — operating rules for agent sessions

Durable conventions for any agent working in this repository. Read this first.

## 1. Run locations & outputs (MANDATORY)

- **Always ask the user where work runs and where outputs go** at the start of
  any run-producing task (smoke test, operational run, batch).
- **Default output root:** `/pscratch/sd/t/tianyihu/efc-prediction` (use only
  if the user does not specify otherwise).
- **Never write outputs inside the git repo.** The repo `outputs/` directory is
  a git-ignored placeholder only.
- **Directory layout the engine creates:**
  ```
  {output-root}/{site}/{case-name}/{test|operational}/attempt{N}/
  ```
  - `test/` and `operational/` are kept separate.
  - Each run/retry goes to its own `attempt{N}/` (both test AND operational).
  - The engine creates the directories and selects the next free `attempt{N}`.

## 2. Testing policy

- New engine work is validated with a **smoke test** first.
- **Retry budget: 3.** If the smoke test fails, the agent may retry up to 3
  times. If still failing after 3 attempts, **stop and ask the user** for
  additional test approval before continuing.
- Each test attempt writes to its own `test/attempt{N}/` folder.

## 3. Environment

- Interpreter:
  `/global/common/software/m4803/conda/envs/OLMT_pm_TianyiPY311/bin/python`
  (has xarray, scikit-learn, joblib, pandas, numpy, scipy).
- The old CLM README's `/opt/anaconda3/envs/neon-efc/bin/python` does **not**
  exist on this machine — do not use it.

## 4. Code boundaries

- `clm_ref/` is **frozen reference** — do not edit; its embedded paths are
  intentionally broken.
- New development happens in `elm/` (engine) and `run_efc.py` (root caller).
- Reuse `predict_with_saved_cumulative70_models.py` logic and
  `model_artifacts/.../manifest.json` as the fixed Stage-2 contract.

## 5. Input data (ELM side)

- Read **run-folder per-variable NetCDF** (`H2OSOI.nc`, `TSOI.nc`, `NPP.nc`),
  NOT the `.pkl` (the pkl collapses soil layers to the top node).
- Soil node depths come from the `levgrnd` coordinate in the files — never
  assume depths.
- Hourly → daily means; `TSOI` K→°C; `NPP_daily` daily-mean flux (gC/m^2/s).

## 6. Handoff

- Keep `HANDOFF.md` current at the end of a working session: state, decisions,
  and the next concrete step.
