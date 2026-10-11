# EFC Abundance Model — ELM Interface

Predict Extracellular Enzyme / Functional Capacity (EFC) abundance from land-model
output using a pre-trained Random-Forest bootstrap ensemble.

This repository drives the EFC model from **ELM** output produced by
[`elm-olmt`](/pscratch/sd/t/tianyihu/elm-olmt), replacing an earlier CLM-based
workflow. The ELM engine is written **from scratch** in `elm/`; the CLM code is
retained in `clm_ref/` **for reference only** and is not maintained.

---

## 1. Status

| Component | State |
|-----------|-------|
| Stage 2 — prediction (`predict_with_saved_cumulative70_models.py`) | **Ready** (to be wrapped by the engine) |
| Model artifacts (`model_artifacts/cumulative70_bootstrap500/`) | **Ready** (8 bundles + manifest) |
| CLM reference code (`clm_ref/`) | **Frozen** — do not edit |
| ELM engine (`elm/`) + root caller (`run_efc.py`) | **Plan finalized; scaffolding next** |
| README / AGENTS / HANDOFF + git tracking | **Current** |

---

## 2. Input data source — run-folder NetCDF (not the pkl)

**Decision (2026-10-11):** the engine reads the **per-variable NetCDF files in the
case run directory** (produced by `ncrcat` during elm-olmt postprocessing), **not**
the `.pkl`. The pkl collapses vertically-resolved variables to a single top layer
(`postprocess.py` slices `[:, index=0, gindex]`), so it cannot provide soil layers.
The run-folder `.nc` files retain the full vertical structure.

Verified against `ABBY_ctrlvertsoc_I20TRCNPRDCTCBC` (2026-10-11):

| File | dims | units | notes |
|------|------|-------|-------|
| `{run-dir}/H2OSOI.nc` | `(time=61320, levgrnd=15, lndgrid=1)` | `mm3/mm3` | soil moisture, all 15 layers |
| `{run-dir}/TSOI.nc`   | `(time=61320, levgrnd=15, lndgrid=1)` | `K`       | soil temperature, all 15 layers |
| `{run-dir}/NPP.nc`    | `(time=61320, lndgrid=1)`             | `gC/m^2/s`| scalar flux |

- `time` is hourly, expressed in **days** (step = 1/24). Coverage 2018–2024
  (7 years × 8760 h = 61320).
- `levgrnd` depths are read from the file coordinate (meters; e.g.
  0.0071, 0.0279, 0.0623, …), **never assumed**.

---

## 3. Architecture

A thin **root caller** drives an importable **engine package** in `elm/`:

```
run_efc.py                 # CLI entry point; parses arguments, drives the engine
  └── elm/                 # engine (importable package)
        io.py              #   read H2OSOI/TSOI/NPP .nc from run-dir; depths; hourly->daily
        config.py          #   parse elm-olmt .cfg (site, surffile, trans years, use_vertsoilc)
        surface.py         #   PCT_SAND/PCT_CLAY from surffile NetCDF; C/N from NEON csv
        features.py        #   trailing 7/30/365-day metrics (drydown + temperature), all layers
        build.py           #   nc + cfg + surfdata + csv -> consolidated per-case feature table
        predict.py         #   apply the 8 bootstrap bundles -> per-case NetCDF predictions
        paths.py           #   resolve/create run-location dirs (see §6)
```

### CLI (single case per invocation)

```bash
run_efc.py --site ABBY \
           --case-name ABBY_ctrlvertsoc \
           --config  /pscratch/.../ABBY/config/ABBY_ctrlvertsoc.cfg \
           --run-dir /pscratch/.../ABBY_ctrlvertsoc_I20TRCNPRDCTCBC/run \
           --neon-csv-root /pscratch/sd/t/tianyihu/CTSM_Hu/neon_surffiles \
           --output-root   /pscratch/sd/t/tianyihu/efc-prediction \
           --compset I20TRCNPRDCTCBC \   # placeholder; reserved for later use
           --mode test \                 # test | operational  (see §6)
           --use-vertsoilc true \        # default true (see §5)
           --stage all                   # inputs | predict | all
```

- `--run-dir` is **explicit** (the compset suffix in the folder name is not
  reliably reconstructable from the cfg).
- `--compset` is accepted but currently a **placeholder** for future multi-case use.
- Multiple cases per run are **not** supported yet — one case per invocation.

---

## 4. Pipeline overview

```
ELM run (elm-olmt)                      this repo
──────────────────                      ─────────
.cfg config  ───┐
                ├─▶  elm/ engine  ─▶  one consolidated         ─▶  elm/predict  ─▶  per-case
run-dir *.nc  ──┤    (Stage 1)        per-case feature table       (Stage 2)        NetCDF
NEON csv  ─────┤                       (long form, all layers)                       predictions
surfdata nc  ──┘
```

- **Stage 1 (`elm/build.py`):** read `H2OSOI/TSOI/NPP` from the run-dir `.nc`,
  aggregate hourly→daily (no-leap), engineer trailing 7/30/365-day hydrothermal
  features for **all 15 `levgrnd` layers**, attach soil texture + C/N, and write
  **one consolidated long-form feature table** per case.
- **Stage 2 (`elm/predict.py`):** load the 8 bootstrap bundles named in
  `model_artifacts/cumulative70_bootstrap500/manifest.json`, select each
  outcome's feature subset from the master table, and write a per-case **NetCDF**.

Daily aggregation: SM/ST = daily mean; `ST` converted K→°C; `NPP_daily` =
daily-mean flux in gC/m^2/s (matching the trained-model feature).

---

## 5. Soil layers — all `levgrnd` layers

The CLM workflow hardcoded the top 5 layers. The ELM engine uses **all layers
present in the run-dir files** (`levgrnd`, 15 layers for the verified case).

E3SM evidence (`components/elm/src/main/elm_varpar.F90`, lines 216–220):

```fortran
if (use_vertsoilc) then
   nlevdecomp      = nlevsoi      ! vertically resolved decomposition
   nlevdecomp_full = nlevgrnd
else
   nlevdecomp      = 1            ! single bulk layer
   nlevdecomp_full = 1
end if
```

The verified production case uses `use_vertsoilc = .true.`. Both `H2OSOI` and
`TSOI` are written on `levgrnd` (15) in the run-dir files, so the engine builds
features on **`levgrnd`**, with depths read from the data.

- `--use-vertsoilc` is a CLI argument, **default `true`**.
- Loud runtime check: if the loaded soil variables are single-layer
  (`use_vertsoilc=.false.` at run time), the engine raises rather than silently
  mis-handling the shape.

---

## 6. Run locations & output policy (repo rule)

**The agent ALWAYS asks the user where work runs and outputs go** at the start of
any run-producing task. **Default root: `/pscratch/sd/t/tianyihu/efc-prediction`.**

Layout (the engine creates these directories and selects the next free attempt):

```
/pscratch/sd/t/tianyihu/efc-prediction/
  {site}/
    {case-name}/
      test/
        attempt1/  attempt2/  ...     # smoke tests / dev runs; each retry isolated
      operational/
        attempt1/  attempt2/  ...     # production runs; re-runs get new attempts
```

- `test/` and `operational/` outputs are kept **separate**.
- Each retry (test or operational) goes to its own `attemptN/` subfolder.
- Outputs are **never** written inside the git repo; the repo `outputs/` path is a
  git-ignored placeholder only.

This rule is also recorded in `AGENTS.md` for agent sessions.

---

## 7. Other data sources

| Input | Source | Notes |
|-------|--------|-------|
| Site, case name, transient years, `use_vertsoilc`, surffile path | **`.cfg`** passed via `--config` | `[simulation] sites/case_names`, `[run_lengths] trans_startyear/nyears_trans`, `[case_options] surffile/use_vertsoilc`. |
| Soil texture `PCT_SAND`, `PCT_CLAY` | **surfdata NetCDF** referenced by `[case_options] surffile` | Read directly from the surface-data file. |
| Soil C / N / (C:N) | **`{neon-csv-root}/{SITE}_surfaceData.csv`** | NEON schema: `carbonTot`, `nitrogenTot`, `biogeoTopDepth/BottomDepth`, `bulkDensExclCoarseFrag`. C% = carbonTot/10, N% = nitrogenTot/10. |

- **Sites:** NEON — ABBY, JERC, RMNP, TALL, YELL, TEAK, WREF, SOAP, OSBS, …
- **Compset:** transient / historical — **`I20TRCNPRDCTCBC`** (`I20` family).

---

## 8. Outputs (simplified — not byte-compatible with CLM)

### Stage-1 feature table (per case) — one consolidated table
- Long form: `site, Date, soil_layer, soil_node_depth_cm, <engineered features...>`.
- **Parquet** (compact), with an optional CSV copy for inspection.
- The predictor selects each outcome's feature subset in-memory from the manifest.

Feature families (per layer, trailing W=7 / M=30 / Y=365 days):
`SM`, `ST`, `NPP_daily`, `SM_A_*_esm`, `SM_tau_*_esm`, `SM_theta_res_*_esm`,
`A_*_HT/LT/T`, `CV_*_T`, plus static `sand`, `clay`, `soil c (%)`, `soil n (%)`, `c/n`.
Drydown + temperature metrics are **ported into `elm/features.py`** (from
`clm_ref/efc_workflow.py`), so the engine has no legacy dependency.

### Stage-2 predictions (per case) — NetCDF
- One `.nc` per case; dims `(time, soil_layer)`; coord `soil_node_depth_cm(soil_layer)`.
- Variables per outcome × stat: `carbohydrate_mean/median/sd/q025/q975`, etc.,
  for the 8 manifest outcomes.
- Default: 5 summary stats; optional full 500-member `bootstrap` dimension (flag).
- Self-describing attributes (units, compset, provenance, bootstrap count);
  round-trips into `xarray`.

### Provenance
- One consolidated JSON per case (run-dir, cfg, surffile, NEON csv paths, compset,
  transient years, layer count + depths, date coverage, verified units).

---

## 9. Layout (repo)

```
efc-model-refactor/
  run_efc.py                   # root caller (CLI) — to build
  elm/                         # engine package — to build
    io.py config.py surface.py features.py build.py predict.py paths.py
  predict_with_saved_cumulative70_models.py   # existing Stage-2 (to be wrapped)
  model_artifacts/             # 8 bundles (git-ignored) + manifest.json (tracked)
  clm_ref/                     # frozen CLM reference (do not edit)
  AGENTS.md                    # agent operating rules (run-location policy, env)
  HANDOFF.md                   # current state + decisions for session handoff
  outputs/                     # git-ignored placeholder (real outputs go to efc-prediction)
```

---

## 10. Environment

```bash
/global/common/software/m4803/conda/envs/OLMT_pm_TianyiPY311/bin/python
```

Has xarray, scikit-learn, joblib, pandas, numpy, scipy. (The old CLM README's
`/opt/anaconda3/envs/neon-efc/bin/python` does not exist on this machine.)

---

## 11. Version control notes

- The ~2 GB `*.joblib` bundles, all generated outputs (in `efc-prediction/`, not
  the repo), and ELM `*.pkl`/`*.nc` inputs are **git-ignored**; only source, the
  small `manifest.json`, and docs are tracked.
- The ELM engine is developed fresh in `elm/` + `run_efc.py`; `clm_ref/` is
  read-only reference with intentionally broken embedded paths.
