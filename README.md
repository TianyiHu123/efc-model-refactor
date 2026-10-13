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
| ELM engine (`elm/`) + root caller (`run_efc.py`) | **Built & smoke-tested** (ABBY) |
| Stage 1 — inputs (`elm/build.py` + `io`/`surface`/`features`) | **Done** |
| Stage 2 — prediction (`elm/predict.py`, wraps the saved bundles) | **Done** |
| Plotting (`elm/plotting.py`) | **Done** (reviewed + smoke-tested) |
| Operational Slurm run (`slurm/run_efc_single_case.slurm`) | **Verified** (ABBY) |
| Model artifacts (`model_artifacts/cumulative70_bootstrap500/`) | **Ready** (8 bundles + manifest) |
| CLM reference code (`clm_ref/`) | **Frozen** — do not edit |

The original `predict_with_saved_cumulative70_models.py` is retained as the
Stage-2 contract reference; the engine's `elm/predict.py` is the maintained path.
See `HANDOFF.md` for the current-state pointer (done / active / next).

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
        plotting.py        #   render PNG figures (EFC predictions + driver features)
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
           --stage all \                 # inputs | predict | all
           --jobs 2                      # Stage-2 predict parallelism (see §6b)
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
- The `.slurm` script and the Slurm `--output/--error` logs live **in the same
  `attemptN/` folder** as the scientific outputs (pre-create the attempt dir,
  submit from there, pass it to the engine via `--output-dir`). The repo
  `slurm/` holds examples only.
- Outputs are **never** written inside the git repo; the repo `outputs/` path is a
  git-ignored placeholder only.

This rule is also recorded in `AGENTS.md` for agent sessions.

---

## 6b. Compute & Slurm (operational runs)

**Operational runs go through Slurm, not the login node.** Stage 2 (prediction)
dominates wall time: on the verified ABBY case the full run is ~19 min (Stage 1
~1.4 min / ~0.1 GB; whole run peaks at ~4.6 GB RAM).

Measured from the first operational run (ABBY_ctrlvertsoc, job 59704245,
`seff`): **the workload is effectively near-serial** — ~137% CPU (~1.4 cores
busy on average), ~19 min wall regardless of cores allocated. `--jobs` applies
joblib thread parallelism across the 500 bootstrap members, but scikit-learn RF
`.predict` on these small inputs holds the GIL enough that threads barely
engage. **This is a performance limitation, not a correctness issue** — results
are numerically identical to serial (verified max abs diff 0.00e+00).

Practical guidance:

- **Do not over-allocate cores.** More cores do not reduce wall time here and
  only increase the `shared` charge and backfill wait. Recommended operational
  resources (the example-script defaults): **`--cpus-per-task=4`** (2 physical
  cores), **`--mem=8GB`** (observed ~4.6 GB), **`--time=00:40:00`** (observed
  ~19 min). `--jobs` is set to the physical-core count (2).
- **Pin BLAS/OpenMP to 1 thread** in the job (the example script exports
  `OMP_NUM_THREADS=1` etc.) to avoid nested oversubscription.
- Example single-case submission script:
  [`slurm/run_efc_single_case.slurm`](slurm/run_efc_single_case.slurm) —
  Perlmutter CPU, `--qos=shared`, `--account=m4803`, `--cpus-per-task=4`,
  `--mem=8GB`, `--time=00:40:00`, `--mode operational`. Submit from a
  pre-created attempt dir (see §6); case overridable via `SITE` / `CASE_NAME` /
  `CONFIG` / `RUN_DIR` env vars.
- One case per job. For many sites, wrap this in a Slurm **job array** (not
  included yet).

> **Known limitation / future work:** real speedup needs **process-based**
> parallelism across the 15 layers and/or 8 outcomes (loky backend), not threads
> across 500 tiny predicts. Until then, treat each case as ~20 min near-serial.
> Also: `shared`-queue waits were 1–6 h during testing and `squeue --start`
> estimates were unreliable — plan batches accordingly.

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

## 8a. Feature definitions (model inputs)

All features are computed **per soil layer, per prediction day** in
`elm/features.py` (dynamic/hydrothermal) and `elm/surface.py` (static soil
properties). Dynamic features use three **trailing windows** that *exclude the
current day*; the engine requires a 365-day warmup before the first prediction.

| Window label | Trailing length | Rolling smoothing (temperature) |
|--------------|-----------------|---------------------------------|
| `W` | 7 days | 1 day (no smoothing) |
| `M` | 30 days | 7-day centered mean |
| `Y` | 365 days | 7-day centered mean |

Below, `*` is a placeholder for the window label (`W` / `M` / `Y`).

### Instantaneous (current-day) hydrothermal
| Feature | Definition | Units |
|---------|------------|-------|
| `SM` | Daily-mean soil moisture (volumetric) at the current day | mm3/mm3 |
| `ST` | Daily-mean soil temperature at the current day (K→°C) | °C |
| `NPP_daily` | Daily-mean net primary productivity flux (site-level, repeated per layer) | gC/m^2/s |

### Soil-moisture drydown metrics (`drydown_metrics`, per window)
Computed from the trailing soil-moisture series by detecting monotonically
**decreasing** sub-sequences (length ≥ 3) with a 5%-of-range tolerance.
| Feature | Definition | Units |
|---------|------------|-------|
| `SM_A_*_esm` | Mean of the **minima** of the detected drydown sequences (asymptotic/residual moisture level, `A`). If no sequence is found, falls back to the window mean. | mm3/mm3 |
| `SM_tau_*_esm` | Mean **length** (in days) of the detected drydown sequences (drydown timescale, `tau`). 0 if none found. | days |
| `SM_theta_res_*_esm` | Mean **amplitude** (max − min) of the detected drydown sequences (drawdown magnitude, `theta_res`). 0 if none found. | mm3/mm3 |

### Soil-temperature metrics (`temperature_metrics`, per window)
Computed on the trailing temperature series converted to °C. `HT`/`LT` use a
rolling-window centered mean (see table above); `T`/`CV` use the raw °C series.
| Feature | Definition | Units |
|---------|------------|-------|
| `A_*_HT` | Max of the rolling-mean temperature over the window (high-temperature extreme) | °C |
| `A_*_LT` | Min of the rolling-mean temperature over the window (low-temperature extreme) | °C |
| `A_*_T` | Mean temperature over the window | °C |
| `CV_*_T` | Coefficient of variation of temperature over the window (`scipy.stats.variation`) | dimensionless |

### Static soil properties (per layer; constant in time)
| Feature | Source | Definition | Units |
|---------|--------|------------|-------|
| `sand` | surfdata `PCT_SAND` | Sand fraction (deepest value carried into bedrock `levgrnd` nodes) | % |
| `clay` | surfdata `PCT_CLAY` | Clay fraction (deepest value carried forward) | % |
| `soil c (%)` | NEON csv `carbonTot` | Soil carbon mass percent = `carbonTot / 10` | % |
| `soil n (%)` | NEON csv `nitrogenTot` | Soil nitrogen mass percent = `nitrogenTot / 10` | % |
| `c/n` | NEON csv | Carbon-to-nitrogen ratio = `carbonTot / nitrogenTot` | dimensionless |

The full engineered table (long form) therefore has, per layer per day: the 3
instantaneous columns, 7 window metrics × 3 windows = 21 dynamic columns, and 5
static columns, plus identifiers (`site`, `Date`, `soil_layer`,
`soil_node_depth_cm`) and `history_start` / `history_end`.

---

## 8b. Predicted variables (model outputs)

The ensemble predicts **8 EFC (extracellular enzyme / functional capacity)
outcomes**, each a cumulative enzyme-activity class. The 8 outcomes come from
`manifest.json`; each is written to the per-case NetCDF under a slugged name
(outcome lowercased, spaces → `_`).

| Outcome (manifest) | NetCDF variable prefix | Enzyme / functional group |
|--------------------|------------------------|---------------------------|
| Carbohydrate | `carbohydrate` | Carbohydrate (C-acquisition) degrading capacity |
| Lignin | `lignin` | Lignin / oxidative (recalcitrant-C) degrading capacity |
| Organic N | `organic_n` | Organic-nitrogen acquisition capacity |
| MonomerN | `monomern` | Nitrogen-monomer acquisition capacity |
| Inorganic N | `inorganic_n` | Inorganic-nitrogen acquisition capacity |
| Organic P | `organic_p` | Organic-phosphorus acquisition capacity |
| Monomer P | `monomer_p` | Phosphorus-monomer acquisition capacity |
| Inorganic P | `inorganic_p` | Inorganic-phosphorus acquisition capacity |

Each outcome is predicted by **500 bootstrap RandomForest members**. The NetCDF
stores, per outcome, 5 summary statistics across the 500 members, each on the
`(time, soil_layer)` grid:

| NetCDF variable | Statistic across the 500 bootstrap members |
|-----------------|----------------------------------------------|
| `<prefix>_mean` | Ensemble mean |
| `<prefix>_median` | Ensemble median |
| `<prefix>_sd` | Ensemble standard deviation (ddof=1) |
| `<prefix>_q025` | 2.5th percentile (lower 95% bound) |
| `<prefix>_q975` | 97.5th percentile (upper 95% bound) |

Optionally (with `--keep-bootstraps`) a full `<prefix>_bootstrap` variable on
`(time, soil_layer, bootstrap)` retains all 500 members.

> **Units of the outcomes:** the engine is unit-agnostic for the targets — it
> reproduces whatever the trained models output (the training target scale from
> `enzyme_meta_data_with_hydrothermal_11.csv`). The physical units of the
> predicted enzyme-activity values **cannot be concluded from this repo's
> code**; they are defined by the training dataset / source publication
> (AOGS_EFC Submission3), not by the engine. Flagging this so a plotting package
> can label axes correctly once the target units are confirmed.

---

## 8c. Key features for plotting (derived from the model contract)

**Source of truth:** the models are *already* feature-selected. The manifest's
`feature_source` is `selected15_cumulative70_features.json` ("cumulative70" =
the top predictors explaining a cumulative ~70% of RandomForest importance from
the training-time importance/robustness analysis). Each bundle's `features` list
in `manifest.json` **is** that outcome's retained important-feature set — so the
union below is the complete set of model-relevant inputs; everything else in the
Stage-1 table is unused by the models.

> **Scope note:** the per-feature importance *ranking/scores* were computed
> during training and are **not** stored in this repo (only the selected names
> survive in the manifest). A true per-feature importance ranking **cannot be
> concluded from this repo's code** without re-running an importance analysis on
> the loaded forests. The ranking below is by **cross-outcome usage frequency**
> (how many of the 8 models retained the feature), which is a defensible proxy
> for "generally important" and is fully derivable from the manifest.

### Feature usage across the 8 outcomes (from `manifest.json`)

| Feature | # outcomes using it (of 8) | Outcomes |
|---------|:--:|---------|
| `soil c (%)` | **8** | all |
| `c/n` | **6** | Lignin, MonomerN, Inorganic N, Organic P, Monomer P, Inorganic P |
| `A_Y_T` | **4** | Lignin, MonomerN, Inorganic N, Monomer P |
| `SM_tau_Y_esm` | 3 | Lignin, Inorganic N, Inorganic P |
| `NPP_daily` | 3 | Carbohydrate, MonomerN, Organic P |
| `clay` | 3 | Organic N, Inorganic N, Monomer P |
| `SM_theta_res_Y_esm` | 2 | Lignin, Organic P |
| `sand` | 2 | Organic P, Inorganic P |
| `SM_A_Y_esm` | 2 | Carbohydrate, Inorganic N |
| `SM` | 2 | MonomerN, Inorganic N |
| `SM_theta_res_M_esm` | 2 | Organic N, MonomerN |
| `SM_theta_res_W_esm` | 1 | Organic N |
| `SM_tau_W_esm` | 1 | Monomer P |
| `CV_Y_T` | 1 | Inorganic P |

### Recommended key features to plot

- **Primary (dominant, used by most/all models):**
  `soil c (%)` (8/8), `c/n` (6/8), `A_Y_T` (4/8). These three are the backbone
  predictors across the enzyme classes — a soil-chemistry pair plus the annual
  mean soil temperature.
- **Secondary (recurring hydrothermal/texture drivers):**
  `NPP_daily`, `SM_tau_Y_esm`, `clay`, `SM`, `SM_A_Y_esm`,
  `SM_theta_res_Y_esm`, `sand`. These capture productivity, annual soil-moisture
  drydown behavior, and texture.
- **Observation on window scale:** the retained dynamic features are
  overwhelmingly the **annual (`Y`) window** (drydown `SM_*_Y_esm`, temperature
  `A_Y_T` / `CV_Y_T`), with only `SM_theta_res_M/W` and `SM_tau_W` from shorter
  windows. For plotting, the annual-window metrics are the most model-relevant.

### Per-outcome feature sets (exact model inputs)

| Outcome | Features (trained order) |
|---------|--------------------------|
| Carbohydrate | `soil c (%)`, `SM_A_Y_esm`, `NPP_daily` |
| Lignin | `SM_theta_res_Y_esm`, `A_Y_T`, `c/n`, `SM_tau_Y_esm`, `soil c (%)` |
| Organic N | `ST`, `clay`, `soil c (%)`, `SM_theta_res_M_esm`, `SM_theta_res_W_esm` |
| MonomerN | `A_Y_T`, `SM`, `SM_theta_res_M_esm`, `SM_theta_res_Y_esm`, `NPP_daily`, `c/n`, `soil c (%)` |
| Inorganic N | `soil c (%)`, `A_Y_T`, `SM_A_Y_esm`, `c/n`, `SM`, `SM_tau_Y_esm`, `clay` |
| Organic P | `SM_theta_res_Y_esm`, `sand`, `c/n`, `NPP_daily`, `soil c (%)` |
| Monomer P | `A_Y_T`, `SM_tau_W_esm`, `clay`, `c/n`, `soil c (%)` |
| Inorganic P | `c/n`, `sand`, `CV_Y_T`, `soil c (%)`, `SM_tau_Y_esm` |

> Note: `ST` (current-day soil temperature) appears only for **Organic N**;
> it is the one instantaneous temperature feature retained by any model.

---

## 8d. Plotting (`elm/plotting.py`)

Renders PNG figures from a case's Stage-1/Stage-2 products
(`{case}_EFC_predictions.nc` + `{case}_features.parquet`, CSV fallback). Uses
the existing env (matplotlib, headless `Agg`), DPI 150, no new dependencies.

**Output layout — one self-contained folder per EFC outcome:**

```
{plot-dir}/{outcome_slug}/
  {outcome}_prediction_15layer.png   # EFC: 15 panels (one per layer), mean +
                                      # median (dashed) + q025-q975 shaded band
  <that outcome's driver features>    # from manifest.json features[]; shared
                                      # features are re-plotted in each folder
```

**Driver-feature figures** are dispatched by how the column varies in the
feature table (detected programmatically, not hardcoded):

| Feature kind | Example features | Figure |
|--------------|------------------|--------|
| static, depth-varying | `soil c (%)`, `c/n`, `sand`, `clay` | `*_profile.png` — value vs depth (log-depth, surface on top) |
| dynamic, per-layer | `A_Y_T`, `SM`, `ST`, `SM_*_esm`, `CV_Y_T` | `*_15layer.png` — 15-panel timeseries |
| dynamic, site-level | `NPP_daily` (identical across layers) | `*_timeseries.png` — single timeseries |

- EFC y-axes are labeled **"unitless (placeholder)"** (target units unknown from
  repo code; see §8b). Driver-feature axes use the units in §8a.
- Feature units/labels follow §8a; feature→folder membership follows the
  per-outcome sets in §8c.

### CLI

Plotting is driven through `run_efc.py`:

| Flag | Effect |
|------|--------|
| `--plot` | after Stage 2, render into `{output-dir}/plots` (no-op + warning under `--stage inputs`) |
| `--plot-only` | skip Stage 1/2; render from existing products in `--output-dir` (required) |
| `--plot-vars` | `all` (default) or comma list of outcome names/slugs, e.g. `lignin,carbohydrate` |
| `--plot-dir` | destination root (default `{output-dir}/plots`) |

```bash
# plot one outcome from an existing operational attempt (no recompute)
run_efc.py --plot-only \
  --site ABBY --case-name ABBY_ctrlvertsoc \
  --output-dir .../ABBY/ABBY_ctrlvertsoc/operational/attempt2 \
  --plot-dir   .../ABBY/ABBY_ctrlvertsoc/plot \
  --plot-vars  lignin
```

`--plot` is included in `slurm/run_efc_single_case.slurm`, so operational runs
emit figures alongside the scientific outputs.

---

## 9. Layout (repo)

```
efc-model-refactor/
  run_efc.py                   # root caller (CLI)
  elm/                         # engine package
    io.py config.py surface.py features.py build.py predict.py plotting.py paths.py
  predict_with_saved_cumulative70_models.py   # Stage-2 contract reference (not maintained)
  model_artifacts/             # 8 bundles (git-ignored) + manifest.json (tracked)
  slurm/                       # example Slurm submission scripts (single case)
  clm_ref/                     # frozen CLM reference (do not edit)
  AGENTS.md                    # agent operating rules (run-location policy, env)
  HANDOFF.md                   # current-state pointer (done / active / next)
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
