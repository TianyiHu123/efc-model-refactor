# EFC Abundance Model — ELM Interface

Predict Extracellular Enzyme / Functional Capacity (EFC) abundance from land-model
output using a pre-trained Random-Forest bootstrap ensemble.

This repository drives the EFC model from **ELM** output produced by
[`elm-olmt`](/pscratch/sd/t/tianyihu/elm-olmt), replacing an earlier CLM-based
workflow. The ELM engine is being written **from scratch** in `elm/`; the CLM
code is retained in `clm_ref/` **for reference only** and is not maintained.

---

## 1. Status

| Component | State |
|-----------|-------|
| Stage 2 — prediction (`predict_with_saved_cumulative70_models.py`) | **Ready** (to be wrapped by the engine) |
| Model artifacts (`model_artifacts/cumulative70_bootstrap500/`) | **Ready** (8 bundles + manifest) |
| CLM reference code (`clm_ref/`) | **Frozen** — do not edit |
| ELM engine (`elm/`) + root caller (`run_efc.py`) | **Not started** — blocked on NPP (see §5) |
| README + git tracking | **Current** |

---

## 2. Architecture

A thin **root caller** drives an importable **engine package** in `elm/`:

```
run_efc.py                 # CLI entry point; parses arguments, drives the engine
  └── elm/                 # engine (importable package)
        io.py              #   load ELMcase pkl; extract .output, taxis, provenance
        config.py          #   parse elm-olmt .cfg (sites, surffile, trans years, use_vertsoilc)
        surface.py         #   PCT_SAND/PCT_CLAY from surffile NetCDF; C/N from NEON csv
        features.py        #   trailing 7/30/365-day metrics (drydown + temperature), all layers
        build.py           #   pkl + cfg + surfdata + csv -> consolidated per-site feature table
        predict.py         #   apply the 8 bootstrap bundles -> per-site NetCDF predictions
```

### CLI (planned)

```bash
run_efc.py --sites ABBY JERC RMNP \
           --compset I20TRCNPRDCTCBC \
           --pkl-root     /pscratch/sd/t/tianyihu/elm-olmt/pklfiles \
           --config-root  /pscratch/sd/t/tianyihu/elm-olmt/config_files \
           --neon-csv-root /pscratch/sd/t/tianyihu/CTSM_Hu/neon_surffiles \
           --output-root  ./outputs \
           --use-vertsoilc true \        # default true (see §4)
           --stage all                   # inputs | predict | all
```

---

## 3. Pipeline overview

```
ELM run (elm-olmt)                     this repo
──────────────────                     ─────────
.cfg config  ─┐
              ├─▶  elm/ engine  ─▶  one consolidated          ─▶  elm/predict  ─▶  per-site
pkl output  ──┤    (Stage 1)        per-site feature table        (Stage 2)        NetCDF
NEON csv  ───┤                      (long form, all layers)                        predictions
surfdata nc ─┘
```

- **Stage 1 (`elm/build.py`):** read ELM `.pkl` output + metadata, engineer the
  trailing 7/30/365-day hydrothermal features for **all soil layers**, attach
  soil texture and C/N, and write **one consolidated long-form feature table**
  per site.
- **Stage 2 (`elm/predict.py`):** load the 8 bootstrap bundles named in
  `model_artifacts/cumulative70_bootstrap500/manifest.json`, select each
  outcome's feature subset directly from the master table, and write a
  per-site **NetCDF** file.

---

## 4. Soil layers — use all available layers (not a fixed 5)

The CLM workflow hardcoded the top `N_LAYERS = 5`. The ELM engine instead uses
**all available soil layers**, because the EFC predictands live in the BGC
decomposition column whose depth resolution depends on `use_vertsoilc`.

Evidence from E3SM source
(`components/elm/src/main/elm_varpar.F90`, lines 216–220):

```fortran
if (use_vertsoilc) then
   nlevdecomp      = nlevsoi      ! vertically resolved decomposition == soil layers
   nlevdecomp_full = nlevgrnd
else
   nlevdecomp      = 1            ! single bulk layer
   nlevdecomp_full = 1
end if
```

and (lines 196–200): `nlevgrnd = 15`, `nlevsoi = nlevsoifl` (standard) or
`8 + nlev_equalspace`.

Consequences for the engine:
- With **`use_vertsoilc = .true.`** (the default assumption): the decomposition
  grid equals `nlevsoi`, so we engineer features for **all `levsoi` layers**.
  `H2OSOI` is on `levsoi`, `TSOI` is on `levgrnd`; the first `nlevsoi` `levgrnd`
  nodes share depths with `levsoi` and are matched accordingly. Node depths are
  **read from the data**, never assumed.
- `use_vertsoilc` is a **root-caller CLI argument, default `true`**.
- The engine performs a **loud runtime check**: if a loaded `.pkl` turns out to
  be single-layer (`nlevdecomp = 1`, i.e. `use_vertsoilc=.false.` at run time),
  it raises rather than silently mis-handling the shape. (Note: the example
  `ABBY_ctrl.cfg` shows `use_vertsoilc = .false.`; the actual EFC production run
  flag must be confirmed from the real `.cfg`/`.pkl`.)

---

## 5. Gating item before coding starts

**NPP must be present in the ELM postprocessing output.** `NPP_daily` is a
required predictor (used by Carbohydrate, MonomerN, Organic P). Whether `NPP`
appears in the `.pkl .output` dict depends on the `[postprocessing] variables`
list used for the run. If it is missing, the ELM run will be re-done to include
`NPP` before Stage-1 development begins.

Other items to confirm at build time (first task of Stage 1):
1. `.output['taxis']` calendar/units and how to map it to daily `Date`.
2. Units of `H2OSOI` (m3/m3), `TSOI` (K), `NPP` (gC/m^2/s) in the pkl.
3. Actual soil-layer count and node depths; confirm `use_vertsoilc` state.

---

## 6. Data sources (ELM side)

| Input | Source | Notes |
|-------|--------|-------|
| `H2OSOI`, `TSOI`, `NPP`, time axis `taxis` | **`elm-olmt/pklfiles/{SITE}_{case}_I20TRCNPRDCTCBC.pkl`** | Postprocessed `model_ELM.main.ELMcase` object; variables in `.output` (dict, confirmed to hold `H2OSOI`/`TSOI`). Unpickling requires `elm-olmt` on `sys.path`. |
| Site, compset, transient years, `use_vertsoilc`, surffile path | **`elm-olmt/config_files/**/{SITE}_*.cfg`** | `[simulation] sites`, `[run_lengths] trans_startyear / nyears_trans`, `[case_options] surffile / use_vertsoilc`. |
| Soil texture `PCT_SAND`, `PCT_CLAY` | **surfdata NetCDF** referenced by `[case_options] surffile` in the `.cfg` | Read directly from the surface-data file (decision 2026-10-09). |
| Soil C / N / (C:N) | **`/pscratch/sd/t/tianyihu/CTSM_Hu/neon_surffiles/{SITE}_surfaceData.csv`** | NEON schema: `carbonTot`, `nitrogenTot`, `biogeoTopDepth/BottomDepth`, `sandTotal/clayTotal`, `bulkDensExclCoarseFrag`. C% = carbonTot/10, N% = nitrogenTot/10. |

- **Sites:** NEON — ABBY, JERC, RMNP, TALL, YELL, TEAK, WREF, SOAP, OSBS, …
- **Compset:** transient / historical only — **`I20TRCNPRDCTCBC`** (`I20` family).

---

## 7. Outputs (simplified — not byte-compatible with CLM)

Because the engine is rebuilt, outputs are redesigned for clarity:

### Stage-1 feature table (per site) — one consolidated table
- Long form: `site, Date, soil_layer, soil_node_depth_cm, <engineered features...>`.
- Written as **Parquet** (compact; large all-layer × multi-decade daily volume),
  with an optional CSV copy for inspection.
- Replaces the old "master + 8 per-outcome CSVs" layout. The predictor selects
  each outcome's feature subset in-memory from the manifest.

Engineered feature families (per layer, trailing windows W=7 / M=30 / Y=365 days):
`SM`, `ST`, `NPP_daily`, `SM_A_*_esm`, `SM_tau_*_esm`, `SM_theta_res_*_esm`,
`A_*_HT/LT/T`, `CV_*_T`, plus static `sand`, `clay`, `soil c (%)`, `soil n (%)`, `c/n`.
Drydown + temperature metrics are **ported into `elm/features.py`** (originally
from `clm_ref/efc_workflow.py`), so the engine has no dependency on legacy code.

### Stage-2 predictions (per site) — NetCDF
- One `.nc` per site; dims `(time, soil_layer)`; coord `soil_node_depth_cm(soil_layer)`.
- Data variables per outcome × summary stat, e.g. `carbohydrate_mean`,
  `carbohydrate_median`, `carbohydrate_sd`, `carbohydrate_q025`,
  `carbohydrate_q975`, for the 8 outcomes in the manifest.
- Default: 5 summary stats only. Optionally retain the full 500-member draws
  under a `bootstrap` dimension (flag).
- Self-describing: units, compset, provenance, bootstrap count as attributes;
  round-trips directly into `xarray`.

### Provenance
- One consolidated JSON per site (pkl/cfg/surffile paths, compset, transient
  years, layer count + depths, date coverage, verified units, NPP present y/n).

---

## 8. Layout

```
efc-model-refactor/
  run_efc.py                   # NEW root caller (CLI) — to build
  elm/                         # NEW engine package — to build
    io.py  config.py  surface.py  features.py  build.py  predict.py
  predict_with_saved_cumulative70_models.py   # existing Stage-2 (to be wrapped)
  model_artifacts/             # 8 bundles (git-ignored) + manifest.json (tracked)
  clm_ref/                     # frozen CLM reference (do not edit)
    build_rmnp_layer_model_inputs.py
    build_multisite_layer_model_inputs.py
    efc_workflow.py            # source of _drydown_metrics (ported into elm/features.py)
  outputs/                     # generated feature tables + NetCDF predictions (git-ignored)
```

---

## 9. Environment

Use the OLMT Perlmutter conda environment (xarray, scikit-learn, joblib,
pandas, numpy, scipy; also needed to unpickle ELM `.pkl` objects):

```bash
/global/common/software/m4803/conda/envs/OLMT_pm_TianyiPY311/bin/python
```

> The old CLM README referenced `/opt/anaconda3/envs/neon-efc/bin/python`, which
> does not exist on this machine. Use the path above.

---

## 10. Version control notes

- The ~2 GB of `*.joblib` bundles, generated outputs (`outputs/`, legacy
  `model_inputs/` / `model_predictions/`), and ELM `*.pkl` files are
  **git-ignored**; only source, the small `manifest.json`, and docs are tracked.
- The ELM engine is developed fresh in `elm/` + `run_efc.py`; the CLM scripts in
  `clm_ref/` are read-only reference with intentionally broken embedded paths.
