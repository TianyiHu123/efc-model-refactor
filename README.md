# EFC Abundance Model — ELM Interface

Predict Extracellular Enzyme / Functional Capacity (EFC) abundance from land-model
output using a pre-trained Random-Forest bootstrap ensemble.

This repository is being refactored to drive the EFC model from **ELM** output
produced by [`elm-olmt`](/pscratch/sd/t/tianyihu/elm-olmt), replacing an earlier
CLM-based workflow. The CLM code is retained **for reference only** and will not
be modified.

---

## 1. Status

| Component | State |
|-----------|-------|
| Stage 2 — prediction (`predict_with_saved_cumulative70_models.py`) | **Ready, reused unchanged** |
| Model artifacts (`model_artifacts/cumulative70_bootstrap500/`) | **Ready** (8 bundles + manifest) |
| Feature engineering (`efc_workflow._drydown_metrics` + temperature metrics) | **Ready, reused as the scientific contract** |
| CLM Stage-1 builders (`build_*_layer_model_inputs.py`) | **Reference only — do not edit** |
| ELM Stage-1 builder (`elm/`) | **Not started** — blocked on NPP (see §5) |
| README + git tracking | **This commit** |

---

## 2. Pipeline overview

```
ELM run (elm-olmt)                     this repo
──────────────────                     ─────────
.cfg config  ─┐
              ├─▶  elm/ Stage-1  ─▶  per-site, per-outcome        ─▶  Stage-2 predictor  ─▶  per-layer
pkl output  ──┤    (to build)        model-input CSVs                 (reused as-is)          EFC predictions
NEON csv  ───┤                       (model_inputs/)                                          (model_predictions/)
surfdata nc ─┘
```

- **Stage 1 (to build, `elm/`):** read ELM output + site metadata, engineer the
  trailing 7/30/365-day hydrothermal features for the top 5 soil layers, attach
  soil texture and C/N, and write the per-outcome model-input CSVs in the exact
  column order the trained models expect.
- **Stage 2 (reused unchanged):** `predict_with_saved_cumulative70_models.py`
  loads the 8 bootstrap bundles and writes per-layer predictions with
  mean / median / sd / 2.5% / 97.5% summaries.

The ELM Stage-1 output **must conform to the existing Stage-2 input contract** so
the trained models and predictor are reused without change.

---

## 3. Data sources (ELM side)

| Input | Source | Notes |
|-------|--------|-------|
| Soil moisture `H2OSOI`, soil temperature `TSOI`, `NPP`, time axis `taxis` | **`elm-olmt/pklfiles/{SITE}_{case}_I20TRCNPRDCTCBC.pkl`** | Postprocessed `model_ELM.main.ELMcase` object; variables live in `.output` (a dict). Unpickling requires `elm-olmt` on `sys.path`. |
| Experiment metadata (site, compset, transient years, surffile path) | **`elm-olmt/config_files/**/{SITE}_*.cfg`** | `[simulation] sites`, `[run_lengths] trans_startyear / nyears_trans`, `[case_options] surffile`, `[postprocessing]`. |
| Soil texture `PCT_SAND`, `PCT_CLAY` | **surfdata NetCDF** referenced by `[case_options] surffile` in the `.cfg` | Read directly from the surface-data file (per decision 2026-10-09), not from the pkl's embedded `siteinfo`. |
| Soil C / N / (C:N) | **`/pscratch/sd/t/tianyihu/CTSM_Hu/neon_surffiles/{SITE}_surfaceData.csv`** | Same NEON schema the CLM reader used: `carbonTot`, `nitrogenTot`, `biogeoTopDepth/BottomDepth`, `bulkDensExclCoarseFrag`. C% = carbonTot/10, N% = nitrogenTot/10. |

### Targets
- **Sites:** NEON sites — ABBY, JERC, RMNP, TALL, YELL, TEAK, WREF, SOAP, OSBS, …
- **Compset:** transient / historical runs only — **`I20TRCNPRDCTCBC`** (the `I20` family).

### Key difference from CLM
- **`levsoi` / `levgrnd` node depths differ** between the CLM and ELM
  configurations. Stage 1 must read the ELM depths from the data rather than
  assuming the CLM depths. The top-5-layer convention (`N_LAYERS = 5`) is kept.
- Variable **names** are expected to match CLM (`H2OSOI`, `TSOI`, `NPP`,
  `PCT_SAND`, `PCT_CLAY`); units to be verified against the pkl at build time.

---

## 4. What Stage 1 must produce (contract with Stage 2)

For each site directory `model_inputs/{SITE}_cumulative70/`:

- One five-layer master feature table.
- Eight per-outcome model-input CSVs (predictors in trained order) for the
  outcomes in `model_artifacts/cumulative70_bootstrap500/manifest.json`:
  Carbohydrate, Lignin, Organic N, MonomerN, Inorganic N, Organic P,
  Monomer P, Inorganic P.
- Identifier columns required by the predictor:
  `site, Date, soil_layer, soil_node_depth_cm`.
- A JSON provenance/coverage audit (recommended).

Engineered feature families (per layer, trailing windows W=7 / M=30 / Y=365 days):
`SM`, `ST`, `NPP_daily`, `SM_A_*_esm`, `SM_tau_*_esm`, `SM_theta_res_*_esm`,
`A_*_HT/LT/T`, `CV_*_T`, plus static `sand`, `clay`, `soil c (%)`, `soil n (%)`, `c/n`.
The drydown metrics come from `efc_workflow._drydown_metrics`; temperature
metrics replicate the CLM temperature-metric logic.

---

## 5. Gating item before coding starts

**NPP must be present in the ELM postprocessing output.** `NPP_daily` is a
required predictor (used by Carbohydrate, MonomerN, Organic P). Whether `NPP`
appears in `.pkl .output` depends on the `[postprocessing] variables` list used
for the run. If it is missing, the ELM run will be re-done to include `NPP`
before Stage-1 development begins.

Other items to confirm at build time (first task of Stage 1):
1. `.output['taxis']` calendar/units and how to map it to daily `Date`.
2. Units of `H2OSOI` (m3/m3), `TSOI` (K), `NPP` (gC/m^2/s) in the pkl.
3. ELM `levsoi` / `levgrnd` depths and alignment of the first five nodes.

---

## 6. Planned layout

```
efc-model-refactor/
  elm/                         # NEW ELM Stage-1 (to build)
    build_elm_model_inputs.py  #   .pkl + .cfg + NEON csv + surfdata -> model-input CSVs
    elm_io.py                  #   load ELMcase pkl; extract output/taxis
    config.py                  #   parse elm-olmt .cfg (sites, surffile, trans years)
    features.py                #   trailing 7/30/365 metrics (drydown + temperature)
  predict_with_saved_cumulative70_models.py   # REUSED unchanged (Stage 2)
  model_artifacts/             # 8 bootstrap bundles (git-ignored) + manifest.json (tracked)
  efc_workflow.py              # reference: _drydown_metrics
  build_rmnp_layer_model_inputs.py          # CLM reference, untouched
  build_multisite_layer_model_inputs.py     # CLM reference, untouched
  model_inputs/ , model_predictions/        # generated outputs (git-ignored)
```

---

## 7. Environment

Use the OLMT Perlmutter conda environment (has xarray, scikit-learn, joblib,
pandas, numpy, scipy; also needed to unpickle ELM `.pkl` objects):

```bash
/global/common/software/m4803/conda/envs/OLMT_pm_TianyiPY311/bin/python
```

> Note: the old CLM README referenced `/opt/anaconda3/envs/neon-efc/bin/python`,
> which does not exist on this machine. Use the path above.

---

## 8. Version control notes

- The ~2 GB of `*.joblib` bundles and all generated outputs (`model_inputs/`,
  `model_predictions/`) and ELM `*.pkl` files are **git-ignored**; only source,
  the small `manifest.json`, and docs are tracked.
- The ELM interface is being developed fresh in `elm/`; the CLM scripts are kept
  as read-only reference.
