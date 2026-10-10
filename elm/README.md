# ELM engine package (placeholder — not yet scaffolded)

This package will hold the from-scratch ELM EFC engine, driven by the root
caller `../run_efc.py`. Planned modules:

- `io.py` — load the `model_ELM.main.ELMcase` pickle; extract `.output`,
  `taxis`, and provenance.
- `config.py` — parse the elm-olmt `.cfg` (sites, surffile path, transient
  years, `use_vertsoilc`).
- `surface.py` — read `PCT_SAND` / `PCT_CLAY` from the surffile NetCDF and
  soil C/N from the NEON surface CSV.
- `features.py` — trailing 7/30/365-day drydown + temperature metrics for all
  soil layers (ports `_drydown_metrics` from `../clm_ref/efc_workflow.py`).
- `build.py` — orchestrate Stage 1 into one consolidated per-site feature table.
- `predict.py` — apply the 8 bootstrap bundles and write per-site NetCDF.

No code is written yet. See the repository README for the full plan and the NPP
gating item that must be resolved before development begins.
