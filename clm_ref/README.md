# CLM reference code (frozen — do not edit)

These scripts are the original CLM-based EFC input/prediction workflow, copied
from another machine. They are kept **only as reference** while the ELM engine
is rebuilt in `../elm/`.

- `build_rmnp_layer_model_inputs.py` — single-site (RMNP) CLM Stage-1 builder;
  defines `read_daily`, `layer_features`, `N_LAYERS = 5`.
- `build_multisite_layer_model_inputs.py` — multi-site CLM Stage-1 builder.
- `efc_workflow.py` — source of `_drydown_metrics` (soil-moisture drydown
  feature engineering). The ELM engine ports this logic into
  `../elm/features.py`; this copy is retained for provenance.

Notes:

- Paths and relative imports embedded in these files are **intentionally left
  broken** after the move into `clm_ref/`. They are not meant to run from here.
- The scientific feature definitions (drydown metrics, trailing 7/30/365-day
  temperature metrics) are the contract the trained models expect and are
  reproduced faithfully by the ELM engine.
