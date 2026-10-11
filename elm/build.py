"""Stage 1 — build one consolidated long-form feature table per case.

Combines daily hydrothermal arrays (``elm.io``), static soil properties
(``elm.surface``), and trailing features (``elm.features``) into a single
long-form table (one row per Date x soil_layer), plus a provenance dict.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from . import features as _features
from .config import CaseConfig
from .io import DailyCase
from .surface import SoilProperties

IDENTIFIER_COLUMNS = ["site", "Date", "soil_layer", "soil_node_depth_cm"]


def build_feature_table(
    cfg: CaseConfig,
    daily: DailyCase,
    soil: SoilProperties,
) -> pd.DataFrame:
    """Return the consolidated long-form feature table (all layers)."""
    frames = []
    for layer in range(daily.nlayer):
        frame = _features.layer_features(
            daily.dates, daily.sm, daily.st, daily.npp,
            layer=layer, depth_cm=float(daily.depths_cm[layer]), site=cfg.site,
        )
        frame["sand"] = soil.sand[layer]
        frame["clay"] = soil.clay[layer]
        frame["soil c (%)"] = soil.soil_c_percent[layer]
        frame["soil n (%)"] = soil.soil_n_percent[layer]
        frame["c/n"] = soil.c_n[layer]
        frames.append(frame)
    master = (
        pd.concat(frames, ignore_index=True)
        .sort_values(["Date", "soil_layer"])
        .reset_index(drop=True)
    )
    return master


def validate_model_features(master: pd.DataFrame, model_features: list[str]) -> None:
    """Raise if any required model feature is missing or contains NaN."""
    missing = sorted(set(model_features) - set(master.columns))
    if missing:
        raise ValueError(f"Feature table missing model features: {missing}")
    nan_cols = [f for f in model_features if master[f].isna().any()]
    if nan_cols:
        raise ValueError(f"Feature table has NaN in model features: {nan_cols}")
    if master.duplicated(["Date", "soil_layer"]).any():
        raise ValueError("Feature table has duplicate (Date, soil_layer) rows")


def write_feature_table(
    master: pd.DataFrame, out_dir: Path, case_name: str, write_csv: bool = True
) -> dict[str, str]:
    """Write Parquet (always) and optional CSV; return written paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    parquet_path = out_dir / f"{case_name}_features.parquet"
    try:
        master.to_parquet(parquet_path, index=False)
        paths["parquet"] = str(parquet_path)
    except Exception as exc:  # pyarrow/fastparquet unavailable
        paths["parquet_error"] = f"{type(exc).__name__}: {exc}"
    if write_csv:
        csv_path = out_dir / f"{case_name}_features.csv"
        master.to_csv(csv_path, index=False, date_format="%Y-%m-%d")
        paths["csv"] = str(csv_path)
    if "parquet" not in paths and "csv" not in paths:
        raise RuntimeError(
            f"failed to write the feature table in any format: "
            f"{paths.get('parquet_error', 'parquet unavailable')}; CSV disabled."
        )
    return paths


def build_provenance(
    cfg: CaseConfig,
    daily: DailyCase,
    soil: SoilProperties,
    run_dir: Path,
    neon_csv: Path,
    compset: str,
    master: pd.DataFrame,
) -> dict:
    return {
        "site": cfg.site,
        "case_name": cfg.case_name,
        "compset": compset,
        "run_dir": str(run_dir),
        "config_surffile": str(cfg.surffile),
        "neon_csv": str(neon_csv),
        "use_vertsoilc": cfg.use_vertsoilc,
        "trans_startyear": cfg.trans_startyear,
        "nyears_trans": cfg.nyears_trans,
        "date_first": str(daily.dates.min().date()),
        "date_last": str(daily.dates.max().date()),
        "n_daily_source_days": int(len(daily.dates)),
        "n_soil_layers": int(daily.nlayer),
        "soil_node_depths_cm": [float(x) for x in daily.depths_cm],
        "texture_native_nlevsoi": soil.texture_layers_native,
        "texture_extended_to_levgrnd": soil.texture_extended,
        "cn_below_deepest_horizon_layers": soil.cn_below_horizon_layers,
        "units": {"H2OSOI": "mm3/mm3", "TSOI": "K->degC", "NPP_daily": "gC/m^2/s"},
        "feature_rows": int(len(master)),
        "feature_first_date": str(pd.to_datetime(master.Date).min().date()),
        "feature_last_date": str(pd.to_datetime(master.Date).max().date()),
        "history_warmup_days": _features._WARMUP,
    }


def write_provenance(provenance: dict, out_dir: Path, case_name: str) -> str:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{case_name}_provenance.json"
    path.write_text(json.dumps(provenance, indent=2) + "\n")
    return str(path)
