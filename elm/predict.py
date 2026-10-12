"""Stage 2 — apply the saved bootstrap bundles and write per-case NetCDF.

Reuses the fixed contract from ``model_artifacts/.../manifest.json`` and the
existing bundle format (``{"features": [...], "models": [RF, ...]}``). For each
outcome and soil layer, all bootstrap members predict; summary statistics are
written to a NetCDF with dims ``(time, soil_layer)``.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xarray as xr
from joblib import Parallel, delayed

SUMMARY_STATS = ("mean", "median", "sd", "q025", "q975")


def _slug(outcome: str) -> str:
    return outcome.lower().replace(" ", "_")


def load_manifest(model_dir: Path) -> dict:
    return json.loads((model_dir / "manifest.json").read_text())


def predict_case(
    master: pd.DataFrame,
    model_dir: Path,
    out_dir: Path,
    case_name: str,
    site: str,
    compset: str,
    keep_bootstraps: bool = False,
    n_jobs: int = 1,
) -> str:
    """Run all 8 bundles over the feature table; write a per-case NetCDF.

    ``n_jobs`` sets the number of threads each RandomForest uses for
    ``.predict`` (overridden at runtime on the loaded estimators). This does
    not change predictions, only parallelizes tree evaluation across cores.
    """
    model_dir = Path(model_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(model_dir)

    layers = np.sort(master["soil_layer"].unique())
    # Common (time, layer) grid from layer 1's date axis (all layers share dates).
    base = (
        master.loc[master.soil_layer.eq(layers[0])]
        .sort_values("Date").reset_index(drop=True)
    )
    times = pd.to_datetime(base["Date"].to_numpy())
    n_time, n_layer = len(times), len(layers)

    depth_by_layer = (
        master.drop_duplicates("soil_layer")
        .set_index("soil_layer")["soil_node_depth_cm"]
        .reindex(layers).to_numpy(dtype=float)
    )

    data_vars: dict[str, tuple] = {}
    bootstrap_members = None

    for outcome in manifest["outcomes"]:
        prefix = _slug(outcome)
        bundle = joblib.load(model_dir / f"cumulative70_{prefix}_bootstrap500.joblib")
        feats = bundle["features"]
        models = bundle["models"]
        n_boot = len(models)
        # Each estimator predicts single-threaded; parallelism is applied ACROSS
        # the bootstrap members below (the right axis). Overriding each model's
        # own n_jobs instead *slows* these many small predicts (verified), which
        # is what caused the earlier timeout.
        for m in models:
            try:
                m.n_jobs = 1
            except AttributeError:
                pass

        stat_arrays = {s: np.full((n_time, n_layer), np.nan) for s in SUMMARY_STATS}
        boot_array = (
            np.full((n_time, n_layer, n_boot), np.nan) if keep_bootstraps else None
        )

        for li, layer in enumerate(layers):
            lt = (
                master.loc[master.soil_layer.eq(layer)]
                .sort_values("Date").reset_index(drop=True)
            )
            if len(lt) != n_time:
                raise ValueError(
                    f"layer {layer} has {len(lt)} rows; expected {n_time}"
                )
            missing = sorted(set(feats) - set(lt.columns))
            if missing:
                raise ValueError(f"{outcome}: feature table missing {missing}")
            X = lt[feats]
            if n_jobs and n_jobs > 1:
                # Parallelize across the 500 bootstrap members (threads: RF
                # predict releases the GIL; no data copy, numerically identical).
                cols = Parallel(n_jobs=n_jobs, prefer="threads")(
                    delayed(m.predict)(X) for m in models
                )
                preds = np.column_stack(cols)  # (n_time, n_boot)
            else:
                preds = np.column_stack([m.predict(X) for m in models])
            stat_arrays["mean"][:, li] = preds.mean(axis=1)
            stat_arrays["median"][:, li] = np.median(preds, axis=1)
            stat_arrays["sd"][:, li] = preds.std(axis=1, ddof=1)
            stat_arrays["q025"][:, li] = np.quantile(preds, 0.025, axis=1)
            stat_arrays["q975"][:, li] = np.quantile(preds, 0.975, axis=1)
            if keep_bootstraps:
                boot_array[:, li, :] = preds

        for s in SUMMARY_STATS:
            data_vars[f"{prefix}_{s}"] = (("time", "soil_layer"), stat_arrays[s])
        if keep_bootstraps:
            data_vars[f"{prefix}_bootstrap"] = (
                ("time", "soil_layer", "bootstrap"), boot_array
            )
            bootstrap_members = n_boot
        del bundle, models

    coords = {
        "time": ("time", times),
        "soil_layer": ("soil_layer", layers.astype(int)),
        "soil_node_depth_cm": ("soil_layer", depth_by_layer),
    }
    ds = xr.Dataset(data_vars=data_vars, coords=coords)
    ds.attrs.update({
        "title": "EFC abundance predictions (cumulative70 bootstrap ensemble)",
        "site": site,
        "case_name": case_name,
        "compset": compset,
        "bootstrap_models_per_outcome": int(manifest["bootstrap_models_per_outcome"]),
        "outcomes": ", ".join(manifest["outcomes"]),
        "summary_stats": ", ".join(SUMMARY_STATS),
    })
    ds["soil_node_depth_cm"].attrs["units"] = "cm"
    if keep_bootstraps and bootstrap_members:
        ds = ds.assign_coords(bootstrap=np.arange(1, bootstrap_members + 1))

    out_path = out_dir / f"{case_name}_EFC_predictions.nc"
    ds.to_netcdf(out_path)
    return str(out_path)
