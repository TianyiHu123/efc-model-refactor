"""Build five-layer RMNP inputs for the Submission3 cumulative-70% EFC models.

The workflow reads only CLM history (clm2.h0) NetCDF files, computes daily
means, and engineers the same trailing 7/30/365-observation hydrothermal
features used to train the updated Submission3 models. Each soil layer is
processed independently; NPP is a site-level daily flux and is repeated for
each layer.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import variation

from efc_workflow import _drydown_metrics


ROOT = Path(__file__).resolve().parent
DEFAULT_CLM = ROOT / "EFC inputs from CLM" / "RMNP"
DEFAULT_SURFDATA = (
    ROOT / "EFC inputs from CLM"
    / "surfdata_1x1pt_RMNP_hist_78pfts_CMIP6_simyr2000_c260422_updated.nc"
)
DEFAULT_SURFACE_CSV = ROOT / "EFC inputs from CLM" / "site_surface_Files" / "RMNP_surfaceData.csv"
DEFAULT_FEATURES = (
    ROOT / "Submission3_ISME_Verson" / "analysis_outputs"
    / "importance_robustness" / "results"
    / "selected15_cumulative70_features.json"
)
DEFAULT_OUTPUT = ROOT / "model_inputs" / "RMNP_cumulative70"
N_LAYERS = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clm-folder", type=Path, default=DEFAULT_CLM)
    parser.add_argument("--surfdata", type=Path, default=DEFAULT_SURFDATA)
    parser.add_argument("--surface-csv", type=Path, default=DEFAULT_SURFACE_CSV)
    parser.add_argument("--feature-json", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def depth_resolved_carbon_nitrogen(surface_csv: Path, depths_cm: np.ndarray) -> pd.DataFrame:
    """Map CLM node centers to NEON horizons and convert g/kg to mass percent."""
    columns = [
        "siteID", "horizonName", "biogeoTopDepth", "biogeoBottomDepth",
        "carbonTot", "nitrogenTot",
    ]
    surface = pd.read_csv(surface_csv, usecols=columns)
    surface = surface.loc[surface.siteID.eq("RMNP")].copy()
    for column in ["biogeoTopDepth", "biogeoBottomDepth", "carbonTot", "nitrogenTot"]:
        surface[column] = pd.to_numeric(surface[column], errors="coerce")
    surface = surface.dropna(subset=columns[2:])
    # Duplicate depth intervals are replicate observations. Average them before
    # mapping so every CLM node receives one reproducible horizon value.
    horizons = (
        surface.groupby(["horizonName", "biogeoTopDepth", "biogeoBottomDepth"], as_index=False)
        .agg(carbonTot=("carbonTot", "mean"), nitrogenTot=("nitrogenTot", "mean"))
        .sort_values(["biogeoTopDepth", "biogeoBottomDepth"])
    )
    records = []
    for layer, depth in enumerate(depths_cm, start=1):
        match = horizons.loc[
            horizons.biogeoTopDepth.le(depth) & horizons.biogeoBottomDepth.gt(depth)
        ]
        if len(match) != 1:
            raise ValueError(f"Depth {depth} cm matched {len(match)} RMNP horizons")
        horizon = match.iloc[0]
        carbon_g_kg = float(horizon.carbonTot)
        nitrogen_g_kg = float(horizon.nitrogenTot)
        if carbon_g_kg <= 0 or nitrogen_g_kg <= 0:
            raise ValueError(f"Nonpositive C or N for layer {layer}")
        records.append({
            "soil_layer": layer,
            "biogeo_horizon": horizon.horizonName,
            "biogeo_top_depth_cm": float(horizon.biogeoTopDepth),
            "biogeo_bottom_depth_cm": float(horizon.biogeoBottomDepth),
            "soil c (%)": carbon_g_kg / 10.0,
            "soil n (%)": nitrogen_g_kg / 10.0,
            "c/n": carbon_g_kg / nitrogen_g_kg,
        })
    return pd.DataFrame(records)


def read_soil_texture(surfdata_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return the first five RMNP sand and clay layers from CLM surfdata."""
    with xr.open_dataset(surfdata_path, decode_times=False) as surfdata:
        required = {"PCT_SAND", "PCT_CLAY"}
        missing = required - set(surfdata.data_vars)
        if missing:
            raise KeyError(f"Surfdata is missing variables: {sorted(missing)}")
        sand = np.asarray(surfdata.PCT_SAND.values, dtype=float).squeeze()[:N_LAYERS]
        clay = np.asarray(surfdata.PCT_CLAY.values, dtype=float).squeeze()[:N_LAYERS]
    if sand.shape != (N_LAYERS,) or clay.shape != (N_LAYERS,):
        raise ValueError("Surfdata did not provide five sand/clay layers")
    if not np.all((sand >= 0) & (sand <= 100)) or not np.all((clay >= 0) & (clay <= 100)):
        raise ValueError("Surfdata sand/clay percentages are outside 0-100")

    return sand, clay


def timestamp_index(values) -> pd.DatetimeIndex:
    result = []
    for value in values:
        if hasattr(value, "year"):
            result.append(pd.Timestamp(int(value.year), int(value.month), int(value.day)))
        else:
            result.append(pd.Timestamp(value).normalize())
    return pd.DatetimeIndex(result)


def read_daily(files: list[Path]) -> tuple[pd.DatetimeIndex, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with xr.open_mfdataset(
        files, combine="by_coords", data_vars="all", coords="different",
        compat="no_conflicts", join="outer"
    ) as source:
        required = {"H2OSOI", "TSOI", "NPP"}
        missing = required - set(source.data_vars)
        if missing:
            raise KeyError(f"Missing NetCDF variables: {sorted(missing)}")
        expected_units = {"H2OSOI": {"mm3/mm3", "m3/m3"}, "TSOI": {"K"}, "NPP": {"gC/m^2/s"}}
        for variable, allowed in expected_units.items():
            actual = source[variable].attrs.get("units")
            if actual not in allowed:
                raise ValueError(f"Unexpected {variable} units: {actual!r}; expected {sorted(allowed)}")
        if source.sizes.get("levsoi", 0) < N_LAYERS or source.sizes.get("levgrnd", 0) < N_LAYERS:
            raise ValueError("NetCDF does not contain five soil/ground layers")

        days = timestamp_index(source.time.values)
        selected = source[["H2OSOI", "TSOI", "NPP"]].assign_coords(day=("time", days))
        daily = selected.groupby("day").mean("time").sortby("day").load()
        dates = pd.DatetimeIndex(daily.day.values)
        sm = np.asarray(daily.H2OSOI.isel(levsoi=slice(0, N_LAYERS), lndgrid=0).values, dtype=float)
        st = np.asarray(daily.TSOI.isel(levgrnd=slice(0, N_LAYERS), lndgrid=0).values, dtype=float)
        npp = np.asarray(daily.NPP.isel(lndgrid=0).values, dtype=float)
        sm_depth = np.asarray(source.levsoi.values[:N_LAYERS], dtype=float) * 100.0
        st_depth = np.asarray(source.levgrnd.values[:N_LAYERS], dtype=float) * 100.0

    if dates.has_duplicates or not dates.is_monotonic_increasing:
        raise ValueError("Daily dates must be unique and increasing")
    if sm.shape != st.shape or sm.shape[0] != len(dates) or len(npp) != len(dates):
        raise ValueError("Unexpected daily array shape after layer selection")
    if not np.allclose(sm_depth, st_depth, atol=1e-5):
        raise ValueError("First-five H2OSOI and TSOI node depths do not align")
    return dates, sm, st, npp, sm_depth


def temperature_metrics(values_kelvin: np.ndarray, rolling_days: int) -> dict[str, float]:
    values_c = pd.Series(np.asarray(values_kelvin, dtype=float) - 273.15)
    rolling = values_c.rolling(rolling_days, center=True, min_periods=rolling_days).mean()
    return {
        "HT": float(rolling.max()),
        "LT": float(rolling.min()),
        "T": float(values_c.mean()),
        "CV": float(variation(values_c, nan_policy="omit")),
    }


def layer_features(dates: pd.DatetimeIndex, sm: np.ndarray, st: np.ndarray, npp: np.ndarray,
                   layer: int, depth_cm: float) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for current in range(365, len(dates)):
        row: dict[str, object] = {
            "site": "RMNP",
            "Date": dates[current],
            "soil_layer": layer + 1,
            "soil_node_depth_cm": depth_cm,
            "SM": float(sm[current, layer]),
            "ST": float(st[current, layer] - 273.15),
            "NPP_daily": float(npp[current]),
            "history_start": dates[current - 365],
            "history_end": dates[current - 1],
        }
        for label, count, rolling_days in (("W", 7, 1), ("M", 30, 7), ("Y", 365, 7)):
            moisture = _drydown_metrics(sm[current - count:current, layer])
            temp = temperature_metrics(st[current - count:current, layer], rolling_days)
            row.update({
                f"SM_A_{label}_esm": moisture[0],
                f"SM_tau_{label}_esm": moisture[1],
                f"SM_theta_res_{label}_esm": moisture[2],
                f"A_{label}_HT": temp["HT"],
                f"A_{label}_LT": temp["LT"],
                f"A_{label}_T": temp["T"],
                f"CV_{label}_T": temp["CV"],
            })
        row["hottest_week"] = row["A_Y_HT"]
        records.append(row)
    return pd.DataFrame(records)


def main() -> None:
    args = parse_args()
    files = sorted(args.clm_folder.glob("*.clm2.h0.*.nc"))
    if not files:
        raise FileNotFoundError(f"No CLM h0 files in {args.clm_folder}")
    dates, sm, st, npp, depths = read_daily(files)

    sand, clay = read_soil_texture(args.surfdata)
    chemistry = depth_resolved_carbon_nitrogen(args.surface_csv, depths)

    layers = []
    for layer, depth in enumerate(depths):
        frame = layer_features(dates, sm, st, npp, layer, float(depth))
        frame["sand"] = sand[layer]
        frame["clay"] = clay[layer]
        layer_chemistry = chemistry.loc[chemistry.soil_layer.eq(layer + 1)].iloc[0]
        for column in [
            "biogeo_horizon", "biogeo_top_depth_cm", "biogeo_bottom_depth_cm",
            "soil c (%)", "soil n (%)", "c/n",
        ]:
            frame[column] = layer_chemistry[column]
        layers.append(frame)
    master = pd.concat(layers, ignore_index=True).sort_values(["Date", "soil_layer"]).reset_index(drop=True)

    with args.feature_json.open() as handle:
        selected: dict[str, list[str]] = json.load(handle)
    all_features = sorted({feature for features in selected.values() for feature in features})
    missing = sorted(set(all_features) - set(master.columns))
    if missing:
        raise ValueError(f"Engineered table is missing model features: {missing}")
    if master[all_features].isna().any().any():
        raise ValueError("Model feature table contains missing values")
    # CLM occasionally produces a tiny oversaturation above 1.0 at the top
    # node (observed maximum 1.0093). Preserve model output rather than clip it.
    if not master.SM.between(0, 1.05).all() or not master.ST.between(-80, 70).all():
        raise ValueError("Hydrothermal values failed physical-range checks")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    master.to_csv(args.output_dir / "RMNP_five_layer_all_features.csv", index=False, date_format="%Y-%m-%d")
    identifier_columns = ["site", "Date", "soil_layer", "soil_node_depth_cm"]
    for outcome, features in selected.items():
        safe_name = outcome.lower().replace(" ", "_")
        master[identifier_columns + features].to_csv(
            args.output_dir / f"RMNP_{safe_name}_cumulative70_model_input.csv",
            index=False,
            date_format="%Y-%m-%d",
        )

    audit = {
        "site": "RMNP",
        "source_folder": str(args.clm_folder),
        "source_file_count": len(files),
        "source_first_date": str(dates.min().date()),
        "source_last_date": str(dates.max().date()),
        "prediction_first_date": str(master.Date.min().date()),
        "prediction_last_date": str(master.Date.max().date()),
        "daily_source_rows": len(dates),
        "output_rows": len(master),
        "rows_per_layer": int(len(master) / N_LAYERS),
        "soil_node_depths_cm": depths.tolist(),
        "soil_moisture_min": float(master.SM.min()),
        "soil_moisture_max": float(master.SM.max()),
        "soil_temperature_c_min": float(master.ST.min()),
        "soil_temperature_c_max": float(master.ST.max()),
        "soil_texture_source": str(args.surfdata),
        "sand_source_variable": "PCT_SAND",
        "clay_source_variable": "PCT_CLAY",
        "sand_percent_by_layer": sand.tolist(),
        "clay_percent_by_layer": clay.tolist(),
        "soil_c_n_and_cn_source": str(args.surface_csv),
        "soil_c_source_column": "carbonTot",
        "soil_n_source_column": "nitrogenTot",
        "soil_c_n_source_units": "g/kg; divided by 10 to obtain mass percent",
        "soil_cn_definition": "carbonTot / nitrogenTot (dimensionless)",
        "soil_chemistry_by_layer": chemistry.to_dict(orient="records"),
        "npp_definition": "daily mean NPP flux rate; original units gC/m^2/s",
        "history_definition": "prior 7/30/365 observed no-leap-calendar days; current day excluded",
        "outcome_features": selected,
    }
    (args.output_dir / "RMNP_model_input_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
