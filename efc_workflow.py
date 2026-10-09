"""Reproducible feature engineering, training, and prediction for lignin EFC."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import variation
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split


SUPPORTED_SITES = ("RMNP", "TALL", "JERC", "TEAK", "YELL")
LIGNIN_FEATURES = ("SM", "CV_Y_T", "SM_theta_res_Y_esm", "SM_A_Y_esm")
DEPTH_DIMS = ("levgrnd", "levsoi", "levlak", "levdcmp", "lndgrid")


def choose_sites(value: str | None = None) -> list[str]:
    """Validate a comma-separated interactive or supplied site selection."""
    if value is None:
        value = input(
            "NEON site(s), comma-separated "
            f"[{', '.join(SUPPORTED_SITES)}; Enter = all]: "
        ).strip()
    sites = list(SUPPORTED_SITES) if not value else [
        item.strip().upper() for item in value.split(",") if item.strip()
    ]
    unknown = sorted(set(sites) - set(SUPPORTED_SITES))
    if unknown:
        raise ValueError(f"Unsupported NEON site(s): {', '.join(unknown)}")
    return list(dict.fromkeys(sites))


def netcdf_files(data_root: Path, site: str) -> list[Path]:
    """Return the CLM h0 files in the project's current directory layout."""
    folder = data_root / site / "hist"
    files = sorted(path for path in folder.glob("*.nc") if "h0" in path.name)
    if not files:
        raise FileNotFoundError(f"No CLM h0 NetCDF files found in {folder}")
    return files


def select_surface_five_layer_mean(data: xr.DataArray) -> xr.DataArray:
    """Match the original notebook's unweighted first-five-layer mean."""
    for dim in data.dims:
        if dim not in DEPTH_DIMS:
            continue
        size = data.sizes[dim]
        if size == 1:
            data = data.isel({dim: 0})
        elif size < 5:
            data = data.mean(dim)
        else:
            data = data.isel({dim: slice(0, 5)}).mean(dim)
    return data.squeeze(drop=True)


def read_daily_site(data_root: Path, site: str) -> pd.DataFrame:
    """Read one site's CLM files and return one daily record per date."""
    files = netcdf_files(data_root, site)
    with xr.open_mfdataset(files, combine="by_coords") as dataset:
        required = {"H2OSOI", "TSOI", "NPP"}
        missing = required - set(dataset.data_vars)
        if missing:
            raise KeyError(f"{site} is missing NetCDF variables: {sorted(missing)}")

        dataset = dataset.assign_coords(
            time=dataset.indexes["time"].to_datetimeindex()
        )
        # Match the original notebook exactly. Grouping only observed model
        # dates preserves the CLM no-leap calendar and does not manufacture a
        # Gregorian February 29 with missing values.
        daily = (
            dataset[sorted(required)]
            .groupby("time.date")
            .mean("time")
            .rename({"date": "time"})
        )
        core = xr.Dataset(
            {
                "SM": select_surface_five_layer_mean(daily["H2OSOI"]),
                "ST": select_surface_five_layer_mean(daily["TSOI"]),
                # Daily-mean NPP flux rate, matching the existing CSV convention.
                "NPP_daily": select_surface_five_layer_mean(daily["NPP"]),
            }
        ).load()

    frame = core.to_dataframe().reset_index()
    unexpected = [c for c in frame.columns if c not in {"time", "SM", "ST", "NPP_daily"}]
    if unexpected or frame["time"].duplicated().any():
        raise ValueError(
            f"{site} did not reduce to one row per day; remaining dimensions: "
            f"{unexpected}"
        )
    return (
        frame.rename(columns={"time": "Date"})
        .sort_values("Date")
        .assign(site=site)
        .reset_index(drop=True)
    )


def _decreasing_sequences(values: np.ndarray, tolerance_ratio: float = 0.05):
    values = np.asarray(values, dtype=float).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size < 2:
        return []
    tolerance = np.ptp(values) * tolerance_ratio
    sequences: list[list[float]] = []
    current = [float(values[0])]
    for index in range(1, len(values)):
        current_value = float(values[index])
        previous = float(values[index - 1])
        small_reversal = (
            current_value - previous <= tolerance
            and index < len(values) - 1
            and values[index + 1] < current_value
        )
        if current_value < previous or small_reversal:
            current.append(current_value)
        else:
            if len(current) > 1:
                sequences.append(current)
            current = [current_value]
    if len(current) > 1:
        sequences.append(current)
    return sequences


def _drydown_metrics(values: np.ndarray) -> tuple[float, float, float]:
    sequences = [seq for seq in _decreasing_sequences(values) if len(seq) >= 3]
    if not sequences:
        finite = np.asarray(values, dtype=float)
        finite = finite[np.isfinite(finite)]
        return (float(finite.mean()) if finite.size else np.nan, 0.0, 0.0)
    return (
        float(np.mean([min(seq) for seq in sequences])),
        float(np.mean([len(seq) for seq in sequences])),
        float(np.mean([max(seq) - min(seq) for seq in sequences])),
    )


def _temperature_metrics(values_kelvin: np.ndarray, window: int):
    values = pd.Series(np.asarray(values_kelvin, dtype=float) - 273.15)
    rolling = values.rolling(window=window, center=True, min_periods=window).mean()
    return {
        "HT": float(rolling.max()),
        "LT": float(rolling.min()),
        "T": float(values.mean()),
        "CV": float(variation(values, nan_policy="omit")),
    }


def engineer_features(daily: pd.DataFrame) -> pd.DataFrame:
    """Create trailing predictors after 365 prior observations are available.

    This follows the original notebook's row-based historical window: the
    record at position 365 uses positions 0 through 364, and earlier records
    are not eligible for prediction.
    """
    daily = daily.sort_values("Date").copy()
    sm = daily["SM"].to_numpy(dtype=float)
    st = daily["ST"].to_numpy(dtype=float)
    records = []
    for index in range(len(daily)):
        row: dict[str, float] = {}
        for label, days in (("W", 7), ("M", 30), ("Y", 365)):
            start = index - days
            if start < 0:
                row.update(
                    {
                        f"SM_A_{label}_esm": np.nan,
                        f"SM_tau_{label}_esm": np.nan,
                        f"SM_theta_res_{label}_esm": np.nan,
                        f"A_{label}_HT": np.nan,
                        f"A_{label}_LT": np.nan,
                        f"A_{label}_T": np.nan,
                        f"CV_{label}_T": np.nan,
                    }
                )
                continue
            moisture = _drydown_metrics(sm[start:index])
            temperatures = _temperature_metrics(
                st[start:index], window=1 if label == "W" else 7
            )
            row.update(
                {
                    f"SM_A_{label}_esm": moisture[0],
                    f"SM_tau_{label}_esm": moisture[1],
                    f"SM_theta_res_{label}_esm": moisture[2],
                    f"A_{label}_HT": temperatures["HT"],
                    f"A_{label}_LT": temperatures["LT"],
                    f"A_{label}_T": temperatures["T"],
                    f"CV_{label}_T": temperatures["CV"],
                }
            )
        records.append(row)
    features = pd.concat([daily.reset_index(drop=True), pd.DataFrame(records)], axis=1)
    features["ST"] = features["ST"] - 273.15
    if len(features) <= 365:
        raise ValueError(
            "At least 366 daily observations are required: 365 historical "
            "observations plus the first prediction date."
        )
    return features.iloc[365:].reset_index(drop=True)


def load_soil_properties(path: Path) -> pd.DataFrame:
    soil = pd.read_csv(path)
    required = {"site", "soil_c_percent", "soil_n_percent", "c_n", "sand", "clay"}
    missing = required - set(soil.columns)
    if missing:
        raise ValueError(f"Soil-property CSV is missing columns: {sorted(missing)}")
    return soil


def build_prediction_table(
    data_root: Path, soil_path: Path, sites: Iterable[str]
) -> pd.DataFrame:
    soil = load_soil_properties(soil_path)
    frames = []
    for site in sites:
        site_features = engineer_features(read_daily_site(data_root, site))
        frames.append(site_features.merge(soil, on="site", how="left", validate="many_to_one"))
    result = pd.concat(frames, ignore_index=True)
    missing_soil = result.loc[result["soil_c_percent"].isna(), "site"].unique()
    if len(missing_soil):
        raise ValueError(f"Missing soil properties for: {', '.join(missing_soil)}")
    return result


def train_lignin_model(
    training_path: Path, model_path: Path, metrics_path: Path
) -> RandomForestRegressor:
    """Train the same four-feature lignin RF used by the original notebook."""
    if training_path.suffix.lower() == ".csv":
        training = pd.read_csv(training_path)
    elif training_path.suffix.lower() in {".pkl", ".pickle"}:
        training = pd.read_pickle(training_path)
    else:
        raise ValueError(
            f"Unsupported training-data format: {training_path.suffix}. "
            "Use CSV or pickle."
        )
    required = set(LIGNIN_FEATURES) | {"Lignin"}
    missing = required - set(training.columns)
    if missing:
        raise ValueError(f"Training table is missing columns: {sorted(missing)}")
    clean = training[list(LIGNIN_FEATURES) + ["Lignin"]].dropna()
    x_train, x_test, y_train, y_test = train_test_split(
        clean[list(LIGNIN_FEATURES)],
        clean["Lignin"],
        test_size=0.2,
        random_state=42,
    )
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(x_train, y_train)
    score = r2_score(y_test, model.predict(x_test))
    joblib.dump(model, model_path)
    pd.DataFrame(
        [{"target": "Lignin", "test_r2": score, "n_training_rows": len(clean)}]
    ).to_csv(metrics_path, index=False)
    return model


def predict_lignin(
    table: pd.DataFrame,
    model_path: Path,
    output_path: Path,
) -> pd.DataFrame:
    missing = set(LIGNIN_FEATURES) - set(table.columns)
    if missing:
        raise ValueError(f"Prediction table is missing columns: {sorted(missing)}")
    usable = table.dropna(subset=list(LIGNIN_FEATURES)).copy()
    model = joblib.load(model_path)
    expected = list(getattr(model, "feature_names_in_", LIGNIN_FEATURES))
    if expected != list(LIGNIN_FEATURES):
        raise ValueError(
            f"Model expects features {expected}, workflow provides {list(LIGNIN_FEATURES)}"
        )
    usable["EFC_lignin"] = model.predict(usable[list(LIGNIN_FEATURES)])
    columns = ["site", "Date", *LIGNIN_FEATURES, "EFC_lignin"]
    usable[columns].to_csv(output_path, index=False)
    return usable[columns]
