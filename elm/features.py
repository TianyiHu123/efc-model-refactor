"""Trailing hydrothermal feature engineering (ported from clm_ref).

Reproduces the exact feature contract the pre-trained EFC models expect, as
implemented in ``clm_ref/build_rmnp_layer_model_inputs.py`` (``layer_features``)
and ``clm_ref/efc_workflow.py`` (``_drydown_metrics``). Generalized to an
arbitrary number of soil layers.

For each prediction day ``current`` (starting at index 365, i.e. once 365 prior
days exist) and each layer, three trailing windows are computed:

    label  count  rolling_days
    W      7      1
    M      30     7
    Y      365    7

The window excludes the current day: ``values[current - count : current]``.

Per window:
  - soil-moisture drydown metrics -> SM_A_{L}_esm, SM_tau_{L}_esm, SM_theta_res_{L}_esm
  - temperature metrics (deg C)   -> A_{L}_HT, A_{L}_LT, A_{L}_T, CV_{L}_T

Instantaneous columns at ``current``: SM, ST (deg C), NPP_daily.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import variation

_WINDOWS = (("W", 7, 1), ("M", 30, 7), ("Y", 365, 7))
_WARMUP = 365


# --- ported verbatim (logic) from clm_ref/efc_workflow.py ---------------------
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


def drydown_metrics(values: np.ndarray) -> tuple[float, float, float]:
    """(A, tau, theta_res) drydown metrics; mirrors clm_ref._drydown_metrics."""
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


def temperature_metrics(values_kelvin: np.ndarray, rolling_days: int) -> dict[str, float]:
    """HT/LT/T/CV temperature metrics in deg C; mirrors clm_ref.temperature_metrics."""
    values_c = pd.Series(np.asarray(values_kelvin, dtype=float) - 273.15)
    rolling = values_c.rolling(rolling_days, center=True, min_periods=rolling_days).mean()
    return {
        "HT": float(rolling.max()),
        "LT": float(rolling.min()),
        "T": float(values_c.mean()),
        "CV": float(variation(values_c, nan_policy="omit")),
    }
# -----------------------------------------------------------------------------


def layer_features(
    dates: pd.DatetimeIndex,
    sm: np.ndarray,
    st: np.ndarray,
    npp: np.ndarray,
    layer: int,
    depth_cm: float,
    site: str,
) -> pd.DataFrame:
    """Build the long-form trailing-feature table for one soil layer."""
    records: list[dict[str, object]] = []
    n = len(dates)
    for current in range(_WARMUP, n):
        row: dict[str, object] = {
            "site": site,
            "Date": dates[current],
            "soil_layer": layer + 1,
            "soil_node_depth_cm": depth_cm,
            "SM": float(sm[current, layer]),
            "ST": float(st[current, layer] - 273.15),
            "NPP_daily": float(npp[current]),
            "history_start": dates[current - _WARMUP],
            "history_end": dates[current - 1],
        }
        for label, count, rolling_days in _WINDOWS:
            moisture = drydown_metrics(sm[current - count:current, layer])
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
        records.append(row)
    return pd.DataFrame(records)
