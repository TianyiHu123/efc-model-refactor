"""Read ELM run-folder NetCDF and aggregate hourly -> daily.

Source files (produced by ``ncrcat`` in elm-olmt postprocessing), in the case
run directory:

    H2OSOI.nc  (time, levgrnd, lndgrid)  units mm3/mm3   -> soil moisture
    TSOI.nc    (time, levgrnd, lndgrid)  units K         -> soil temperature
    NPP.nc     (time, lndgrid)           units gC/m^2/s  -> scalar flux

``time`` is ``days since {start}`` on a ``noleap`` calendar at hourly cadence.
We aggregate to daily means (no-leap) and read node depths from the ``levgrnd``
coordinate (meters -> cm). Soil layers are ALL ``levgrnd`` layers present.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

_EXPECTED_UNITS = {
    "H2OSOI": {"mm3/mm3", "m3/m3"},
    "TSOI": {"K"},
    "NPP": {"gC/m^2/s"},
}


@dataclass
class DailyCase:
    dates: pd.DatetimeIndex          # (ndays,) no-leap daily dates
    sm: np.ndarray                   # (ndays, nlayer) soil moisture mm3/mm3
    st: np.ndarray                   # (ndays, nlayer) soil temperature K
    npp: np.ndarray                  # (ndays,) daily-mean NPP gC/m^2/s
    depths_cm: np.ndarray            # (nlayer,) levgrnd node depths in cm

    @property
    def nlayer(self) -> int:
        return self.sm.shape[1]


def _require(run_dir: Path, name: str) -> Path:
    path = run_dir / f"{name}.nc"
    if not path.is_file():
        raise FileNotFoundError(f"Expected run-dir file not found: {path}")
    return path


def _check_units(var: str, actual: str | None) -> None:
    allowed = _EXPECTED_UNITS[var]
    if actual not in allowed:
        raise ValueError(
            f"{var}: unexpected units {actual!r}; expected one of {sorted(allowed)}"
        )


def _day_index(time_values: np.ndarray) -> np.ndarray:
    """Floor hourly 'days since start' stamps to integer day index."""
    # Round first to absorb float noise (e.g. 0.041666668), then floor.
    return np.floor(np.round(np.asarray(time_values, dtype=float), 6)).astype(int)


def _parse_time_reference(attrs: dict) -> tuple[int, int, int, str]:
    """Return (epoch_year, epoch_month, epoch_day, calendar) from time attrs.

    Expects units like 'days since YYYY-MM-DD[ HH:MM:SS]'. Raises if the units
    are not day-based or cannot be parsed.
    """
    units = str(attrs.get("units", "")).strip()
    calendar = str(attrs.get("calendar", "")).strip().lower()
    if "since" not in units or not units.lower().startswith("days"):
        raise ValueError(
            f"time units must be 'days since ...'; got {units!r}"
        )
    datepart = units.split("since", 1)[1].strip().split()[0]  # 'YYYY-MM-DD'
    try:
        y, m, d = (int(x) for x in datepart.split("-")[:3])
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"cannot parse epoch date from time units {units!r}") from exc
    return y, m, d, calendar


def _validate_daily_grid(day_idx: np.ndarray, steps_per_day_expected: int | None = None) -> None:
    """Assert day indices start at 0, are contiguous, and have uniform counts.

    Guards against truncated boundary days or gaps that would skew daily means.
    """
    unique, counts = np.unique(day_idx, return_counts=True)
    if unique[0] != 0:
        raise ValueError(f"time axis does not start at day 0 (first day index {unique[0]})")
    expected = np.arange(unique[0], unique[-1] + 1)
    if not np.array_equal(unique, expected):
        missing = sorted(set(expected.tolist()) - set(unique.tolist()))
        raise ValueError(f"non-contiguous day indices; missing days: {missing[:10]}")
    if len(set(counts.tolist())) != 1:
        from collections import Counter
        raise ValueError(
            "non-uniform samples per day (truncated/partial day present): "
            f"counts={dict(Counter(counts.tolist()))}"
        )
    steps = int(counts[0])
    if steps_per_day_expected is not None and steps != steps_per_day_expected:
        raise ValueError(
            f"expected {steps_per_day_expected} samples/day, found {steps}"
        )


def read_case(run_dir: str | Path, start_year: int, use_vertsoilc: bool) -> DailyCase:
    """Read the three run-dir variables and return daily-aggregated arrays.

    Raises loudly if the soil variables are single-layer while
    ``use_vertsoilc`` is True (indicates a non-vertical run mislabeled).
    """
    run_dir = Path(run_dir)
    h2o_path = _require(run_dir, "H2OSOI")
    tsoi_path = _require(run_dir, "TSOI")
    npp_path = _require(run_dir, "NPP")

    with xr.open_dataset(h2o_path, decode_times=False) as h2o, \
         xr.open_dataset(tsoi_path, decode_times=False) as tsoi, \
         xr.open_dataset(npp_path, decode_times=False) as npp_ds:

        _check_units("H2OSOI", h2o["H2OSOI"].attrs.get("units"))
        _check_units("TSOI", tsoi["TSOI"].attrs.get("units"))
        _check_units("NPP", npp_ds["NPP"].attrs.get("units"))

        if "levgrnd" not in h2o.coords and "levgrnd" not in h2o.variables:
            raise KeyError("H2OSOI.nc has no 'levgrnd' coordinate")
        nlayer = int(h2o.sizes.get("levgrnd", 0))
        if nlayer < 1:
            raise ValueError("H2OSOI.nc has no levgrnd layers")
        if use_vertsoilc and nlayer == 1:
            raise ValueError(
                "use_vertsoilc=True but soil variables are single-layer "
                "(levgrnd=1); refusing to proceed with ambiguous vertical shape."
            )
        if int(tsoi.sizes.get("levgrnd", 0)) != nlayer:
            raise ValueError("H2OSOI and TSOI have different levgrnd sizes")

        depths_cm = np.asarray(h2o["levgrnd"].values, dtype=float) * 100.0

        # Validate the time reference against the caller's start_year (M1).
        epoch_year, epoch_month, epoch_day, calendar = _parse_time_reference(
            h2o["time"].attrs
        )
        if calendar and calendar not in ("noleap", "365_day"):
            raise ValueError(
                f"expected a noleap/365_day calendar; got {calendar!r}"
            )
        if (epoch_month, epoch_day) != (1, 1):
            raise ValueError(
                f"time epoch must be Jan 1; got {epoch_year}-{epoch_month:02d}-{epoch_day:02d}"
            )
        if epoch_year != start_year:
            raise ValueError(
                f"time epoch year {epoch_year} != trans_startyear {start_year}; "
                "date construction would be wrong."
            )

        # Daily grouping via integer day index (no-leap).
        time_values = np.asarray(h2o["time"].values, dtype=float)
        day_idx = _day_index(time_values)
        # Enforce a complete, contiguous hourly grid (24/day) (M2).
        _validate_daily_grid(day_idx, steps_per_day_expected=24)

        sm_hourly = np.asarray(h2o["H2OSOI"].isel(lndgrid=0).values, dtype=float)
        st_hourly = np.asarray(tsoi["TSOI"].isel(lndgrid=0).values, dtype=float)
        npp_hourly = np.asarray(npp_ds["NPP"].isel(lndgrid=0).values, dtype=float)

    # shape sanity
    if sm_hourly.shape != st_hourly.shape:
        raise ValueError("H2OSOI and TSOI hourly shapes differ")
    if sm_hourly.shape[0] != len(day_idx) or npp_hourly.shape[0] != len(day_idx):
        raise ValueError("variable/time length mismatch")

    sm = _daily_mean_2d(sm_hourly, day_idx)
    st = _daily_mean_2d(st_hourly, day_idx)
    npp = _daily_mean_1d(npp_hourly, day_idx)

    dates = _noleap_dates(start_year, n_days=sm.shape[0])
    if len(dates) != sm.shape[0]:
        raise ValueError(
            f"date count {len(dates)} != daily rows {sm.shape[0]}"
        )
    return DailyCase(dates=dates, sm=sm, st=st, npp=npp, depths_cm=depths_cm)


def _daily_mean_2d(values: np.ndarray, day_idx: np.ndarray) -> np.ndarray:
    frame = pd.DataFrame(values)
    frame["_day"] = day_idx
    grouped = frame.groupby("_day", sort=True).mean()
    return grouped.to_numpy(dtype=float)


def _daily_mean_1d(values: np.ndarray, day_idx: np.ndarray) -> np.ndarray:
    series = pd.Series(values)
    return series.groupby(day_idx, sort=True).mean().to_numpy(dtype=float)


def _noleap_dates(start_year: int, n_days: int) -> pd.DatetimeIndex:
    """Build n_days consecutive no-leap dates from Jan 1 of start_year.

    Uses a 365-day year (drops Feb 29) to match ELM's noleap calendar, then
    represents the dates as pandas Timestamps for downstream convenience.
    """
    dates: list[pd.Timestamp] = []
    year, month, day = start_year, 1, 1
    days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    for _ in range(n_days):
        dates.append(pd.Timestamp(year, month, day))
        day += 1
        if day > days_in_month[month - 1]:
            day = 1
            month += 1
            if month > 12:
                month = 1
                year += 1
    return pd.DatetimeIndex(dates)
