"""EFC prediction + driver-feature plotting (per-case, per-outcome figures).

Reads the two Stage-1/Stage-2 products of a case attempt directory:

    {case}_EFC_predictions.nc      (Stage 2: dims (time, soil_layer))
    {case}_features.parquet|csv    (Stage 1: long form, one row per Date x soil_layer)

and writes PNG figures, one self-contained folder per EFC outcome:

    {plot_dir}/{outcome}/
        {outcome}_prediction_15layer.png   # mean + median lines, q025-q975 band, 15 rows
        <driver features for THAT outcome>  # re-plotted in every outcome folder

Which features belong to each outcome is taken from ``manifest.json`` (the
exact trained-model input set per bundle). Each driver feature is rendered
according to how it varies in the feature table (detected programmatically,
not hardcoded), matching README §8a:

    static, depth-varying  (e.g. soil c (%), c/n, sand, clay)  -> depth profile
    dynamic, per-layer     (e.g. A_Y_T, SM, ST, SM_*_esm, CV_Y_T) -> 15-panel timeseries
    dynamic, site-level    (NPP_daily; identical across layers)  -> single timeseries

The EFC target scale is unit-agnostic in this repo (see README §8b), so EFC
y-axes are labeled "unitless (placeholder)". Driver-feature units are taken
from the table below (README §8a).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless (no display on compute/login nodes)
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xarray as xr  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DPI = 150
EFC_UNIT_LABEL = "unitless (placeholder)"
_VARY_TOL = 1e-12  # std threshold to call a column "varying" along an axis

# Driver-feature units (README §8a). Window-suffixed names resolve via prefix.
_FEATURE_UNITS = {
    "SM": "mm3/mm3",
    "ST": "degC",
    "NPP_daily": "gC/m^2/s",
    "sand": "%",
    "clay": "%",
    "soil c (%)": "%",
    "soil n (%)": "%",
    "c/n": "dimensionless",
}
# Prefix-based units for the windowed dynamic metrics.
_FEATURE_UNIT_PREFIXES = (
    ("SM_A_", "mm3/mm3"),
    ("SM_theta_res_", "mm3/mm3"),
    ("SM_tau_", "days"),
    ("A_", "degC"),      # A_W_HT / A_M_T / A_Y_T ...
    ("CV_", "dimensionless"),
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _slug(name: str) -> str:
    """manifest outcome -> NetCDF prefix / folder name (lowercase, spaces->_)."""
    return name.lower().replace(" ", "_")


def _safe_filename(feature: str) -> str:
    """Turn a feature column name into a filesystem-safe stem.

    e.g. 'soil c (%)' -> 'soil_c_percent'; 'c/n' -> 'c_n'; 'A_Y_T' -> 'A_Y_T'.
    """
    stem = (
        feature.replace("%", "percent")
        .replace("/", "_")
        .replace("(", "")
        .replace(")", "")
        .replace(" ", "_")
    )
    while "__" in stem:
        stem = stem.replace("__", "_")
    return stem.strip("_")


def _feature_units(feature: str) -> str:
    if feature in _FEATURE_UNITS:
        return _FEATURE_UNITS[feature]
    for prefix, unit in _FEATURE_UNIT_PREFIXES:
        if feature.startswith(prefix):
            return unit
    return "unitless"


def load_manifest(model_dir: str | Path) -> dict:
    return json.loads((Path(model_dir) / "manifest.json").read_text())


def load_feature_table(out_dir: str | Path, case_name: str) -> pd.DataFrame:
    """Load the Stage-1 table, preferring Parquet, falling back to CSV."""
    out_dir = Path(out_dir)
    parquet = out_dir / f"{case_name}_features.parquet"
    csv = out_dir / f"{case_name}_features.csv"
    if parquet.is_file():
        try:
            df = pd.read_parquet(parquet)
        except Exception as exc:  # pyarrow/fastparquet missing -> try CSV
            print(f"[plot][warn] could not read {parquet.name} "
                  f"({type(exc).__name__}: {exc}); falling back to CSV")
            df = None
        else:
            df["Date"] = pd.to_datetime(df["Date"])
            return df
    if csv.is_file():
        df = pd.read_csv(csv, parse_dates=["Date"])
        return df
    raise FileNotFoundError(
        f"No feature table found in {out_dir} "
        f"(looked for {parquet.name} and {csv.name})"
    )


def open_predictions(out_dir: str | Path, case_name: str) -> xr.Dataset:
    out_dir = Path(out_dir)
    nc = out_dir / f"{case_name}_EFC_predictions.nc"
    if not nc.is_file():
        raise FileNotFoundError(f"Predictions NetCDF not found: {nc}")
    return xr.open_dataset(nc)


def _variation(df: pd.DataFrame, feature: str) -> tuple[bool, bool]:
    """Return (varies_by_layer, varies_by_time) for a feature column.

    Uses a pivot (index=Date, columns=soil_layer). A column "varies by layer"
    if, on at least one date, the spread across layers exceeds the tolerance;
    it "varies by time" if any single layer's values change over time.
    """
    piv = df.pivot_table(index="Date", columns="soil_layer", values=feature)
    vals = piv.to_numpy(dtype=float)
    varies_layer = bool(np.nanmax(np.nanstd(vals, axis=1)) > _VARY_TOL)
    varies_time = bool(np.nanmax(np.nanstd(vals, axis=0)) > _VARY_TOL)
    return varies_layer, varies_time


# ---------------------------------------------------------------------------
# Figure renderers
# ---------------------------------------------------------------------------
def plot_efc_prediction(
    ds: xr.Dataset, outcome_prefix: str, out_path: Path, title_outcome: str
) -> None:
    """15-row (one per soil layer) EFC timeseries: mean + median + q-band."""
    required = [f"{outcome_prefix}_{s}" for s in ("mean", "median", "q025", "q975")]
    missing = [v for v in required if v not in ds.data_vars]
    if missing:
        raise KeyError(f"NetCDF missing variables for {outcome_prefix}: {missing}")

    times = pd.to_datetime(ds["time"].values)
    layers = ds["soil_layer"].values
    depths = ds["soil_node_depth_cm"].values
    n_layer = len(layers)

    mean = ds[f"{outcome_prefix}_mean"].values     # (time, layer)
    median = ds[f"{outcome_prefix}_median"].values
    q025 = ds[f"{outcome_prefix}_q025"].values
    q975 = ds[f"{outcome_prefix}_q975"].values

    fig, axes = plt.subplots(
        n_layer, 1, figsize=(11, 1.7 * n_layer), sharex=True, squeeze=False
    )
    axes = axes[:, 0]
    for li in range(n_layer):
        ax = axes[li]
        ax.fill_between(
            times, q025[:, li], q975[:, li],
            color="tab:blue", alpha=0.25, linewidth=0, label="q025-q975",
        )
        ax.plot(times, mean[:, li], color="tab:blue", lw=1.0, label="mean")
        ax.plot(times, median[:, li], color="tab:orange", lw=0.8, ls="--",
                label="median")
        ax.set_ylabel(EFC_UNIT_LABEL, fontsize=7)
        ax.tick_params(labelsize=7)
        ax.text(
            0.995, 0.92,
            f"layer {int(layers[li])} ({depths[li]:.1f} cm)",
            transform=ax.transAxes, ha="right", va="top", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.7", alpha=0.8),
        )
        if li == 0:
            ax.legend(loc="upper left", fontsize=7, ncol=3, framealpha=0.8)
    axes[-1].xaxis.set_major_locator(mdates.YearLocator())
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[-1].set_xlabel("date")
    fig.suptitle(
        f"EFC prediction: {title_outcome} (mean, median, 95% band) by soil layer",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)


def plot_feature_profile(
    df: pd.DataFrame, feature: str, out_path: Path, title_outcome: str
) -> None:
    """Static (time-invariant) feature: value vs depth (surface at top)."""
    prof = (
        df.drop_duplicates("soil_layer")
        .sort_values("soil_node_depth_cm")[["soil_node_depth_cm", feature]]
    )
    fig, ax = plt.subplots(figsize=(5, 7))
    ax.plot(prof[feature].to_numpy(), prof["soil_node_depth_cm"].to_numpy(),
            marker="o", color="tab:green")
    ax.invert_yaxis()  # surface on top
    ax.set_xlabel(f"{feature} [{_feature_units(feature)}]")
    ax.set_ylabel("soil node depth (cm)")
    ax.set_yscale("log")  # depths span 0.7 cm -> 3500 cm
    ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.set_title(f"{title_outcome} driver: {feature} (depth profile)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)


def plot_feature_layers_timeseries(
    df: pd.DataFrame, feature: str, out_path: Path, title_outcome: str
) -> None:
    """Dynamic per-layer feature: 15 stacked timeseries panels (one per layer)."""
    piv = df.pivot_table(index="Date", columns="soil_layer", values=feature)
    depth_by_layer = (
        df.drop_duplicates("soil_layer")
        .set_index("soil_layer")["soil_node_depth_cm"]
    )
    layers = list(piv.columns)
    times = pd.to_datetime(piv.index)
    n_layer = len(layers)
    unit = _feature_units(feature)

    fig, axes = plt.subplots(
        n_layer, 1, figsize=(11, 1.7 * n_layer), sharex=True, squeeze=False
    )
    axes = axes[:, 0]
    for li, layer in enumerate(layers):
        ax = axes[li]
        ax.plot(times, piv[layer].to_numpy(), color="tab:purple", lw=1.0)
        ax.set_ylabel(unit, fontsize=7)
        ax.tick_params(labelsize=7)
        depth = float(depth_by_layer.get(layer, np.nan))
        ax.text(
            0.995, 0.92, f"layer {int(layer)} ({depth:.1f} cm)",
            transform=ax.transAxes, ha="right", va="top", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.7", alpha=0.8),
        )
    axes[-1].xaxis.set_major_locator(mdates.YearLocator())
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[-1].set_xlabel("date")
    fig.suptitle(
        f"{title_outcome} driver: {feature} [{unit}] by soil layer", fontsize=12
    )
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)


def plot_feature_single_timeseries(
    df: pd.DataFrame, feature: str, out_path: Path, title_outcome: str
) -> None:
    """Dynamic site-level feature (identical across layers): one timeseries."""
    # Site-level features (only NPP_daily here) are byte-identical across layers
    # by construction, so layer 1's series fully represents the site.
    one = (
        df[df.soil_layer.eq(df.soil_layer.min())]
        .sort_values("Date")[["Date", feature]]
    )
    unit = _feature_units(feature)
    fig, ax = plt.subplots(figsize=(11, 3.2))
    ax.plot(pd.to_datetime(one["Date"]), one[feature].to_numpy(),
            color="tab:red", lw=1.0)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_xlabel("date")
    ax.set_ylabel(f"{feature} [{unit}]")
    ax.grid(True, ls=":", alpha=0.4)
    ax.set_title(f"{title_outcome} driver: {feature} (site-level timeseries)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)


def plot_feature(
    df: pd.DataFrame, feature: str, out_dir: Path, title_outcome: str
) -> str:
    """Dispatch a single driver feature to the right renderer by its variation.

    Returns the written PNG path.
    """
    if feature not in df.columns:
        raise KeyError(f"feature {feature!r} not in table columns")
    varies_layer, varies_time = _variation(df, feature)
    stem = _safe_filename(feature)

    if not varies_time and varies_layer:
        out_path = out_dir / f"{stem}_profile.png"
        plot_feature_profile(df, feature, out_path, title_outcome)
    elif varies_time and not varies_layer:
        out_path = out_dir / f"{stem}_timeseries.png"
        plot_feature_single_timeseries(df, feature, out_path, title_outcome)
    elif varies_time and varies_layer:
        out_path = out_dir / f"{stem}_15layer.png"
        plot_feature_layers_timeseries(df, feature, out_path, title_outcome)
    else:
        # Constant everywhere (no variation at all): still emit a tiny profile
        # so the folder is self-contained and the user can see the value.
        out_path = out_dir / f"{stem}_profile.png"
        plot_feature_profile(df, feature, out_path, title_outcome)
    return str(out_path)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def _resolve_outcomes(manifest: dict, plot_vars) -> list[str]:
    """Map a requested selection (names/slugs/'all') to manifest outcome names."""
    all_outcomes = list(manifest["outcomes"])
    if plot_vars is None or (isinstance(plot_vars, str) and plot_vars.lower() == "all"):
        return all_outcomes
    if isinstance(plot_vars, str):
        requested = [v.strip() for v in plot_vars.split(",") if v.strip()]
    else:
        requested = list(plot_vars)
    by_slug = {_slug(o): o for o in all_outcomes}
    by_lower = {o.lower(): o for o in all_outcomes}
    resolved: list[str] = []
    unknown: list[str] = []
    for req in requested:
        key = req.strip()
        if key in all_outcomes:
            resolved.append(key)
        elif _slug(key) in by_slug:
            resolved.append(by_slug[_slug(key)])
        elif key.lower() in by_lower:
            resolved.append(by_lower[key.lower()])
        else:
            unknown.append(req)
    if unknown:
        raise ValueError(
            f"unknown --plot-vars {unknown}; valid outcomes: {all_outcomes} "
            f"(or slugs: {list(by_slug)})"
        )
    # de-duplicate, preserve order
    seen: set[str] = set()
    return [o for o in resolved if not (o in seen or seen.add(o))]


def generate_plots(
    out_dir: str | Path,
    case_name: str,
    model_dir: str | Path,
    plot_dir: str | Path | None = None,
    plot_vars=None,
) -> dict:
    """Generate all requested per-outcome plot folders.

    Parameters
    ----------
    out_dir : directory holding ``{case}_EFC_predictions.nc`` and the feature table.
    case_name : case label used in the product filenames.
    model_dir : model artifacts dir containing ``manifest.json`` (for per-outcome
        feature sets).
    plot_dir : destination root for figures (default ``{out_dir}/plots``).
    plot_vars : 'all' | comma-string | iterable of outcome names/slugs.

    Returns a summary dict: {outcome: [png paths...]}.
    """
    out_dir = Path(out_dir)
    plot_dir = Path(plot_dir) if plot_dir is not None else out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(model_dir)
    outcomes = _resolve_outcomes(manifest, plot_vars)
    feats_by_outcome = {b["outcome"]: list(b["features"]) for b in manifest["bundles"]}

    df = load_feature_table(out_dir, case_name)
    ds = open_predictions(out_dir, case_name)

    summary: dict[str, list[str]] = {}
    try:
        for outcome in outcomes:
            prefix = _slug(outcome)
            folder = plot_dir / prefix
            folder.mkdir(parents=True, exist_ok=True)
            written: list[str] = []

            efc_png = folder / f"{prefix}_prediction_15layer.png"
            plot_efc_prediction(ds, prefix, efc_png, outcome)
            written.append(str(efc_png))

            for feature in feats_by_outcome.get(outcome, []):
                if feature not in df.columns:
                    # Should not happen (validated at Stage 1), but keep going.
                    print(f"[plot][warn] {outcome}: feature {feature!r} missing "
                          f"from table; skipping")
                    continue
                written.append(plot_feature(df, feature, folder, outcome))

            summary[outcome] = written
            print(f"[plot] {outcome}: {len(written)} figure(s) -> {folder}")
    finally:
        ds.close()

    return summary
