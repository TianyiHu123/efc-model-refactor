"""Generate five-layer cumulative-70% EFC inputs for multiple NEON sites."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from build_rmnp_layer_model_inputs import N_LAYERS, layer_features, read_daily


ROOT = Path(__file__).resolve().parent
CLM_ROOT = ROOT / "EFC inputs from CLM"
SURFACE_ROOT = CLM_ROOT / "site_surface_Files"
FEATURE_JSON = (
    ROOT / "Submission3_ISME_Verson" / "analysis_outputs" / "importance_robustness"
    / "results" / "selected15_cumulative70_features.json"
)
DEFAULT_SITES = ("CLBJ", "YELL", "TALL", "JERC", "WOOD", "TEAK", "WREF")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sites", nargs="+", default=list(DEFAULT_SITES))
    parser.add_argument("--output-root", type=Path, default=ROOT / "model_inputs")
    return parser.parse_args()


def choose_history_files(site: str) -> tuple[list[Path], Path]:
    site_root = CLM_ROOT / site
    all_files = sorted(site_root.glob("**/*.clm2.h0.*.nc"))
    if not all_files:
        raise FileNotFoundError(f"No CLM history files for {site}")
    final_files = [p for p in all_files if any("final" in part.lower() for part in p.parts)]
    candidates = final_files or all_files

    # A few archived runs contain a second root-level copy. Prefer the normal
    # land history directory, then a hist directory, then the shallowest set.
    parent_groups: dict[Path, list[Path]] = {}
    for path in candidates:
        parent_groups.setdefault(path.parent, []).append(path)
    ranked = sorted(
        parent_groups,
        key=lambda p: (
            0 if tuple(x.lower() for x in p.parts[-2:]) == ("lnd", "hist") else
            1 if p.name.lower() == "hist" else 2,
            -len(parent_groups[p]), len(p.parts), str(p),
        ),
    )
    folder = ranked[0]
    return sorted(parent_groups[folder]), folder


def surfdata_path(site: str) -> Path:
    matches = sorted(CLM_ROOT.glob(f"surfdata_1x1pt_{site}_hist_*.nc"))
    if len(matches) != 1:
        raise ValueError(f"Expected one surfdata file for {site}; found {len(matches)}")
    return matches[0]


def read_texture(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with xr.open_dataset(path, decode_times=False) as data:
        for name in ("PCT_SAND", "PCT_CLAY", "ORGANIC"):
            if name not in data:
                raise KeyError(f"{path.name} is missing {name}")
        sand = np.asarray(data.PCT_SAND.values, dtype=float).squeeze()[:N_LAYERS]
        clay = np.asarray(data.PCT_CLAY.values, dtype=float).squeeze()[:N_LAYERS]
        organic = np.asarray(data.ORGANIC.values, dtype=float).squeeze()[:N_LAYERS]
    if any(values.shape != (N_LAYERS,) for values in (sand, clay, organic)):
        raise ValueError(f"Unexpected first-five-layer shape in {path.name}")
    return sand, clay, organic


def chemistry_and_carbon_audit(site: str, depths: np.ndarray, organic: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    path = SURFACE_ROOT / f"{site}_surfaceData.csv"
    columns = [
        "siteID", "horizonName", "biogeoTopDepth", "biogeoBottomDepth",
        "carbonTot", "nitrogenTot", "bulkDensExclCoarseFrag",
    ]
    source = pd.read_csv(path, usecols=columns)
    source = source.loc[source.siteID.eq(site)].copy()
    numeric = columns[2:]
    for column in numeric:
        source[column] = pd.to_numeric(source[column], errors="coerce")
    source = source.dropna(subset=numeric)
    source["horizon_midpoint_cm"] = (
        source.biogeoTopDepth + source.biogeoBottomDepth
    ) / 2.0
    valid_n = (
        source.loc[source.nitrogenTot.gt(0)]
        .groupby("horizon_midpoint_cm", as_index=False)
        .nitrogenTot.mean()
        .sort_values("horizon_midpoint_cm")
    )
    if valid_n.empty:
        raise ValueError(f"{site} has no positive nitrogenTot values for interpolation")

    chemistry_rows, audit_rows = [], []
    for layer, (depth, netcdf_organic) in enumerate(zip(depths, organic), start=1):
        matches = source.loc[
            source.biogeoTopDepth.le(depth) & source.biogeoBottomDepth.gt(depth)
        ].copy()
        if matches.empty:
            raise ValueError(f"{site} layer {layer} ({depth} cm) matched no CSV horizon")
        identity = ["horizonName", "biogeoTopDepth", "biogeoBottomDepth"]
        horizons = matches.groupby(identity, as_index=False).agg(
            carbonTot=("carbonTot", "mean"), nitrogenTot=("nitrogenTot", "mean")
        )
        if len(horizons) != 1:
            raise ValueError(f"{site} layer {layer} matched multiple horizon intervals")
        horizon = horizons.iloc[0]
        if horizon.carbonTot <= 0:
            raise ValueError(f"{site} layer {layer} has nonpositive soil C")
        nitrogen_interpolated = bool(horizon.nitrogenTot <= 0)
        nitrogen_value = float(horizon.nitrogenTot)
        if nitrogen_interpolated:
            nitrogen_value = float(np.interp(
                depth,
                valid_n.horizon_midpoint_cm.to_numpy(float),
                valid_n.nitrogenTot.to_numpy(float),
            ))
            if nitrogen_value <= 0:
                raise ValueError(f"{site} layer {layer} nitrogen interpolation was nonpositive")
        chemistry_rows.append({
            "soil_layer": layer,
            "biogeo_horizon": str(horizon.horizonName),
            "biogeo_top_depth_cm": float(horizon.biogeoTopDepth),
            "biogeo_bottom_depth_cm": float(horizon.biogeoBottomDepth),
            "soil c (%)": float(horizon.carbonTot) / 10.0,
            "soil n (%)": nitrogen_value / 10.0,
            "c/n": float(horizon.carbonTot) / nitrogen_value,
            "nitrogen_interpolated": nitrogen_interpolated,
            "nitrogen_interpolation_method": (
                "linear interpolation of positive nitrogenTot by horizon midpoint"
                if nitrogen_interpolated else "observed nitrogenTot"
            ),
        })

        # NetCDF ORGANIC is kg OM/m3 and states C=0.58*OM. For every CSV
        # replicate, reconstruct ORGANIC from bulk density (g/cm3) and C (g/kg).
        matches["reconstructed_organic_kg_m3"] = (
            matches.bulkDensExclCoarseFrag * matches.carbonTot / 0.58
        )
        matches["absolute_difference"] = (
            matches.reconstructed_organic_kg_m3 - netcdf_organic
        ).abs()
        matches["netcdf_implied_soil_c_percent"] = (
            float(netcdf_organic) * 0.58 / matches.bulkDensExclCoarseFrag / 10.0
        )
        matches["soil_c_difference_percentage_points"] = (
            matches.netcdf_implied_soil_c_percent - matches.carbonTot / 10.0
        ).abs()
        closest = matches.sort_values("absolute_difference").iloc[0]
        # carbonTot is reported to 0.1 g/kg, equivalent to 0.01 percentage
        # points. Allow 0.05 points to distinguish rounding from a source mismatch.
        percent_tolerance = 0.05
        audit_rows.append({
            "site": site, "soil_layer": layer, "soil_node_depth_cm": float(depth),
            "biogeo_horizon": str(horizon.horizonName),
            "csv_carbon_g_kg": float(horizon.carbonTot),
            "csv_soil_c_percent": float(horizon.carbonTot) / 10.0,
            "csv_nitrogen_g_kg": float(horizon.nitrogenTot),
            "used_nitrogen_g_kg": nitrogen_value,
            "used_cn": float(horizon.carbonTot) / nitrogen_value,
            "nitrogen_interpolated": nitrogen_interpolated,
            "netcdf_organic_kg_m3": float(netcdf_organic),
            "closest_csv_bulk_density_g_cm3": float(closest.bulkDensExclCoarseFrag),
            "reconstructed_organic_kg_m3": float(closest.reconstructed_organic_kg_m3),
            "absolute_difference_kg_m3": float(closest.absolute_difference),
            "netcdf_implied_soil_c_percent": float(closest.netcdf_implied_soil_c_percent),
            "soil_c_difference_percentage_points": float(closest.soil_c_difference_percentage_points),
            "carbon_consistency_tolerance_percentage_points": percent_tolerance,
            "carbon_consistent": bool(closest.soil_c_difference_percentage_points <= percent_tolerance),
            "comparison_formula": "ORGANIC = bulkDensExclCoarseFrag * carbonTot / 0.58",
        })
    return pd.DataFrame(chemistry_rows), pd.DataFrame(audit_rows)


def build_site(site: str, output_root: Path, selected: dict[str, list[str]]) -> dict[str, object]:
    files, history_folder = choose_history_files(site)
    dates, sm, st, npp, depths = read_daily(files)
    surf_path = surfdata_path(site)
    sand, clay, organic = read_texture(surf_path)
    chemistry, carbon_audit = chemistry_and_carbon_audit(site, depths, organic)

    frames = []
    for index, depth in enumerate(depths):
        frame = layer_features(dates, sm, st, npp, index, float(depth))
        frame["site"] = site
        frame["sand"] = sand[index]
        frame["clay"] = clay[index]
        row = chemistry.loc[chemistry.soil_layer.eq(index + 1)].iloc[0]
        for column in [
            "biogeo_horizon", "biogeo_top_depth_cm", "biogeo_bottom_depth_cm",
            "soil c (%)", "soil n (%)", "c/n",
        ]:
            frame[column] = row[column]
        frames.append(frame)
    master = pd.concat(frames, ignore_index=True).sort_values(["Date", "soil_layer"])
    model_features = sorted({feature for features in selected.values() for feature in features})
    if master[model_features].isna().any().any():
        raise ValueError(f"{site} contains missing model features")
    if master.duplicated(["Date", "soil_layer"]).any():
        raise ValueError(f"{site} contains duplicate date-layer rows")

    output = output_root / f"{site}_cumulative70"
    output.mkdir(parents=True, exist_ok=True)
    master.to_csv(output / f"{site}_five_layer_all_features.csv", index=False, date_format="%Y-%m-%d")
    identifiers = ["site", "Date", "soil_layer", "soil_node_depth_cm"]
    for outcome, features in selected.items():
        safe = outcome.lower().replace(" ", "_")
        master[identifiers + features].to_csv(
            output / f"{site}_{safe}_cumulative70_model_input.csv",
            index=False, date_format="%Y-%m-%d",
        )
    carbon_audit.to_csv(output / f"{site}_soil_carbon_consistency.csv", index=False)
    result = {
        "site": site, "history_folder": str(history_folder), "history_file_count": len(files),
        "source_first_date": str(dates.min().date()), "source_last_date": str(dates.max().date()),
        "output_first_date": str(master.Date.min().date()), "output_last_date": str(master.Date.max().date()),
        "rows": len(master), "rows_per_layer": int(len(master) / N_LAYERS),
        "soil_node_depths_cm": depths.tolist(), "surfdata": str(surf_path),
        "surface_csv": str(SURFACE_ROOT / f"{site}_surfaceData.csv"),
        "soil_chemistry_by_layer": chemistry.to_dict(orient="records"),
        "carbon_consistency_by_layer": carbon_audit.carbon_consistent.tolist(),
        "all_first_five_carbon_consistent": bool(carbon_audit.carbon_consistent.all()),
        "outcome_features": selected,
    }
    (output / f"{site}_model_input_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    args = parse_args()
    sites = [site.upper() for site in args.sites]
    with FEATURE_JSON.open() as handle:
        selected = json.load(handle)
    results = []
    for site in sites:
        result = build_site(site, args.output_root, selected)
        results.append(result)
        print(site, result["rows"], result["all_first_five_carbon_consistent"], result["history_folder"])
    (args.output_root / "multisite_cumulative70_audit.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
