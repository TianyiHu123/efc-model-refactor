"""Static soil properties: texture (surfdata) and C/N (NEON csv).

Texture
-------
``PCT_SAND`` / ``PCT_CLAY`` live on ``nlevsoi`` in the surfdata NetCDF (10 layers
for the verified ABBY case), but the engine engineers features on ``levgrnd``
(15 layers). Texture is mapped by layer index for the top ``nlevsoi`` layers;
deeper ``levgrnd`` nodes (bedrock, no surfdata texture) carry the deepest soil
value forward. This mapping is recorded in provenance.

Carbon / Nitrogen
-----------------
Ported from ``clm_ref`` (``depth_resolved_carbon_nitrogen``): each soil node
center depth is matched to the NEON horizon interval that contains it;
``carbonTot`` / ``nitrogenTot`` (g/kg) are converted to mass percent (/10) and
``c/n`` is ``carbonTot / nitrogenTot``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


@dataclass
class SoilProperties:
    sand: np.ndarray                 # (nlayer,) percent
    clay: np.ndarray                 # (nlayer,) percent
    soil_c_percent: np.ndarray       # (nlayer,)
    soil_n_percent: np.ndarray       # (nlayer,)
    c_n: np.ndarray                  # (nlayer,)
    texture_layers_native: int       # nlevsoi count in surfdata
    texture_extended: bool           # True if deepest value carried forward
    cn_below_horizon_layers: int     # layers deeper than the deepest NEON horizon


def read_texture(surffile: str | Path, nlayer: int) -> tuple[np.ndarray, np.ndarray, int, bool]:
    """Return (sand, clay, native_nlevsoi, extended) aligned to ``nlayer``."""
    surffile = Path(surffile)
    if not surffile.is_file():
        raise FileNotFoundError(f"Surfdata file not found: {surffile}")
    with xr.open_dataset(surffile, decode_times=False) as ds:
        for name in ("PCT_SAND", "PCT_CLAY"):
            if name not in ds.variables:
                raise KeyError(f"{surffile.name} is missing {name}")
        # Select a single gridcell explicitly rather than relying on squeeze
        # order, then keep the soil-level axis (nlevsoi).
        sand_da = ds["PCT_SAND"]
        clay_da = ds["PCT_CLAY"]
        level_dim = next((d for d in sand_da.dims if "lev" in d.lower()), None)
        if level_dim is None:
            raise ValueError(
                f"{surffile.name} PCT_SAND has no soil-level dim; dims={sand_da.dims}"
            )
        for other in [d for d in sand_da.dims if d != level_dim]:
            sand_da = sand_da.isel({other: 0})
            clay_da = clay_da.isel({other: 0})
        sand = np.asarray(sand_da.values, dtype=float).reshape(-1)
        clay = np.asarray(clay_da.values, dtype=float).reshape(-1)
    native = int(sand.shape[0])
    if clay.shape[0] != native:
        raise ValueError("PCT_SAND and PCT_CLAY have different layer counts")
    if not (np.all((sand >= 0) & (sand <= 100)) and np.all((clay >= 0) & (clay <= 100))):
        raise ValueError("sand/clay percentages outside 0-100")

    extended = False
    if native == nlayer:
        return sand, clay, native, extended
    if native < nlayer:
        # carry deepest soil value forward into deeper levgrnd (bedrock) nodes
        pad = nlayer - native
        sand = np.concatenate([sand, np.repeat(sand[-1], pad)])
        clay = np.concatenate([clay, np.repeat(clay[-1], pad)])
        extended = True
        return sand, clay, native, extended
    # native > nlayer: truncate to the engineered layers
    return sand[:nlayer], clay[:nlayer], native, extended


def read_carbon_nitrogen(
    neon_csv: str | Path, site: str, depths_cm: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Return (soil_c_percent, soil_n_percent, c_n, n_below_horizon) by depth.

    Ported from clm_ref. Duplicate horizon intervals are averaged. If a horizon
    reports non-positive nitrogen, it is linearly interpolated from positive
    nitrogen values at neighboring horizon midpoints. ``n_below_horizon`` counts
    layers deeper than the deepest NEON horizon (they reuse the deepest horizon's
    chemistry) for provenance.
    """
    neon_csv = Path(neon_csv)
    if not neon_csv.is_file():
        raise FileNotFoundError(f"NEON surface CSV not found: {neon_csv}")
    cols = ["siteID", "horizonName", "biogeoTopDepth", "biogeoBottomDepth",
            "carbonTot", "nitrogenTot"]
    df = pd.read_csv(neon_csv, usecols=cols)
    df = df.loc[df.siteID.eq(site)].copy()
    for c in ("biogeoTopDepth", "biogeoBottomDepth", "carbonTot", "nitrogenTot"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=cols[2:])
    if df.empty:
        raise ValueError(f"No usable NEON horizons for site {site}")

    horizons = (
        df.groupby(["horizonName", "biogeoTopDepth", "biogeoBottomDepth"], as_index=False)
        .agg(carbonTot=("carbonTot", "mean"), nitrogenTot=("nitrogenTot", "mean"))
        .sort_values(["biogeoTopDepth", "biogeoBottomDepth"])
    )
    valid_n = (
        df.loc[df.nitrogenTot.gt(0)]
        .assign(mid=lambda d: (d.biogeoTopDepth + d.biogeoBottomDepth) / 2.0)
        .groupby("mid", as_index=False).nitrogenTot.mean()
        .sort_values("mid")
    )

    soil_c, soil_n, c_n = [], [], []
    deepest_bottom = float(horizons.biogeoBottomDepth.max())
    below_count = 0
    for depth in depths_cm:
        match = horizons.loc[
            horizons.biogeoTopDepth.le(depth) & horizons.biogeoBottomDepth.gt(depth)
        ]
        if match.empty:
            # below the deepest reported horizon: use the deepest horizon
            match = horizons.iloc[[-1]]
            if depth >= deepest_bottom:
                below_count += 1
        if len(match) > 1:
            match = match.iloc[[0]]
        row = match.iloc[0]
        carbon = float(row.carbonTot)
        nitrogen = float(row.nitrogenTot)
        if carbon <= 0:
            raise ValueError(f"{site}: nonpositive soil carbon at depth {depth} cm")
        if nitrogen <= 0:
            if valid_n.empty:
                raise ValueError(f"{site}: cannot interpolate nitrogen (no positive values)")
            nitrogen = float(np.interp(
                depth, valid_n.mid.to_numpy(float), valid_n.nitrogenTot.to_numpy(float)
            ))
            if nitrogen <= 0:
                raise ValueError(f"{site}: nitrogen interpolation nonpositive at {depth} cm")
        soil_c.append(carbon / 10.0)
        soil_n.append(nitrogen / 10.0)
        c_n.append(carbon / nitrogen)
    return np.array(soil_c), np.array(soil_n), np.array(c_n), below_count


def load_soil_properties(
    surffile: str | Path, neon_csv: str | Path, site: str, depths_cm: np.ndarray
) -> SoilProperties:
    nlayer = len(depths_cm)
    sand, clay, native, extended = read_texture(surffile, nlayer)
    soil_c, soil_n, c_n, below_count = read_carbon_nitrogen(neon_csv, site, depths_cm)
    return SoilProperties(
        sand=sand, clay=clay, soil_c_percent=soil_c, soil_n_percent=soil_n,
        c_n=c_n, texture_layers_native=native, texture_extended=extended,
        cn_below_horizon_layers=below_count,
    )
