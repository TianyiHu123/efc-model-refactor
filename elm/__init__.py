"""ELM EFC engine.

Reads elm-olmt run-folder NetCDF output and site metadata, engineers the
trailing hydrothermal features expected by the pre-trained EFC bootstrap
ensemble, and writes per-case predictions as NetCDF.

Driven by the root caller ``run_efc.py``. See the repository README and
AGENTS.md for conventions (run-location policy, environment, data source).
"""

from __future__ import annotations

__all__ = [
    "config",
    "paths",
    "io",
    "surface",
    "features",
    "build",
    "predict",
]
