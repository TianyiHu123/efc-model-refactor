#!/usr/bin/env python
"""Root caller for the ELM EFC engine (one case per invocation).

Drives the ``elm`` engine end to end:

    config (.cfg)  ->  io (run-dir .nc, hourly->daily)  ->  surface (texture + C/N)
                   ->  features (trailing W/M/Y)         ->  build (Stage-1 table)
                   ->  predict (Stage-2 NetCDF)

Output location policy (README §6, AGENTS.md §1):
    {output-root}/{site}/{case-name}/{mode}/attempt{N}/

Example
-------
    run_efc.py \
        --site ABBY \
        --case-name ABBY_ctrlvertsoc \
        --config  /pscratch/.../ABBY/config/ABBY_ctrlvertsoc.cfg \
        --run-dir /pscratch/.../ABBY_ctrlvertsoc_I20TRCNPRDCTCBC/run \
        --neon-csv-root /pscratch/sd/t/tianyihu/CTSM_Hu/neon_surffiles \
        --output-root   /pscratch/sd/t/tianyihu/efc-prediction \
        --mode test --stage all
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from elm import build as _build
from elm import io as _io
from elm import plotting as _plotting
from elm import predict as _predict
from elm import surface as _surface
from elm.config import load_case_config
from elm.paths import VALID_MODES, next_attempt_dir

REPO_ROOT = Path(__file__).resolve().parent
MODEL_DIR = REPO_ROOT / "model_artifacts" / "cumulative70_bootstrap500"
DEFAULT_OUTPUT_ROOT = "/pscratch/sd/t/tianyihu/efc-prediction"


def _str2bool(value: str) -> bool:
    v = value.strip().lower().strip(".")
    if v in ("true", "t", "1", "yes", "y"):
        return True
    if v in ("false", "f", "0", "no", "n"):
        return False
    raise argparse.ArgumentTypeError(f"expected a boolean, got {value!r}")


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ELM EFC engine (one case per run).")
    p.add_argument("--site", required=True)
    p.add_argument("--case-name", required=True)
    p.add_argument("--config", required=True, type=Path, help="path to elm-olmt .cfg")
    p.add_argument("--run-dir", required=True, type=Path,
                   help="case run directory containing H2OSOI.nc/TSOI.nc/NPP.nc")
    p.add_argument("--neon-csv-root", required=True, type=Path)
    p.add_argument("--output-root", type=Path, default=Path(DEFAULT_OUTPUT_ROOT))
    p.add_argument("--output-dir", type=Path, default=None,
                   help="explicit output dir (e.g. a pre-created attempt folder); "
                        "when set, overrides the auto-selected attempt under "
                        "--output-root. Used by Slurm runs so script/logs/outputs "
                        "co-locate in one attempt folder.")
    p.add_argument("--compset", default="I20TRCNPRDCTCBC",
                   help="placeholder; reserved for later multi-case use")
    p.add_argument("--mode", choices=VALID_MODES, default="test")
    p.add_argument("--use-vertsoilc", type=_str2bool, default=True)
    p.add_argument("--stage", choices=("inputs", "predict", "all"), default="all")
    p.add_argument("--keep-bootstraps", action="store_true",
                   help="retain full 500-member bootstrap dimension in the NetCDF")
    p.add_argument("--jobs", type=int, default=1,
                   help="threads per RandomForest for Stage-2 predict (parallelism); "
                        "use the number of allocated cores on a compute node")
    p.add_argument("--no-csv", action="store_true", help="skip CSV copy of the table")
    p.add_argument("--plot", action="store_true",
                   help="after the run, render PNG figures (EFC predictions + "
                        "driver features) into {output-dir}/plots")
    p.add_argument("--plot-only", action="store_true",
                   help="skip Stage 1/2; only render figures from existing "
                        "outputs in --output-dir (requires --output-dir).")
    p.add_argument("--plot-vars", default="all",
                   help="which EFC outcomes to plot: 'all' (default) or a "
                        "comma-separated list of outcome names/slugs "
                        "(e.g. 'lignin,carbohydrate').")
    p.add_argument("--plot-dir", type=Path, default=None,
                   help="destination root for figures (default {output-dir}/plots).")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    # --- plot-only: skip Stage 1/2; render from existing products ---
    if args.plot_only:
        if args.output_dir is None:
            print("[error] --plot-only requires --output-dir pointing at an "
                  "attempt folder with existing outputs.", file=sys.stderr)
            return 2
        src_dir = args.output_dir.expanduser().resolve()
        if not src_dir.is_dir():
            print(f"[error] --output-dir not found: {src_dir}", file=sys.stderr)
            return 2
        plot_dir = (args.plot_dir.expanduser().resolve()
                    if args.plot_dir is not None else src_dir / "plots")
        print(f"[info] plot-only: reading {src_dir}; writing figures to {plot_dir}")
        summary = _plotting.generate_plots(
            out_dir=src_dir, case_name=args.case_name, model_dir=MODEL_DIR,
            plot_dir=plot_dir, plot_vars=args.plot_vars,
        )
        n = sum(len(v) for v in summary.values())
        print(f"[done] plot-only: {n} figure(s) across {len(summary)} outcome(s)")
        return 0

    cfg = load_case_config(args.config)
    # Consistency checks between CLI and cfg (warn-level; cfg wins for data,
    # CLI --case-name wins for the output label).
    if cfg.site != args.site:
        print(f"[warn] --site {args.site} != cfg sites {cfg.site}; using cfg.", file=sys.stderr)
    if cfg.case_name != args.case_name:
        print(f"[warn] --case-name {args.case_name} != cfg case_names {cfg.case_name}; "
              f"output will be labeled with --case-name {args.case_name}.", file=sys.stderr)
    if cfg.use_vertsoilc != args.use_vertsoilc:
        print(f"[warn] --use-vertsoilc {args.use_vertsoilc} != cfg {cfg.use_vertsoilc}; "
              f"using cfg value {cfg.use_vertsoilc}.", file=sys.stderr)

    if args.output_dir is not None:
        out_dir = args.output_dir.expanduser().resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        out_dir = next_attempt_dir(args.output_root, cfg.site, args.case_name, args.mode)
    print(f"[info] output dir: {out_dir}")

    neon_csv = args.neon_csv_root / f"{cfg.site}_surfaceData.csv"

    # --- Stage 1: inputs ---
    daily = _io.read_case(args.run_dir, cfg.trans_startyear, cfg.use_vertsoilc)
    soil = _surface.load_soil_properties(cfg.surffile, neon_csv, cfg.site, daily.depths_cm)
    master = _build.build_feature_table(cfg, daily, soil)

    manifest = json.loads((MODEL_DIR / "manifest.json").read_text())
    model_features = sorted({f for b in manifest["bundles"] for f in b["features"]})
    _build.validate_model_features(master, model_features)

    written = _build.write_feature_table(
        master, out_dir, args.case_name, write_csv=not args.no_csv
    )
    provenance = _build.build_provenance(
        cfg, daily, soil, args.run_dir, neon_csv, args.compset, master
    )
    prov_path = _build.write_provenance(provenance, out_dir, args.case_name)
    print(f"[info] stage-1 table rows={len(master)} layers={daily.nlayer} "
          f"dates={provenance['feature_first_date']}..{provenance['feature_last_date']}")
    print(f"[info] wrote: {written} ; provenance: {prov_path}")

    if args.stage == "inputs":
        if args.plot:
            print("[warn] --plot ignored with --stage inputs (no predictions "
                  "produced to plot).", file=sys.stderr)
        print("[done] stage=inputs")
        return 0

    # --- Stage 2: predict ---
    nc_path = _predict.predict_case(
        master, MODEL_DIR, out_dir, args.case_name, cfg.site, args.compset,
        keep_bootstraps=args.keep_bootstraps, n_jobs=args.jobs,
    )
    print(f"[info] wrote predictions: {nc_path}")

    if args.plot:
        plot_dir = (args.plot_dir.expanduser().resolve()
                    if args.plot_dir is not None else out_dir / "plots")
        print(f"[info] rendering figures to {plot_dir}")
        summary = _plotting.generate_plots(
            out_dir=out_dir, case_name=args.case_name, model_dir=MODEL_DIR,
            plot_dir=plot_dir, plot_vars=args.plot_vars,
        )
        n = sum(len(v) for v in summary.values())
        print(f"[info] plots: {n} figure(s) across {len(summary)} outcome(s)")

    print("[done] stage=all" if args.stage == "all" else "[done] stage=predict")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
