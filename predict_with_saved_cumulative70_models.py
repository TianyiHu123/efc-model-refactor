"""Apply saved cumulative-70% bootstrap models to a site's model-input CSVs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model_artifacts" / "cumulative70_bootstrap500"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("sites", nargs="+", help="Site code(s), e.g. RMNP TALL")
    parser.add_argument("--input-root", type=Path, default=ROOT / "model_inputs")
    parser.add_argument("--output-root", type=Path, default=ROOT / "model_predictions")
    return parser.parse_args()


def slug(value: str) -> str:
    return value.lower().replace(" ", "_")


def main() -> None:
    args = parse_args()
    manifest = json.loads((MODEL_DIR / "manifest.json").read_text())
    for requested_site in args.sites:
        site = requested_site.upper()
        input_dir = args.input_root / f"{site}_cumulative70"
        output_dir = args.output_root / f"{site}_cumulative70_bootstrap500"
        output_dir.mkdir(parents=True, exist_ok=True)
        outputs = {}
        for outcome in manifest["outcomes"]:
            prefix = slug(outcome)
            bundle = joblib.load(MODEL_DIR / f"cumulative70_{prefix}_bootstrap500.joblib")
            path = input_dir / f"{site}_{prefix}_cumulative70_model_input.csv"
            table = pd.read_csv(path)
            features = bundle["features"]
            for layer in range(1, 6):
                layer_table = table.loc[table.soil_layer.eq(layer)].sort_values("Date").reset_index(drop=True)
                if layer not in outputs:
                    outputs[layer] = layer_table[["site", "Date", "soil_layer", "soil_node_depth_cm"]].copy()
                values = np.column_stack([model.predict(layer_table[features]) for model in bundle["models"]])
                raw = pd.DataFrame(values, columns=[f"{prefix}_bootstrap_{i:03d}" for i in range(1, 501)])
                stats = pd.DataFrame({
                    f"{prefix}_mean": values.mean(axis=1),
                    f"{prefix}_median": np.median(values, axis=1),
                    f"{prefix}_sd": values.std(axis=1, ddof=1),
                    f"{prefix}_q025": np.quantile(values, 0.025, axis=1),
                    f"{prefix}_q975": np.quantile(values, 0.975, axis=1),
                })
                outputs[layer] = pd.concat([outputs[layer], raw, stats], axis=1)
            del bundle
        for layer, output in outputs.items():
            output.to_csv(
                output_dir / f"{site}_soil_layer_{layer}_EFC_bootstrap500_predictions.csv",
                index=False, float_format="%.10g",
            )
            print(site, layer, len(output), len(output.columns))


if __name__ == "__main__":
    main()
