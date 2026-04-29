#!/usr/bin/env python
"""Extract feature matrices and activity metrics from model activations."""

from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd

from scalebreak.features import activity_metrics, make_feature_matrices, normalize_features
from scalebreak.io import load_array_store
from scalebreak.utils import copy_config, ensure_subdir, load_yaml, run_info, setup_logging, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="scalebreak_flyvis/configs/analysis.yaml")
    parser.add_argument("--models")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    act_root = Path(cfg["paths"].get("activations_dir", "scalebreak_flyvis/outputs/activations"))
    out_root = ensure_subdir(cfg["paths"].get("features_dir", "scalebreak_flyvis/outputs/features"))
    logger = setup_logging(out_root, "04_extract_features")
    models = [m.strip() for m in args.models.split(",")] if args.models else [p.name for p in act_root.iterdir() if p.is_dir()]
    for model in models:
        act_dir = act_root / model
        if not (act_dir / "metadata.csv").exists():
            continue
        out_model = ensure_subdir(out_root / model)
        copy_config(args.config, out_model)
        if args.dry_run:
            logger.info("Would extract %s", model)
            continue
        activations = load_array_store(act_dir / "activations.zarr", "activations")
        meta = pd.read_csv(act_dir / "metadata.csv")
        metrics = activity_metrics(activations)
        pd.concat([meta, metrics], axis=1).to_csv(out_model / "activity_metrics.csv", index=False)
        matrices = make_feature_matrices(activations, bins=int(cfg.get("features", {}).get("temporal_bins", 5)))
        count = 0
        for base_name, x in matrices.items():
            for norm_name, xn in normalize_features(x).items():
                feature_name = f"{base_name}_{norm_name}"
                feature_dir = ensure_subdir(out_model / feature_name)
                np.save(feature_dir / f"X_{base_name}.npy", xn)
                np.save(feature_dir / "X.npy", xn)
                meta.to_csv(feature_dir / "metadata.csv", index=False)
                write_json({"model": model, "base_feature": base_name, "normalization": norm_name, "shape": list(xn.shape)}, feature_dir / "feature_info.json")
                count += 1
        write_json(run_info("04_extract_features", seed=int(cfg.get("seed", 42)), extra={"model": model, "n_feature_sets": count}), out_model / "run_info.json")
        logger.info("Extracted %d feature sets for %s", count, model)


if __name__ == "__main__":
    main()
