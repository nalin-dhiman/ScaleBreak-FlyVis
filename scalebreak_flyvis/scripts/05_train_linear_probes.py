#!/usr/bin/env python
"""Train linear probes and scale-generalization matrices."""

from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd

from scalebreak.baselines import nuisance_features
from scalebreak.probes import cap_features_by_variance, protocol_metrics, scale_generalization_matrix
from scalebreak.utils import copy_config, ensure_subdir, load_yaml, run_info, setup_logging, write_json


def run_one(x, meta, out_dir: Path, seed: int, targets: list[str], max_features: int | None = None) -> None:
    x = cap_features_by_variance(x, max_features=max_features)
    metrics, reports, cms = protocol_metrics(x, meta, targets=targets, seed=seed)
    metrics.to_csv(out_dir / "metrics.csv", index=False)
    scale_generalization_matrix(x, meta, target="shape", seed=seed).to_csv(out_dir / "scale_generalization_matrix.csv", index=False)
    write_json(reports, out_dir / "classification_report.json")
    np.savez(out_dir / "confusion_matrices.npz", **cms)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="scalebreak_flyvis/configs/analysis.yaml")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    seed = int(cfg.get("seed", 42))
    targets = cfg.get("probes", {}).get("targets", ["shape", "scale", "motion_type", "contrast"])
    max_features = cfg.get("probes", {}).get("max_features", 2048)
    feature_root = Path(cfg["paths"].get("features_dir", "scalebreak_flyvis/outputs/features"))
    out_root = ensure_subdir(cfg["paths"].get("probes_dir", "scalebreak_flyvis/outputs/probes"))
    logger = setup_logging(out_root, "05_train_linear_probes")
    for model_dir in [p for p in feature_root.iterdir() if p.is_dir()]:
        for feat_dir in [p for p in model_dir.iterdir() if p.is_dir() and (p / "X.npy").exists()]:
            out_dir = ensure_subdir(out_root / model_dir.name / feat_dir.name)
            copy_config(args.config, out_dir)
            if args.dry_run:
                logger.info("Would probe %s/%s", model_dir.name, feat_dir.name)
                continue
            x = np.load(feat_dir / "X.npy")
            meta = pd.read_csv(feat_dir / "metadata.csv")
            run_one(x, meta, out_dir, seed, targets, max_features=max_features)
            logger.info("Probed %s/%s", model_dir.name, feat_dir.name)

    stim_meta = Path(cfg["paths"]["stimuli_dir"]) / "metadata.csv"
    if stim_meta.exists():
        meta = pd.read_csv(stim_meta)
        for kind in ["area", "edge", "area_contrast", "area_edge_contrast"]:
            out_dir = ensure_subdir(out_root / f"nuisance_{kind}" / "raw")
            x = nuisance_features(meta, kind)
            run_one(x, meta, out_dir, seed, targets, max_features=max_features)
    write_json(run_info("05_train_linear_probes", seed=seed), out_root / "run_info.json")


if __name__ == "__main__":
    main()
