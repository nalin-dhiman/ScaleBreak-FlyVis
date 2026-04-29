#!/usr/bin/env python
"""Estimate scale breakdown points from decoding curves."""

from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scalebreak.breakpoints import accuracy_by_scale, bootstrap_breakpoints
from scalebreak.utils import copy_config, ensure_subdir, load_yaml, run_info, setup_logging, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="scalebreak_flyvis/configs/analysis.yaml")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    seed = int(cfg.get("seed", 42))
    n_boot = int(cfg.get("breakpoints", {}).get("n_bootstrap", 100))
    selected_features = set(cfg.get("breakpoints", {}).get("feature_types", []))
    feature_root = Path(cfg["paths"].get("features_dir", "scalebreak_flyvis/outputs/features"))
    out_root = ensure_subdir(cfg["paths"].get("breakpoints_dir", "scalebreak_flyvis/outputs/breakpoints"))
    logger = setup_logging(out_root, "07_estimate_breakpoints")
    for model_dir in [p for p in feature_root.iterdir() if p.is_dir()]:
        for feat_dir in [p for p in model_dir.iterdir() if p.is_dir() and (p / "X.npy").exists()]:
            if selected_features and feat_dir.name not in selected_features:
                continue
            out_dir = ensure_subdir(out_root / model_dir.name / feat_dir.name)
            copy_config(args.config, out_dir)
            if args.dry_run:
                continue
            x = np.load(feat_dir / "X.npy")
            meta = pd.read_csv(feat_dir / "metadata.csv")
            bps, curves = bootstrap_breakpoints(x, meta, seed=seed, n_boot=n_boot)
            main_curve = accuracy_by_scale(x, meta, seed=seed)
            bps.to_csv(out_dir / "breakpoints.csv", index=False)
            curves.to_csv(out_dir / "bootstrap_curves.csv", index=False)
            main_curve.to_csv(out_dir / "accuracy_vs_scale.csv", index=False)
            plt.figure(figsize=(5, 4))
            plt.errorbar(main_curve["scale"], main_curve["accuracy"], yerr=main_curve["accuracy_std"], marker="o")
            plt.xlabel("Apparent scale proxy (pixels)")
            plt.ylabel("Shape decoding accuracy")
            plt.tight_layout()
            plt.savefig(out_dir / "fig_accuracy_vs_scale_by_shape.png", dpi=160)
            plt.savefig(out_dir / "fig_information_vs_scale.png", dpi=160)
            plt.close()
            plt.figure(figsize=(5, 3))
            pd.Series(bps["breakpoint_scale"]).dropna().hist()
            plt.xlabel("Bootstrap breakpoint scale")
            plt.tight_layout()
            plt.savefig(out_dir / "fig_breakpoint_by_shape_model.png", dpi=160)
            plt.close()
            logger.info("Breakpoints %s/%s", model_dir.name, feat_dir.name)
    write_json(run_info("07_estimate_breakpoints", seed=seed), out_root / "run_info.json")


if __name__ == "__main__":
    main()
