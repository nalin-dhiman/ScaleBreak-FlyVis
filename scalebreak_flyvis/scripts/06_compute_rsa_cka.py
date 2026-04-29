#!/usr/bin/env python
"""Compute RSA and linear CKA over scale-specific representations."""

from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd

from scalebreak.plotting import save_heatmap
from scalebreak.rsa_cka import cka_by_scale, rsa_summary
from scalebreak.utils import copy_config, ensure_subdir, load_yaml, run_info, setup_logging, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="scalebreak_flyvis/configs/analysis.yaml")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    feature_root = Path(cfg["paths"].get("features_dir", "scalebreak_flyvis/outputs/features"))
    out_root = ensure_subdir(cfg["paths"].get("rsa_cka_dir", "scalebreak_flyvis/outputs/rsa_cka"))
    logger = setup_logging(out_root, "06_compute_rsa_cka")
    for model_dir in [p for p in feature_root.iterdir() if p.is_dir()]:
        for feat_dir in [p for p in model_dir.iterdir() if p.is_dir() and (p / "X.npy").exists()]:
            out_dir = ensure_subdir(out_root / model_dir.name / feat_dir.name)
            copy_config(args.config, out_dir)
            if args.dry_run:
                continue
            x = np.load(feat_dir / "X.npy")
            meta = pd.read_csv(feat_dir / "metadata.csv")
            rsa, summary = rsa_summary(x, meta)
            cka = cka_by_scale(x, meta)
            rsa.to_csv(out_dir / "rsa_matrix.csv")
            cka.to_csv(out_dir / "cka_matrix.csv", index=False)
            summary.to_csv(out_dir / "similarity_summary.csv", index=False)
            save_heatmap(rsa, out_dir / f"fig_rsa_heatmap_{model_dir.name}_{feat_dir.name}.png", "RSA cosine")
            pivot = cka.pivot(index="scale_a", columns="scale_b", values="cka")
            save_heatmap(pivot, out_dir / f"fig_cka_scale_matrix_{model_dir.name}_{feat_dir.name}.png", "Linear CKA")
            logger.info("RSA/CKA %s/%s", model_dir.name, feat_dir.name)
    write_json(run_info("06_compute_rsa_cka", seed=int(cfg.get("seed", 42))), out_root / "run_info.json")


if __name__ == "__main__":
    main()
