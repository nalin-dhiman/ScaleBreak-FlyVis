#!/usr/bin/env python
"""Run nuisance and graph-control summaries."""

from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from scalebreak.baselines import random_sparse_type_edges, shuffled_type_labels
from scalebreak.utils import copy_config, ensure_subdir, load_yaml, run_info, setup_logging, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="scalebreak_flyvis/configs/analysis.yaml")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    out_dir = ensure_subdir(cfg["paths"].get("controls_dir", "scalebreak_flyvis/outputs/controls"))
    logger = setup_logging(out_dir, "08_run_controls")
    copy_config(args.config, out_dir)
    rows = []
    probe_root = Path(cfg["paths"].get("probes_dir", "scalebreak_flyvis/outputs/probes"))
    if probe_root.exists():
        for metrics_path in probe_root.glob("*/*/metrics.csv"):
            metrics = pd.read_csv(metrics_path)
            metrics["model"] = metrics_path.parents[1].name
            metrics["feature_type"] = metrics_path.parent.name
            rows.append(metrics)
    summary = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    summary.to_csv(out_dir / "control_summary.csv", index=False)

    graph_summaries = {}
    type_edges_path = Path(cfg["paths"].get("type_edges", ""))
    if type_edges_path.exists():
        type_edges = pd.read_parquet(type_edges_path)
        random_edges = random_sparse_type_edges(type_edges, seed=int(cfg.get("seed", 42)))
        shuffled_edges = shuffled_type_labels(type_edges, seed=int(cfg.get("seed", 42)))
        random_edges.to_parquet(out_dir / "random_sparse_type_edges.parquet", index=False)
        shuffled_edges.to_parquet(out_dir / "shuffled_type_edges.parquet", index=False)
        graph_summaries = {
            "real_type_edges": int(len(type_edges)),
            "random_sparse_type_edges": int(len(random_edges)),
            "shuffled_type_edges": int(len(shuffled_edges)),
            "note": "Controls preserve edge count; random sparse also reuses the empirical weight distribution.",
        }
    write_json(graph_summaries, out_dir / "control_graph_summaries.json")
    if not summary.empty:
        shape = summary[summary["target"] == "shape"].sort_values("accuracy", ascending=False).head(30)
        plt.figure(figsize=(8, 5))
        labels = shape["model"] + "/" + shape["feature_type"]
        plt.barh(labels[::-1], shape["accuracy"].iloc[::-1])
        plt.xlabel("Shape decoding accuracy")
        plt.tight_layout()
        plt.savefig(out_dir / "fig_control_comparison.png", dpi=180)
        plt.close()
    write_json(run_info("08_run_controls", seed=int(cfg.get("seed", 42))), out_dir / "run_info.json")
    logger.info("Controls summarized with %d metric rows", len(summary))


if __name__ == "__main__":
    main()
