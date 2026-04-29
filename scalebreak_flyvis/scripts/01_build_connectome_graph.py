#!/usr/bin/env python
"""Build neuron-level and type-level optic-lobe structural-prior graphs."""

from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

from scalebreak.connectome import build_graph_tables, plot_graph_diagnostics
from scalebreak.utils import copy_config, ensure_subdir, load_yaml, read_json, run_info, setup_logging, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("--data-dir")
    parser.add_argument("--audit-json")
    parser.add_argument("--out-dir", default="scalebreak_flyvis/outputs/connectome")
    parser.add_argument("--max-edges", type=int)
    parser.add_argument("--max-nodes", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    cfg = load_yaml(args.config) if args.config else {}
    out_dir = ensure_subdir(args.out_dir or cfg.get("connectome_out_dir", "scalebreak_flyvis/outputs/connectome"))
    logger = setup_logging(out_dir, "01_build_connectome_graph")
    copy_config(args.config, out_dir, cfg if cfg else {})
    data_dir = args.data_dir or cfg.get("paths", {}).get("connectome_data_dir")
    audit_json = args.audit_json or cfg.get("paths", {}).get("audit_json", "scalebreak_flyvis/outputs/audits/schema_report.json")
    if args.dry_run:
        logger.info("Dry run: would build graph from %s using %s", data_dir, audit_json)
        return
    schema = read_json(audit_json)
    nodes, edges, type_nodes, type_edges, neuron_summary, type_summary = build_graph_tables(
        data_dir, schema, max_edges=args.max_edges or cfg.get("connectome", {}).get("max_edges"), max_nodes=args.max_nodes
    )
    nodes.to_parquet(out_dir / "neuron_nodes.parquet", index=False)
    edges.to_parquet(out_dir / "neuron_edges.parquet", index=False)
    type_nodes.to_parquet(out_dir / "type_nodes.parquet", index=False)
    type_edges.to_parquet(out_dir / "type_edges.parquet", index=False)
    type_edges.nlargest(100, "total_weight").to_csv(out_dir / "top_type_edges.csv", index=False)
    write_json(neuron_summary, out_dir / "neuron_graph_summary.json")
    write_json(type_summary, out_dir / "type_graph_summary.json")
    plot_graph_diagnostics(nodes, edges, type_edges, out_dir)
    write_json(run_info("01_build_connectome_graph", extra={"mapping": schema.get("recommended_mapping")}), out_dir / "run_info.json")
    logger.info("Wrote %d neuron nodes, %d neuron edges, %d type edges", len(nodes), len(edges), len(type_edges))


if __name__ == "__main__":
    main()
