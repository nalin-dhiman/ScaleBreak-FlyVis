#!/usr/bin/env python
"""Audit local optic-lobe neuPrint export schemas without hardcoded assumptions."""

from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
import pandas as pd

from scalebreak.connectome import recommend_mapping
from scalebreak.io import discover_tables, infer_column_candidates, read_table_sample, table_basic_info
from scalebreak.utils import copy_config, ensure_subdir, load_yaml, run_info, setup_logging, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("--data-dir")
    parser.add_argument("--out-dir", default="scalebreak_flyvis/outputs/audits")
    parser.add_argument("--sample-rows", type=int, default=5)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    cfg = load_yaml(args.config) if args.config else {}
    data_dir = Path(args.data_dir or cfg.get("data_dir") or cfg.get("paths", {}).get("connectome_data_dir"))
    out_dir = ensure_subdir(args.out_dir or cfg.get("audit_out_dir", "scalebreak_flyvis/outputs/audits"))
    logger = setup_logging(out_dir, "00_audit_connectome_tables")
    copy_config(args.config, out_dir, cfg if cfg else {"data_dir": str(data_dir)})

    report = {"data_dir": str(data_dir), "tables": {}}
    inventory = []
    md_lines = [f"# neuPrint Table Schema Report\n\nData directory: `{data_dir}`\n"]
    for path in discover_tables(data_dir):
        logger.info("Auditing %s", path.name)
        try:
            info = table_basic_info(path, sample_rows=args.sample_rows)
            sample = read_table_sample(path, args.sample_rows)
            sample.to_csv(out_dir / f"head_{path.name.replace(':', '_')}.csv", index=False)
            candidates = infer_column_candidates(info["columns"])
            info["candidates"] = candidates
            report["tables"][path.name] = info
            inventory.append(
                {"name": path.name, "suffix": path.suffix, "n_rows": info.get("n_rows"), "n_cols": info.get("n_cols"), "path": str(path)}
            )
            md_lines.append(f"\n## {path.name}\n")
            md_lines.append(f"- shape: ({info.get('n_rows')}, {info.get('n_cols')})\n")
            md_lines.append(f"- columns: `{', '.join(info['columns'])}`\n")
            md_lines.append("- candidate mappings:\n")
            for key, vals in candidates.items():
                md_lines.append(f"  - {key}: {vals[:10]}\n")
            md_lines.append("\nFirst rows are saved as per-table CSV files.\n")
        except Exception as exc:
            logger.exception("Failed auditing %s", path)
            report["tables"][path.name] = {"error": repr(exc)}
            inventory.append({"name": path.name, "suffix": path.suffix, "error": repr(exc), "path": str(path)})

    report["recommended_mapping"] = recommend_mapping(report)
    md_lines.append("\n## Recommended Mapping\n\n")
    for key, value in report["recommended_mapping"].items():
        md_lines.append(f"- {key}: `{value}`\n")

    pd.DataFrame(inventory).to_csv(out_dir / "table_inventory.csv", index=False)
    (out_dir / "schema_report.md").write_text("".join(md_lines), encoding="utf-8")
    write_json(report, out_dir / "schema_report.json")
    write_json(run_info("00_audit_connectome_tables"), out_dir / "run_info.json")
    logger.info("Recommended mapping: %s", report["recommended_mapping"])


if __name__ == "__main__":
    main()
