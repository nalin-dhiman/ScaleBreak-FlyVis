#!/usr/bin/env python
"""Assemble final figures, tables, and REPORT.md."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import _bootstrap  # noqa: F401
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from scalebreak.stats import activity_scale_effects
from scalebreak.models import flyvis_available
from scalebreak.utils import copy_config, ensure_subdir, load_yaml, run_info, setup_logging, write_json


def copy_first(pattern: str, dest: Path) -> bool:
    matches = list(Path("scalebreak_flyvis/outputs").glob(pattern))
    if matches:
        shutil.copy2(matches[0], dest)
        return True
    return False


def conceptual_geometry(path: Path) -> None:
    scales = [2, 4, 8, 16, 24]
    plt.figure(figsize=(7, 4))
    for i, s in enumerate(scales):
        circle = plt.Circle((i, 0), s / 30, fill=False, lw=2)
        plt.gca().add_patch(circle)
        plt.text(i, -0.55, f"scale {s}", ha="center")
    plt.xlim(-0.8, len(scales) - 0.2)
    plt.ylim(-0.8, 1.0)
    plt.axis("off")
    plt.title("Retinal projection varies apparent scale; contrast, blur, motion, and position are controlled separately")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_activity_metric(pattern: str, dest: Path) -> bool:
    matches = list(Path("scalebreak_flyvis/outputs").glob(pattern))
    if not matches:
        return False
    rows = []
    for p in matches:
        df = pd.read_csv(p)
        df["model"] = p.parent.name
        rows.append(df)
    data = pd.concat(rows, ignore_index=True)
    if "mean_abs_activity" not in data:
        return False
    summary = data.groupby(["model", "scale"], as_index=False)["mean_abs_activity"].mean()
    plt.figure(figsize=(7, 4))
    for model, sub in summary.groupby("model"):
        plt.plot(sub["scale"], sub["mean_abs_activity"], marker="o", label=model)
    plt.xlabel("Apparent scale proxy (pixels)")
    plt.ylabel("Mean absolute activity proxy")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(dest, dpi=180)
    plt.close()
    return True


def plot_scale_generalization(pattern: str, dest: Path) -> bool:
    matches = list(Path("scalebreak_flyvis/outputs").glob(pattern))
    if not matches:
        return False
    df = pd.read_csv(matches[0])
    pivot = df.pivot(index="train_scale", columns="test_scale", values="accuracy")
    plt.figure(figsize=(5, 4))
    plt.imshow(pivot.values, vmin=0, vmax=1, cmap="viridis")
    plt.colorbar(label="Accuracy")
    plt.xticks(range(len(pivot.columns)), pivot.columns)
    plt.yticks(range(len(pivot.index)), pivot.index)
    plt.xlabel("Test scale")
    plt.ylabel("Train scale")
    plt.tight_layout()
    plt.savefig(dest, dpi=180)
    plt.close()
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="scalebreak_flyvis/configs/analysis.yaml")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    fig_dir = ensure_subdir(cfg["paths"].get("figures_dir", "scalebreak_flyvis/outputs/figures"))
    table_dir = ensure_subdir(cfg["paths"].get("tables_dir", "scalebreak_flyvis/outputs/tables"))
    logger = setup_logging(fig_dir, "09_make_figures")
    copy_config(args.config, fig_dir)
    conceptual_geometry(fig_dir / "fig01_conceptual_geometry.png")
    copy_first("stimuli/pilot/montage.png", fig_dir / "fig02_stimulus_grid.png")
    effect_rows = []
    for p in Path("scalebreak_flyvis/outputs").glob("features/*/activity_metrics.csv"):
        df = pd.read_csv(p)
        effects = activity_scale_effects(df)
        effects["model"] = p.parent.name
        effect_rows.append(effects)
    (pd.concat(effect_rows, ignore_index=True) if effect_rows else pd.DataFrame()).to_csv(
        table_dir / "table_activity_scale_effects.csv", index=False
    )
    copy_first("connectome/fig_type_graph_adjacency.png", fig_dir / "fig07_connectome_graph_summary.png")
    copy_first("controls/fig_control_comparison.png", fig_dir / "fig08_controls_summary.png")

    probe_rows = []
    for p in Path(cfg["paths"].get("probes_dir", "scalebreak_flyvis/outputs/probes")).glob("*/*/metrics.csv"):
        df = pd.read_csv(p)
        df["model"] = p.parents[1].name
        df["feature_type"] = p.parent.name
        probe_rows.append(df)
    model_comp = pd.concat(probe_rows, ignore_index=True) if probe_rows else pd.DataFrame()
    model_comp.to_csv(table_dir / "table_model_comparison.csv", index=False)

    bp_rows = []
    for p in Path(cfg["paths"].get("breakpoints_dir", "scalebreak_flyvis/outputs/breakpoints")).glob("*/*/breakpoints.csv"):
        df = pd.read_csv(p)
        df["model"] = p.parents[1].name
        df["feature_type"] = p.parent.name
        bp_rows.append(df)
    bps = pd.concat(bp_rows, ignore_index=True) if bp_rows else pd.DataFrame()
    bps.to_csv(table_dir / "table_breakpoints.csv", index=False)
    control = Path(cfg["paths"].get("controls_dir", "scalebreak_flyvis/outputs/controls")) / "control_summary.csv"
    if control.exists():
        shutil.copy2(control, table_dir / "table_control_results.csv")
    else:
        pd.DataFrame().to_csv(table_dir / "table_control_results.csv", index=False)

    plot_activity_metric("features/*/activity_metrics.csv", fig_dir / "fig03_activity_vs_scale.png")
    plot_scale_generalization("probes/*/*/scale_generalization_matrix.csv", fig_dir / "fig04_scale_generalization_matrices.png")
    copy_first("rsa_cka/*/*/fig_cka_scale_matrix_*.png", fig_dir / "fig05_rsa_cka.png")
    copy_first("breakpoints/*/*/fig_accuracy_vs_scale_by_shape.png", fig_dir / "fig06_breakpoint_curves.png")

    fly_ok, fly_msg = flyvis_available()
    report = [
        "# ScaleBreak-FlyVis Pilot Report\n",
        "## Framing\n",
        "This pipeline studies retinal apparent-scale representation, not physical distance or generic object recognition.\n",
        "Activity outputs from local baselines are activity/rate proxies.\n",
        "## What ran\n",
        f"- Config: `{args.config}`\n",
        f"- FlyVis availability: `{fly_ok}` ({fly_msg})\n",
        f"- Model-comparison rows: {len(model_comp)}\n",
        f"- Breakpoint bootstrap rows: {len(bps)}\n",
        "## Critical interpretation\n",
        "A pilot is scientifically interpretable only after matched nuisance controls are compared against the optic-lobe structural-prior and, if installed, FlyVis representations. Pixel area or edge-length success alone is not evidence for scale-stable shape representation.\n",
        "## Next steps\n",
        "- Install optional `zarr` for chunked stores if large full runs are planned.\n",
        "- Install and validate FlyVis APIs before treating FlyVis-specific results as available.\n",
        "- Run the full graph build without `--max-edges` only when sufficient memory and runtime are available.\n",
    ]
    Path("scalebreak_flyvis/outputs/REPORT.md").write_text("\n".join(report), encoding="utf-8")
    write_json(run_info("09_make_figures", seed=int(cfg.get("seed", 42))), fig_dir / "run_info.json")
    logger.info("Wrote report and summary tables")


if __name__ == "__main__":
    main()
