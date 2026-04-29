#!/usr/bin/env python
"""Finalize claim language and consolidate manuscript figures/tables."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


CLAIMS = {
    "robust_dynamic_scale_generalization_in_flyvis": True,
    "evidence_for_flyvis_model_specificity": True,
    "evidence_for_exact_connectome_necessity": "not established",
    "physical_distance_claim": False,
    "generic_object_recognition_claim": False,
    "static_shape_claim_strength": "secondary / weak-control",
    "dynamic_direction_claim_strength": "strong",
    "connectome_causality_claim_strength": "provisional / not causal",
}


def write_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)


def markdown_table(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown(index=False)
    except Exception:
        return "```\n" + df.to_csv(index=False) + "```\n"


def save_table(df: pd.DataFrame, csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    csv_path.with_suffix(".md").write_text(markdown_table(df), encoding="utf-8")


def copy_if_exists(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def save_fig(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    for ext in ["png", "pdf"]:
        fig.savefig(path.with_suffix(f".{ext}"), dpi=180, bbox_inches="tight")


def concept_figure(path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Rectangle

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    ax = axes[0]
    ax.set_title("Retinal apparent scale proxy")
    eye = Circle((0.1, 0.5), 0.08, fill=False, lw=2)
    ax.add_patch(eye)
    for x, r in [(0.35, 0.09), (0.6, 0.055), (0.82, 0.035)]:
        ax.add_patch(Circle((x, 0.5), r, fill=False, lw=2))
        ax.plot([0.1, x], [0.5, 0.5 + r], color="0.4", lw=1)
        ax.plot([0.1, x], [0.5, 0.5 - r], color="0.4", lw=1)
    ax.text(0.1, 0.32, "model input:\nretinal projection", ha="center", fontsize=9)
    ax.text(0.63, 0.78, "not a physical-distance readout", ha="center", fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0.2, 0.85)
    ax.axis("off")

    ax = axes[1]
    ax.set_title("Benchmark variables")
    labels = ["scale", "contrast", "blur", "motion", "retinal position"]
    for i, label in enumerate(labels):
        y = 0.85 - i * 0.16
        ax.add_patch(Rectangle((0.12, y - 0.045), 0.2 + 0.08 * (i % 3), 0.09, fill=False, lw=1.5))
        ax.text(0.5, y, label, va="center", fontsize=10)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.suptitle("ScaleBreak-FlyVis: apparent-scale representations", y=1.02)
    save_fig(fig, path)
    plt.close(fig)


def make_controls_figure(path: Path, v4_tables: Path, strong_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = []
    main = pd.read_csv(v4_tables / "table_v4_main_results.csv")
    rows.extend(main[["model", "offdiag_accuracy"]].rename(columns={"offdiag_accuracy": "accuracy"}).to_dict("records"))
    destructive = pd.read_csv(v4_tables / "destructive_representation_controls.csv")
    rows.extend(destructive[["model", "offdiag_accuracy"]].rename(columns={"offdiag_accuracy": "accuracy"}).to_dict("records"))
    strong_path = strong_dir / "table_strong_vision_controls_summary.csv"
    if strong_path.exists():
        strong = pd.read_csv(strong_path)
        rows.extend(strong.rename(columns={"mean_offdiag_accuracy": "accuracy"})[["model", "accuracy", "ci_low", "ci_high"]].to_dict("records"))
    df = pd.DataFrame(rows).drop_duplicates("model", keep="last")
    order = df.sort_values("accuracy")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(order["model"], order["accuracy"])
    ax.axvline(0.9236111111111112, color="black", linestyle="--", label="FlyVis")
    ax.set_xlabel("Off-diagonal direction accuracy")
    ax.legend()
    save_fig(fig, path)
    plt.close(fig)


def make_feature_family_figure(path: Path, table_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = pd.read_csv(table_path)
    plot = df.set_index("feature_family_task")
    fig, ax = plt.subplots(figsize=(10, 4.8))
    im = ax.imshow(plot.values, aspect="auto", vmin=0, vmax=1, cmap="viridis")
    ax.set_xticks(np.arange(len(plot.columns)), plot.columns, rotation=90, fontsize=7)
    ax.set_yticks(np.arange(len(plot.index)), plot.index, fontsize=8)
    fig.colorbar(im, ax=ax, label="Accuracy")
    save_fig(fig, path)
    plt.close(fig)


def make_ablation_figure(path: Path, table_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = pd.read_csv(table_path).sort_values("drop_accuracy")
    fig, ax = plt.subplots(figsize=(8, 4.8))
    xerr = np.vstack([df["drop_accuracy"] - df["drop_ci_low"], df["drop_ci_high"] - df["drop_accuracy"]])
    ax.barh(df["ablation"], df["drop_accuracy"], xerr=xerr)
    ax.set_xlabel("Accuracy drop vs full FlyVis")
    save_fig(fig, path)
    plt.close(fig)


def make_bits_figure(path: Path, table_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = pd.read_csv(table_path)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(df["model"], df["retained_direction_bits_per_activity"])
    ax.set_ylabel("Retained direction bits per activity proxy")
    ax.tick_params(axis="x", rotation=90)
    save_fig(fig, path)
    plt.close(fig)


def consolidated_main_table(v4_tables: Path, strong_dir: Path) -> pd.DataFrame:
    main = pd.read_csv(v4_tables / "table_v4_main_results.csv")
    strong_path = strong_dir / "table_strong_vision_controls_summary.csv"
    if strong_path.exists():
        strong = pd.read_csv(strong_path)
        add = strong.rename(columns={"mean_offdiag_accuracy": "offdiag_accuracy"})[["model", "offdiag_accuracy"]]
        add["n"] = np.nan
        main = pd.concat([main, add], ignore_index=True)
    return main.drop_duplicates("model", keep="last").sort_values("offdiag_accuracy", ascending=False)


def make_flyvis_scale_figure(path: Path, v2_tables: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = pd.read_csv(v2_tables / "direction_scale_generalization.csv")
    pivot = df.pivot_table(index="train_scale", columns="test_scale", values="accuracy", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    im = ax.imshow(pivot.values, vmin=0, vmax=1, cmap="viridis")
    ax.set_xticks(np.arange(len(pivot.columns)), pivot.columns)
    ax.set_yticks(np.arange(len(pivot.index)), pivot.index)
    ax.set_xlabel("Test apparent scale")
    ax.set_ylabel("Train apparent scale")
    ax.set_title("FlyVis direction scale-generalization")
    fig.colorbar(im, ax=ax, label="Accuracy")
    save_fig(fig, path)
    plt.close(fig)


def patch_claims_text(text: str) -> str:
    return text.replace(
        "evidence_detailed_connectome_model_structure_necessary: true",
        "evidence_for_flyvis_model_specificity: true\n"
        "evidence_for_exact_connectome_necessity: not established",
    ).replace(
        "Call: `True`.\nThis compares FlyVis against pixel, CNN, local RNN, real optic-lobe type-rate, and graph-shuffled type-rate controls with trained linear readouts.",
        "Call: `not established` for exact connectome necessity.\nThe completed analyses support FlyVis model-specificity relative to tested controls, but do not establish exact connectome causality.",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs-dir", default="scalebreak_flyvis/outputs")
    parser.add_argument("--out-dir", default="scalebreak_flyvis/outputs/final_hardening")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    t0 = time.time()
    outputs = Path(args.outputs_dir)
    out = Path(args.out_dir)
    final_fig = out / "final_figures"
    final_tab = out / "final_tables"
    final_fig.mkdir(parents=True, exist_ok=True)
    final_tab.mkdir(parents=True, exist_ok=True)
    v2 = outputs / "flyvis_pilot_v2"
    v4 = outputs / "flyvis_pilot_v4"
    strong = out / "strong_controls"
    temporal = out / "temporal_lesions"

    write_json(CLAIMS, out / "final_claims.json")
    claims_md = ["# Final Claim Status", ""]
    for k, v in CLAIMS.items():
        claims_md.append(f"- `{k}`: `{v}`")
    claims_md.append("\nThese claims refer to retinal apparent scale and retinal projection, not physical distance.")
    (out / "final_claims.md").write_text("\n".join(claims_md), encoding="utf-8")

    patched_dir = out / "patched_reports"
    patched_dir.mkdir(parents=True, exist_ok=True)
    for src in [v2 / "REPORT.md", outputs / "flyvis_pilot_v3" / "REPORT.md", v4 / "REPORT.md"]:
        if src.exists():
            dst = patched_dir / f"{src.parent.name}_REPORT_patched.md"
            if dst.exists() and not args.overwrite:
                continue
            dst.write_text(patch_claims_text(src.read_text(encoding="utf-8")), encoding="utf-8")

    concept_figure(final_fig / "fig1_concept_scale_retinal_projection.png")
    make_flyvis_scale_figure(final_fig / "fig2_flyvis_direction_scale_generalization.png", v2 / "tables")
    make_controls_figure(final_fig / "fig3_controls_hardening.png", v4 / "tables", strong)
    make_feature_family_figure(final_fig / "fig4_feature_family_breakdown.png", v4 / "tables" / "table_v4_feature_family_controls.csv")
    make_ablation_figure(final_fig / "fig5_cell_group_ablation.png", v4 / "tables" / "table_v4_group_ablation.csv")
    if (temporal / "fig_temporal_windows.png").exists():
        copy_if_exists(temporal / "fig_temporal_windows.png", final_fig / "fig6_temporal_lesion.png")
        copy_if_exists(temporal / "fig_temporal_windows.pdf", final_fig / "fig6_temporal_lesion.pdf")
    make_bits_figure(final_fig / "fig7_bits_per_activity.png", v4 / "tables" / "table_v4_representation_metrics.csv")

    table_map = {
        "table2_feature_family_results": v4 / "tables" / "table_v4_feature_family_controls.csv",
        "table3_strong_trained_controls": strong / "table_strong_vision_controls_summary.csv",
        "table4_group_ablation": v4 / "tables" / "table_v4_group_ablation.csv",
        "table5_temporal_lesions": temporal / "table_temporal_lesion_accuracy.csv",
        "table6_representation_metrics": v4 / "tables" / "table_v4_representation_metrics.csv",
    }
    missing = []
    save_table(consolidated_main_table(v4 / "tables", strong), final_tab / "table1_main_results.csv")
    for name, src in table_map.items():
        if src.exists():
            save_table(pd.read_csv(src), final_tab / f"{name}.csv")
        else:
            missing.append(str(src))
    write_json(
        {
            "status": "completed",
            "missing_optional_sources": missing,
            "elapsed_seconds": time.time() - t0,
            "python": sys.version,
        },
        out / "finalization_run_info.json",
    )
    print(f"Wrote final claims, figures, and tables to {out}")


if __name__ == "__main__":
    main()
