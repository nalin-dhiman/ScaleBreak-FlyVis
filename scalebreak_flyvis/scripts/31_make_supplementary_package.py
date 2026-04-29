#!/usr/bin/env python
"""Create supplementary material package for ScaleBreak-FlyVis."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from figures_pub import (  # noqa: E402
    PALETTE,
    clean_axis,
    format_scale_labels,
    hex_mapping,
    model_color,
    panel_label,
    project_hex_frame,
    save_pub,
    save_table,
    set_pub_style,
)


def read_csv(path: Path, warnings: list[str]) -> pd.DataFrame:
    if not path.exists():
        warnings.append(f"Missing source table: {path}")
        return pd.DataFrame()
    return pd.read_csv(path)


def save_placeholder_figure(path: Path, title: str, message: str) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.text(0.5, 0.55, title, ha="center", va="center", fontsize=9, fontweight="bold")
    ax.text(0.5, 0.40, message, ha="center", va="center", fontsize=8)
    ax.axis("off")
    save_pub(fig, path)
    plt.close(fig)


def heatmap(ax, df: pd.DataFrame, title: str, value_col: str = "accuracy") -> None:
    if df.empty:
        ax.text(0.5, 0.5, "TODO: missing table", ha="center", va="center")
        ax.axis("off")
        return
    pivot = df.pivot_table(index="train_scale", columns="test_scale", values=value_col, aggfunc="mean").sort_index().sort_index(axis=1)
    im = ax.imshow(pivot.values, vmin=0, vmax=1, cmap="viridis", aspect="equal")
    ax.set_xticks(np.arange(len(pivot.columns)), format_scale_labels(pivot.columns))
    ax.set_yticks(np.arange(len(pivot.index)), format_scale_labels(pivot.index))
    ax.set_xlabel("Test scale")
    ax.set_ylabel("Train scale")
    ax.set_title(title, pad=3)
    return im


def fig_s1_stimulus_grid(fig_dir: Path, outputs: Path, warnings: list[str]) -> None:
    import matplotlib.pyplot as plt

    stim_path = outputs / "flyvis_pilot_v2" / "stimuli" / "stimuli.npy"
    meta_path = outputs / "flyvis_pilot_v2" / "stimuli" / "metadata.csv"
    coord_path = outputs / "flyvis_pilot_v2" / "stimuli" / "hex_coordinates.csv"
    if not (stim_path.exists() and meta_path.exists() and coord_path.exists()):
        warnings.append("Fig. S1 missing Pilot v2 stimuli or coordinates.")
        save_placeholder_figure(fig_dir / "figS1_full_stimulus_grid", "Fig. S1", "TODO: missing stimulus tensor or coordinates")
        return
    stimuli = np.load(stim_path, mmap_mode="r")
    meta = pd.read_csv(meta_path)
    mapping = hex_mapping(pd.read_csv(coord_path), 28)
    families = ["moving_edge", "moving_bar", "small_translating_target", "looming_disk", "static_shape"]
    family_labels = ["edge", "bar", "target", "looming", "static"]
    scales = sorted(meta["scale"].unique())
    fig, axes = plt.subplots(len(families), len(scales), figsize=(7.2, 4.8))
    for r, family in enumerate(families):
        for c, scale in enumerate(scales):
            ax = axes[r, c]
            sub = meta[(meta["feature_family"] == family) & (meta["scale"] == scale) & (meta["contrast"] == 1.0)]
            if sub.empty:
                ax.text(0.5, 0.5, "NA", ha="center", va="center", fontsize=6)
            else:
                row = sub.iloc[0]
                frame = int(row["n_frames"] * 0.65)
                ax.imshow(project_hex_frame(stimuli[int(row["sample"]), frame, 0], mapping, 28), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if r == 0:
                ax.set_title(str(int(scale)), fontsize=7, pad=1)
            if c == 0:
                ax.set_ylabel(family_labels[r], fontsize=7, rotation=0, ha="right", va="center")
    fig.suptitle("Full stimulus grid examples across retinal apparent scale", fontsize=9, y=1.01)
    save_pub(fig, fig_dir / "figS1_full_stimulus_grid")
    plt.close(fig)


def fig_s2_mapping(fig_dir: Path, outputs: Path, warnings: list[str]) -> None:
    import matplotlib.pyplot as plt

    coord_path = outputs / "flyvis_pilot_v2" / "stimuli" / "hex_coordinates.csv"
    if not coord_path.exists():
        warnings.append("Fig. S2 missing hex coordinates.")
        save_placeholder_figure(fig_dir / "figS2_hex_to_grid_mapping", "Fig. S2", "TODO: missing hex coordinates")
        return
    coords = pd.read_csv(coord_path)
    mapping = hex_mapping(coords, 32)
    occ = np.zeros((32, 32), dtype=float)
    for _, row in mapping.iterrows():
        occ[int(row["grid_y"]), int(row["grid_x"])] += 1
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.0))
    panel_label(axes[0], "A")
    axes[0].scatter(coords["x"], coords["y"], s=4, color=PALETTE["flyvis"], alpha=0.7)
    axes[0].set_title("Native hex coordinates")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    clean_axis(axes[0], grid=False)
    panel_label(axes[1], "B")
    im = axes[1].imshow(occ, cmap="viridis", interpolation="nearest")
    axes[1].set_title("32 × 32 projection occupancy")
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    fig.colorbar(im, ax=axes[1], fraction=0.045, pad=0.03)
    save_pub(fig, fig_dir / "figS2_hex_to_grid_mapping")
    plt.close(fig)


def fig_s3_matrices(fig_dir: Path, outputs: Path, warnings: list[str]) -> None:
    import matplotlib.pyplot as plt

    direction = read_csv(outputs / "flyvis_pilot_v2" / "tables" / "direction_scale_generalization.csv", warnings)
    feature = read_csv(outputs / "flyvis_pilot_v2" / "tables" / "feature_family_scale_generalization.csv", warnings)
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6))
    panel_label(axes[0], "A")
    im = heatmap(axes[0], direction, "FlyVis direction")
    panel_label(axes[1], "B")
    heatmap(axes[1], feature, "FlyVis feature family")
    panel_label(axes[2], "C")
    axes[2].text(0.5, 0.55, "TODO", ha="center", va="center", fontsize=9, fontweight="bold")
    axes[2].text(0.5, 0.40, "Per-model control scale matrices\nnot present in current outputs", ha="center", va="center", fontsize=7)
    axes[2].axis("off")
    if im is not None:
        fig.colorbar(im, ax=axes[:2], fraction=0.035, pad=0.02, label="Accuracy")
    warnings.append("Fig. S3 includes TODO panel because full per-model scale-generalization matrices were not available.")
    save_pub(fig, fig_dir / "figS3_full_scale_generalization_matrices")
    plt.close(fig)


def fig_s4_feature_controls(fig_dir: Path, outputs: Path, warnings: list[str]) -> None:
    import matplotlib.pyplot as plt

    df = read_csv(outputs / "final_hardening" / "final_tables" / "table2_feature_family_results.csv", warnings)
    fig, ax = plt.subplots(figsize=(7.0, 3.3))
    if df.empty:
        ax.text(0.5, 0.5, "TODO", ha="center", va="center")
    else:
        plot = df.set_index("feature_family_task")
        im = ax.imshow(plot.values, vmin=0, vmax=1, cmap="viridis", aspect="auto")
        ax.set_xticks(np.arange(len(plot.columns)), plot.columns, rotation=90, fontsize=6.5)
        ax.set_yticks(np.arange(len(plot.index)), plot.index, fontsize=7)
        fig.colorbar(im, ax=ax, label="Accuracy")
    ax.set_title("Feature-family controls")
    save_pub(fig, fig_dir / "figS4_feature_family_controls")
    plt.close(fig)


def fig_s5_serious_cnn(fig_dir: Path, outputs: Path, warnings: list[str]) -> None:
    import matplotlib.pyplot as plt

    base = outputs / "serious_cnn_baseline"
    curves = read_csv(base / "training_curves.csv", warnings)
    pred_files = sorted((base / "predictions").glob("predictions_TemporalResNet18Small_*.csv"))
    preds = pd.concat([pd.read_csv(p) for p in pred_files], ignore_index=True) if pred_files else pd.DataFrame()
    if preds.empty:
        warnings.append("Fig. S5 missing serious CNN predictions.")
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))
    panel_label(axes[0], "A")
    if curves.empty:
        axes[0].text(0.5, 0.5, "TODO: missing curves", ha="center", va="center")
    else:
        for seed, df in curves.groupby("seed"):
            by = df.groupby("epoch")["val_accuracy"].mean()
            axes[0].plot(by.index, by.values, label=f"seed {seed}", lw=1.2)
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Validation accuracy")
        axes[0].legend(frameon=False, ncol=2, fontsize=6)
        clean_axis(axes[0])
    panel_label(axes[1], "B")
    if preds.empty:
        axes[1].text(0.5, 0.5, "TODO: missing predictions", ha="center", va="center")
    else:
        cm = pd.crosstab(preds["true_label"], preds["pred_label"], normalize="index")
        im = axes[1].imshow(cm.values, vmin=0, vmax=1, cmap="viridis")
        axes[1].set_xticks(np.arange(len(cm.columns)), cm.columns, rotation=90)
        axes[1].set_yticks(np.arange(len(cm.index)), cm.index)
        axes[1].set_title("Confusion")
        fig.colorbar(im, ax=axes[1], fraction=0.045, pad=0.03)
    save_pub(fig, fig_dir / "figS5_serious_cnn_training_confusion")
    plt.close(fig)


def fig_s6_destructive(fig_dir: Path, outputs: Path, warnings: list[str]) -> None:
    import matplotlib.pyplot as plt

    df = read_csv(outputs / "flyvis_pilot_v4" / "tables" / "destructive_representation_controls.csv", warnings)
    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    if df.empty:
        ax.text(0.5, 0.5, "TODO", ha="center", va="center")
    else:
        df = df.sort_values("offdiag_accuracy")
        ax.barh(df["model"], df["offdiag_accuracy"], color=PALETTE["destructive"], edgecolor="black", linewidth=0.35)
        ax.axvline(1 / 6, color="black", linestyle="--", lw=0.9, label="chance")
        ax.set_xlabel("Off-diagonal direction accuracy")
        ax.legend(frameon=False)
        clean_axis(ax)
    ax.set_title("Destructive representation controls collapse decoding")
    save_pub(fig, fig_dir / "figS6_destructive_controls")
    plt.close(fig)


def fig_s7_temporal(fig_dir: Path, outputs: Path, warnings: list[str]) -> None:
    import matplotlib.pyplot as plt

    lesion = read_csv(outputs / "final_hardening" / "temporal_lesions" / "table_temporal_lesion_accuracy.csv", warnings)
    time_df = read_csv(outputs / "final_hardening" / "temporal_lesions" / "table_time_resolved_accuracy.csv", warnings)
    family = read_csv(outputs / "final_hardening" / "temporal_lesions" / "table_temporal_feature_family.csv", warnings)
    fig, axes = plt.subplots(1, 3, figsize=(8.2, 3.15), gridspec_kw={"width_ratios": [1.0, 1.25, 1.05]})
    panel_label(axes[0], "A")
    if not lesion.empty:
        sub = lesion.sort_values("accuracy")
        axes[0].barh(sub["feature_variant"], sub["accuracy"], color=PALETTE["flyvis"])
        axes[0].set_xlabel("Accuracy")
        clean_axis(axes[0])
    panel_label(axes[1], "B")
    if not time_df.empty:
        pretty = {
            "all_dynamic": "all dynamic",
            "moving_edge": "edge",
            "moving_bar": "bar",
            "small_translating_target": "target",
        }
        for fam, df in time_df.groupby("feature_family"):
            if fam in {"all_dynamic", "moving_edge", "moving_bar", "small_translating_target"}:
                axes[1].plot(df.sort_values("frame")["frame"], df.sort_values("frame")["accuracy"], label=pretty.get(fam, fam), lw=1.1)
        axes[1].set_xlabel("Frame")
        axes[1].set_ylabel("Accuracy")
        axes[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=2, frameon=False, fontsize=6)
        axes[1].set_xticks([0, 40, 80, 120, 160])
        clean_axis(axes[1])
    panel_label(axes[2], "C")
    if not family.empty:
        piv = family.pivot_table(index="feature_variant", columns="feature_family", values="accuracy", aggfunc="mean")
        im = axes[2].imshow(piv.values, vmin=0, vmax=1, cmap="viridis", aspect="auto")
        axes[2].set_xticks(np.arange(len(piv.columns)), piv.columns, rotation=90, fontsize=6)
        axes[2].set_yticks(np.arange(len(piv.index)), piv.index, fontsize=5.8)
        fig.colorbar(im, ax=axes[2], fraction=0.045, pad=0.03)
    fig.subplots_adjust(wspace=0.55, bottom=0.28)
    save_pub(fig, fig_dir / "figS7_temporal_lesion_full")
    plt.close(fig)


def fig_s8_ablation(fig_dir: Path, outputs: Path, warnings: list[str]) -> None:
    import matplotlib.pyplot as plt

    group = read_csv(outputs / "final_hardening" / "final_tables" / "table4_group_ablation.csv", warnings)
    rank = read_csv(outputs / "flyvis_pilot_v3" / "tables" / "celltype_ablation_importance.csv", warnings)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
    panel_label(axes[0], "A")
    if not group.empty:
        g = group.sort_values("drop_accuracy")
        axes[0].barh(g["ablation"], g["drop_accuracy"], color=PALETTE["flyvis"])
        axes[0].set_xlabel("Drop")
        clean_axis(axes[0])
    panel_label(axes[1], "B")
    if not rank.empty:
        r = rank.sort_values("drop_accuracy", ascending=False).head(15).sort_values("drop_accuracy")
        axes[1].barh(r["cell_type"], r["drop_accuracy"], color=PALETTE["graph"])
        axes[1].set_xlabel("Single-type drop")
        clean_axis(axes[1])
    save_pub(fig, fig_dir / "figS8_celltype_ablation_ranking")
    plt.close(fig)


def fig_s9_rsa_cka(fig_dir: Path, outputs: Path, warnings: list[str]) -> None:
    import matplotlib.pyplot as plt

    df = read_csv(outputs / "final_hardening" / "final_tables" / "table6_representation_metrics.csv", warnings)
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.15), gridspec_kw={"width_ratios": [1.05, 1.0]})
    panel_label(axes[0], "A")
    panel_label(axes[1], "B")
    if not df.empty:
        order = df.sort_values("rsa_same_direction_cross_scale_margin", ascending=False)
        axes[0].bar(np.arange(len(order)), order["rsa_same_direction_cross_scale_margin"], color=[model_color(m) for m in order["model"]])
        axes[0].set_xticks(np.arange(len(order)), order["model"], rotation=90)
        axes[0].set_ylabel("RSA margin")
        clean_axis(axes[0])
        order = df.sort_values("mean_offdiag_cka", ascending=False)
        axes[1].bar(np.arange(len(order)), order["mean_offdiag_cka"], color=[model_color(m) for m in order["model"]])
        axes[1].set_xticks(np.arange(len(order)), order["model"], rotation=90)
        axes[1].set_ylabel("Mean off-diagonal CKA")
        clean_axis(axes[1])
    save_pub(fig, fig_dir / "figS9_rsa_cka")
    plt.close(fig)


def fig_s10_causality(fig_dir: Path, outputs: Path, warnings: list[str]) -> None:
    import matplotlib.pyplot as plt

    causal = read_csv(outputs / "connectome_causality" / "table_causal_variants.csv", warnings)
    activity = read_csv(outputs / "connectome_causality" / "table_causal_activity_metrics.csv", warnings)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))
    panel_label(axes[0], "A")
    panel_label(axes[1], "B")
    if not causal.empty:
        c = causal.sort_values("offdiag_direction_accuracy")
        short = {
            "full": "full",
            "edge_dropout_proxy_0.10": "edge dropout",
            "t4t5_attenuation_0.50": "T4/T5 atten.",
            "non_t4t5_attenuation_0.50": "non-T4/T5 atten.",
        }
        axes[0].barh(c["variant"].map(lambda v: short.get(str(v), str(v).replace("_", " "))), c["offdiag_direction_accuracy"], color=PALETTE["graph"])
        axes[0].set_xlabel("Accuracy")
        clean_axis(axes[0])
    if not causal.empty and not activity.empty:
        merged = causal.merge(activity[["variant", "mean_abs_activity_proxy"]], on="variant", how="left")
        short = {
            "full": "full",
            "edge_dropout_proxy_0.10": "edge dropout",
            "t4t5_attenuation_0.50": "T4/T5 atten.",
            "non_t4t5_attenuation_0.50": "non-T4/T5 atten.",
        }
        colors = [PALETTE["flyvis"], PALETTE["graph"], PALETTE["local_rnn"], PALETTE["nuisance"], PALETTE["destructive"], PALETTE["pixel"]]
        markers = ["o", "s", "^", "D", "P", "X"]
        for i, (_, r) in enumerate(merged.iterrows()):
            axes[1].scatter(
                r["mean_abs_activity_proxy"],
                r["offdiag_direction_accuracy"],
                color=colors[i % len(colors)],
                marker=markers[i % len(markers)],
                s=42,
                edgecolor="black",
                linewidth=0.35,
                label=short.get(str(r["variant"]), str(r["variant"]).replace("_", " ")),
                zorder=3,
            )
        axes[1].set_xlabel("Activity proxy")
        axes[1].set_ylabel("Accuracy")
        axes[1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=6)
        clean_axis(axes[1])
    fig.subplots_adjust(wspace=0.58, right=0.78)
    save_pub(fig, fig_dir / "figS10_connectome_causality_proxy_variants")
    plt.close(fig)


def build_tables(tab_dir: Path, outputs: Path, warnings: list[str]) -> None:
    meta = read_csv(outputs / "flyvis_pilot_v2" / "stimuli" / "metadata.csv", warnings)
    if not meta.empty:
        schema = pd.DataFrame(
            {
                "column": meta.columns,
                "dtype": [str(meta[c].dtype) for c in meta.columns],
                "missing_fraction": [float(meta[c].isna().mean()) for c in meta.columns],
                "example": [str(meta[c].iloc[0]) for c in meta.columns],
            }
        )
    else:
        schema = pd.DataFrame({"TODO": ["stimulus metadata missing"]})
    save_table(schema, tab_dir / "tableS1_stimulus_metadata_schema.csv")

    rows = []
    for label, path in [
        ("stimuli", outputs / "flyvis_pilot_v2" / "stimuli" / "stimuli.npy"),
        ("flyvis_central_responses", outputs / "flyvis_pilot_v2" / "responses" / "flyvis_central_cell_responses.npy"),
    ]:
        if path.exists():
            arr = np.load(path, mmap_mode="r")
            rows.append({"tensor": label, "path": str(path), "shape": tuple(arr.shape), "dtype": str(arr.dtype)})
        else:
            rows.append({"tensor": label, "path": str(path), "shape": "TODO missing", "dtype": "TODO"})
            warnings.append(f"Missing tensor for Table S2: {path}")
    save_table(pd.DataFrame(rows), tab_dir / "tableS2_tensor_response_shapes.csv")

    hyper = pd.DataFrame(
        [
            {"model": "FlyVis", "training": "pretrained flow/0000/000", "readout": "linear probe", "notes": "central cell responses"},
            {"model": "TemporalResNet18Small", "training": "AdamW, lr 3e-4, weight decay 1e-4, early stopping", "readout": "native classifier", "notes": "10 temporal summary channels"},
            {"model": "pixel", "training": "none", "readout": "linear probe", "notes": "retinal/pixel baseline"},
            {"model": "local RNN", "training": "fixed random", "readout": "linear probe", "notes": "local retinotopic recurrent baseline"},
            {"model": "optic-lobe graph controls", "training": "fixed rate proxy", "readout": "linear probe", "notes": "not FlyVis; structural debug controls"},
        ]
    )
    save_table(hyper, tab_dir / "tableS3_model_control_hyperparameters.csv")

    bootstrap = read_csv(outputs / "flyvis_pilot_v4" / "tables" / "bootstrap_ci_v4.csv", warnings)
    serious = read_csv(outputs / "serious_cnn_baseline" / "table_serious_cnn_summary.csv", warnings)
    rows = []
    if not bootstrap.empty:
        for _, r in bootstrap.iterrows():
            if "offdiag direction accuracy" in r["metric"]:
                rows.append({"metric": r["metric"], "estimate": r["estimate"], "ci_low": r["ci_low"], "ci_high": r["ci_high"]})
    if not serious.empty:
        r = serious.iloc[0]
        rows.append({"metric": "TemporalResNet18Small serious CNN offdiag direction accuracy", "estimate": r["mean_offdiag_accuracy"], "ci_low": r["ci_low"], "ci_high": r["ci_high"]})
    save_table(pd.DataFrame(rows), tab_dir / "tableS4_accuracy_ci_results.csv")

    copies = [
        ("tableS5_feature_family_results.csv", outputs / "final_hardening" / "final_tables" / "table2_feature_family_results.csv"),
        ("tableS6_temporal_lesion_results.csv", outputs / "final_hardening" / "temporal_lesions" / "table_temporal_lesion_accuracy.csv"),
        ("tableS7_cell_group_ablation_results.csv", outputs / "final_hardening" / "final_tables" / "table4_group_ablation.csv"),
        ("tableS8_serious_cnn_per_seed_per_scale.csv", outputs / "serious_cnn_baseline" / "table_serious_cnn_by_seed_scale.csv"),
        ("tableS9_connectome_causality_pilot_results.csv", outputs / "connectome_causality" / "table_causal_variants.csv"),
    ]
    for name, src in copies:
        df = read_csv(src, warnings)
        if df.empty:
            df = pd.DataFrame({"TODO": [f"missing source {src}"]})
        save_table(df, tab_dir / name)

    claims_path = outputs / "final_hardening" / "post_hardening_claims.json"
    if not claims_path.exists():
        claims_path = outputs / "final_hardening" / "final_claims.json"
    if claims_path.exists():
        claims = json.loads(claims_path.read_text(encoding="utf-8"))
        checklist = pd.DataFrame([{"claim": k, "status": v} for k, v in claims.items()])
    else:
        checklist = pd.DataFrame({"TODO": ["claim checklist missing"]})
        warnings.append("Missing final claims JSON for Table S10.")
    save_table(checklist, tab_dir / "tableS10_final_claim_checklist.csv")


def write_material(out: Path, warnings: list[str]) -> None:
    notes = [
        ("Supplementary Note 1: Dataset and stimulus generation", "Stimuli use FlyVis-native tensors with axes `(sample, frame, channel, hex_pixel)`. The experiments manipulate retinal apparent scale and related retinal variables, not physical distance."),
        ("Supplementary Note 2: FlyVis response extraction", "Responses were extracted from the pretrained FlyVis `flow/0000/000` model and summarized at central cell/type level."),
        ("Supplementary Note 3: Control models and training details", "Controls include pixel features, fixed local RNN/CNN-style baselines, optic-lobe type-rate graph controls, and the serious TemporalResNet18Small baseline."),
        ("Supplementary Note 4: Statistical protocol", "Main decoding uses leave-one-apparent-scale-out evaluation with bootstrap confidence intervals where available."),
        ("Supplementary Note 5: Temporal lesion analysis", "Temporal windows and time-resolved probes localize when direction information becomes scale-generalizing."),
        ("Supplementary Note 6: Cell-group and ablation analyses", "T4/T5 groups are informative, but small ablation drops support a distributed representation rather than a single-cell-type mechanism."),
        ("Supplementary Note 7: Connectome-causality pilot", "Current causal variants are response-space proxy perturbations unless direct FlyVis weight editing is validated. Exact connectome necessity is not established."),
        ("Supplementary Note 8: Limitations and negative controls", "The benchmark uses synthetic stimuli and linear probes. Static shape identity is secondary; no generic object-recognition or physical-distance claim is made."),
    ]
    md = ["# Supplementary Material", ""]
    for title, body in notes:
        md.extend([f"## {title}", "", body, ""])
    md.extend(["## Supplementary Figures", ""])
    for i in range(1, 11):
        md.append(f"- Fig. S{i}: see `supplementary_figures/`.")
    md.extend(["", "## Supplementary Tables", ""])
    for i in range(1, 11):
        md.append(f"- Table S{i}: see `supplementary_tables/`.")
    if warnings:
        md.extend(["", "## TODO / Warnings", ""])
        md.extend([f"- {w}" for w in warnings])
    (out / "supplementary_material.md").write_text("\n".join(md), encoding="utf-8")

    tex_lines = [
        r"\documentclass{article}",
        r"\usepackage{graphicx}",
        r"\usepackage{booktabs}",
        r"\usepackage[margin=1in]{geometry}",
        r"\begin{document}",
        r"\section*{Supplementary Material}",
    ]
    for title, body in notes:
        tex_lines.extend([rf"\subsection*{{{title}}}", body.replace("%", r"\%")])
    tex_lines.extend(
        [
            r"\section*{Supplementary Figures and Tables}",
            r"Supplementary figures are provided as PDF/SVG/PNG files in \texttt{supplementary\_figures/}. Supplementary tables are provided as CSV and Markdown files in \texttt{supplementary\_tables/}.",
            r"\end{document}",
        ]
    )
    (out / "supplementary_material.tex").write_text("\n".join(tex_lines), encoding="utf-8")


def write_manifest(out: Path, warnings: list[str]) -> None:
    figs = [
        ("Fig. S1", "full stimulus grid", "Pilot v2 stimuli"),
        ("Fig. S2", "hex-to-grid mapping audit", "Pilot v2 hex coordinates"),
        ("Fig. S3", "scale-generalization matrices", "Pilot v2 scale-generalization tables"),
        ("Fig. S4", "feature-family controls", "final feature-family table"),
        ("Fig. S5", "serious CNN training curves and confusion", "serious CNN outputs"),
        ("Fig. S6", "destructive controls", "Pilot v4 destructive controls table"),
        ("Fig. S7", "temporal lesion full result", "temporal lesion outputs"),
        ("Fig. S8", "cell-type ablation/ranking", "v3 and final ablation tables"),
        ("Fig. S9", "RSA/CKA", "representation metrics"),
        ("Fig. S10", "connectome-causality proxy variants", "connectome causality outputs"),
    ]
    tables = [f"Table S{i}" for i in range(1, 11)]
    lines = ["# Supplementary Manifest", "", "## Figures"]
    for label, desc, source in figs:
        lines.append(f"- {label}: {desc}. Source: `{source}`.")
    lines.extend(["", "## Tables"])
    for label in tables:
        lines.append(f"- {label}: see `supplementary_tables/` CSV and Markdown exports.")
    if warnings:
        lines.extend(["", "## Warnings / TODO"])
        lines.extend([f"- {w}" for w in warnings])
    else:
        lines.extend(["", "## Warnings / TODO", "- None."])
    (out / "supplementary_manifest.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs-dir", default="scalebreak_flyvis/outputs")
    parser.add_argument("--out-dir", default="scalebreak_flyvis/outputs/supplementary")
    args = parser.parse_args()
    set_pub_style()
    outputs = Path(args.outputs_dir)
    out = Path(args.out_dir)
    fig_dir = out / "supplementary_figures"
    tab_dir = out / "supplementary_tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)
    warnings: list[str] = []
    fig_s1_stimulus_grid(fig_dir, outputs, warnings)
    fig_s2_mapping(fig_dir, outputs, warnings)
    fig_s3_matrices(fig_dir, outputs, warnings)
    fig_s4_feature_controls(fig_dir, outputs, warnings)
    fig_s5_serious_cnn(fig_dir, outputs, warnings)
    fig_s6_destructive(fig_dir, outputs, warnings)
    fig_s7_temporal(fig_dir, outputs, warnings)
    fig_s8_ablation(fig_dir, outputs, warnings)
    fig_s9_rsa_cka(fig_dir, outputs, warnings)
    fig_s10_causality(fig_dir, outputs, warnings)
    build_tables(tab_dir, outputs, warnings)
    write_material(out, warnings)
    write_manifest(out, warnings)
    print(f"Wrote supplementary package to {out}")


if __name__ == "__main__":
    main()
