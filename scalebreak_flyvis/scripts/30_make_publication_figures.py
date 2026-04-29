#!/usr/bin/env python
"""Build publication-grade vector figures for ScaleBreak-FlyVis."""

from __future__ import annotations

import argparse
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
    annotate_bars,
    clean_axis,
    format_scale_labels,
    hex_mapping,
    model_color,
    panel_label,
    project_hex_frame,
    save_pub,
    set_pub_style,
)


GROUP_LEGEND = {
    "FlyVis / variant": PALETTE["flyvis"],
    "vision models": PALETTE["cnn"],
    "pixel": PALETTE["pixel"],
    "local RNN": PALETTE["local_rnn"],
    "graph controls": PALETTE["graph"],
    "nuisance": PALETTE["nuisance"],
    "destructive controls": PALETTE["destructive"],
}


def read_csv(path: Path, warnings: list[str]) -> pd.DataFrame:
    if not path.exists():
        warnings.append(f"Missing source table: {path}")
        return pd.DataFrame()
    return pd.read_csv(path)


def ci_lookup(bootstrap: pd.DataFrame, model_name: str) -> tuple[float | None, float | None, float | None]:
    if bootstrap.empty:
        return None, None, None
    pattern = f"{model_name} offdiag direction accuracy"
    row = bootstrap[bootstrap["metric"].str.lower() == pattern.lower()]
    if row.empty:
        return None, None, None
    r = row.iloc[0]
    return float(r["estimate"]), float(r["ci_low"]), float(r["ci_high"])


def serious_cnn_row(path: Path, warnings: list[str]) -> dict[str, float] | None:
    df = read_csv(path, warnings)
    if df.empty:
        return None
    r = df.iloc[0]
    return {
        "estimate": float(r["mean_offdiag_accuracy"]),
        "ci_low": float(r["ci_low"]),
        "ci_high": float(r["ci_high"]),
    }


def figure1_concept(out: Path, outputs: Path, warnings: list[str]) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, FancyArrowPatch, Rectangle

    stim_path = outputs / "flyvis_pilot_v2" / "stimuli" / "stimuli.npy"
    meta_path = outputs / "flyvis_pilot_v2" / "stimuli" / "metadata.csv"
    coord_path = outputs / "flyvis_pilot_v2" / "stimuli" / "hex_coordinates.csv"
    fig = plt.figure(figsize=(7.2, 4.6), constrained_layout=True)
    gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 1.15])
    ax = fig.add_subplot(gs[0, :])
    panel_label(ax, "A", x=-0.03, y=1.02)
    ax.set_title("Retinal apparent scale benchmark", pad=3)
    eye = Circle((0.08, 0.5), 0.055, fill=False, lw=1.5, color=PALETTE["black"])
    ax.add_patch(eye)
    for x, size, label in [(0.35, 0.12, "large\nprojection"), (0.60, 0.075, "medium"), (0.82, 0.045, "small")]:
        ax.add_patch(Circle((x, 0.5), size, fill=False, lw=1.3, color=PALETTE["flyvis"]))
        ax.add_patch(FancyArrowPatch((0.13, 0.5), (x - size, 0.5), arrowstyle="-", lw=0.8, color="0.35"))
        ax.text(x, 0.23, label, ha="center", va="center", fontsize=7)
    for i, text in enumerate(["scale", "contrast", "motion"]):
        ax.add_patch(Rectangle((0.18 + i * 0.17, 0.79), 0.11, 0.08, fill=False, lw=0.9, color="0.2"))
        ax.text(0.235 + i * 0.17, 0.83, text, ha="center", va="center", fontsize=7)
    ax.text(0.08, 0.30, "eye", ha="center", fontsize=7)
    ax.text(
        0.98,
        0.93,
        "retinal projection\nnot physical distance",
        fontsize=7.5,
        ha="right",
        va="top",
        color="0.2",
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0.1, 0.95)
    ax.axis("off")

    if not (stim_path.exists() and meta_path.exists() and coord_path.exists()):
        warnings.append("Figure 1 stimulus examples skipped because Pilot v2 stimuli/metadata/coordinates were missing.")
        return
    stimuli = np.load(stim_path, mmap_mode="r")
    meta = pd.read_csv(meta_path)
    coords = pd.read_csv(coord_path)
    mapping = hex_mapping(coords, 32)
    families = ["moving_edge", "moving_bar", "small_translating_target", "looming_disk"]
    titles = ["moving edge", "moving bar", "small target", "looming disk"]
    for i, (family, title) in enumerate(zip(families, titles)):
        ax_img = fig.add_subplot(gs[1, i])
        if i == 0:
            panel_label(ax_img, "B", x=-0.28, y=1.05)
        candidates = meta[(meta["feature_family"] == family) & (meta["contrast"] == 1.0) & (meta["scale"].isin([8.0, 12.0, 16.0]))]
        if candidates.empty:
            candidates = meta[meta["feature_family"] == family]
        if candidates.empty:
            ax_img.text(0.5, 0.5, "missing", ha="center", va="center")
            warnings.append(f"No stimulus example found for {family}.")
        else:
            row = candidates.iloc[0]
            sample = int(row["sample"])
            frame = int(row["n_frames"] * 0.65)
            img = project_hex_frame(stimuli[sample, frame, 0], mapping, 32)
            ax_img.imshow(img, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            ax_img.set_title(f"{title}\nscale {int(row['scale'])}", pad=2)
        ax_img.set_xticks([])
        ax_img.set_yticks([])
        for spine in ax_img.spines.values():
            spine.set_visible(False)
    save_pub(fig, out / "fig1_concept")
    plt.close(fig)


def figure2_matrix(out: Path, outputs: Path, warnings: list[str]) -> None:
    import matplotlib.pyplot as plt

    pred_path = outputs / "flyvis_pilot_v3" / "tables" / "direction_loso_predictions_all_models.csv"
    df = read_csv(pred_path, warnings)
    fig, ax = plt.subplots(figsize=(6.3, 2.7))
    panel_label(ax, "A", x=-0.16, y=1.04)
    if df.empty:
        ax.text(0.5, 0.5, "TODO: missing LOSO predictions", ha="center", va="center")
    else:
        rename = {
            "flyvis": "FlyVis",
            "pixel": "pixel",
            "cnn": "CNN",
            "local_rnn": "local RNN",
            "nuisance_area_contrast_energy": "nuisance",
        }
        df = df[df["model"].isin(rename)].copy()
        df["model"] = df["model"].map(rename)
        rows = []
        for (model, scale), sub in df.groupby(["model", "heldout_scale"]):
            rows.append({"model": model, "heldout_scale": float(scale), "accuracy": float(sub["correct"].astype(bool).mean())})
        loso = pd.DataFrame(rows)
        serious_path = outputs / "serious_cnn_baseline" / "table_serious_cnn_by_seed_scale.csv"
        if serious_path.exists():
            serious = pd.read_csv(serious_path)
            for scale, sub in serious.groupby("heldout_scale"):
                loso = pd.concat(
                    [loso, pd.DataFrame([{"model": "TemporalResNet18Small", "heldout_scale": float(scale), "accuracy": float(sub["accuracy"].mean())}])],
                    ignore_index=True,
                )
        stn_path = outputs / "stn_cnn_baseline" / "table_stn_cnn_by_seed_scale.csv"
        if stn_path.exists():
            stn = pd.read_csv(stn_path)
            for scale, sub in stn.groupby("heldout_scale"):
                loso = pd.concat(
                    [loso, pd.DataFrame([{"model": "STN-CNN", "heldout_scale": float(scale), "accuracy": float(sub["accuracy"].mean())}])],
                    ignore_index=True,
                )
        order = ["FlyVis", "TemporalResNet18Small", "STN-CNN", "pixel", "CNN", "local RNN", "nuisance"]
        pivot = (
            loso.pivot_table(index="model", columns="heldout_scale", values="accuracy", aggfunc="mean")
            .reindex(order)
            .dropna(how="all")
            .sort_index(axis=1)
        )
        im = ax.imshow(pivot.values, vmin=0, vmax=1, cmap="viridis", aspect="auto")
        ax.set_xticks(np.arange(len(pivot.columns)), format_scale_labels(pivot.columns))
        ax.set_yticks(np.arange(len(pivot.index)), pivot.index)
        ax.set_xlabel("Held-out apparent scale")
        ax.set_ylabel("Model")
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                val = pivot.values[i, j]
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=5.8, color="white" if val < 0.55 else "black")
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cbar.set_label("Accuracy")
    ax.set_title("LOSO held-out-scale accuracy", pad=4)
    save_pub(fig, out / "fig2_scale_matrix")
    plt.close(fig)


def main_controls_dataframe(outputs: Path, warnings: list[str]) -> pd.DataFrame:
    bootstrap = read_csv(outputs / "flyvis_pilot_v4" / "tables" / "bootstrap_ci_v4.csv", warnings)
    rows = []
    for label, lookup in [
        ("FlyVis", "FlyVis"),
        ("pixel", "pixel"),
        ("local RNN", "local RNN"),
        ("CNN", "CNN"),
        ("nuisance", "pixel area/contrast/energy nuisance"),
    ]:
        est, lo, hi = ci_lookup(bootstrap, lookup)
        if est is not None:
            rows.append({"model": label, "accuracy": est, "ci_low": lo, "ci_high": hi})
    serious = serious_cnn_row(outputs / "serious_cnn_baseline" / "table_serious_cnn_summary.csv", warnings)
    if serious is not None:
        rows.append({"model": "TemporalResNet18Small", "accuracy": serious["estimate"], "ci_low": serious["ci_low"], "ci_high": serious["ci_high"]})
    hex_native = serious_cnn_row(outputs / "hex_native_temporal_baseline" / "table_hex_native_summary.csv", warnings)
    if hex_native is not None:
        rows.append({"model": "Hex-native temporal model", "accuracy": hex_native["estimate"], "ci_low": hex_native["ci_low"], "ci_high": hex_native["ci_high"]})
    stn = serious_cnn_row(outputs / "stn_cnn_baseline" / "table_stn_cnn_summary.csv", warnings)
    if stn is not None:
        rows.append({"model": "STN-CNN", "accuracy": stn["estimate"], "ci_low": stn["ci_low"], "ci_high": stn["ci_high"]})
    variant = read_csv(outputs / "flyvis_variability_control" / "table_flyvis_response_noise_variability_summary.csv", warnings)
    if not variant.empty and "noise_fraction_of_feature_std" in variant.columns:
        v = variant[np.isclose(variant["noise_fraction_of_feature_std"], 0.05)]
        if not v.empty:
            r = v.iloc[0]
            rows.append(
                {
                    "model": "FlyVis + response noise",
                    "accuracy": float(r["mean_accuracy"]),
                    "ci_low": float(r["min_accuracy"]),
                    "ci_high": float(r["max_accuracy"]),
                }
            )
    strong = read_csv(outputs / "final_hardening" / "strong_controls" / "table_strong_vision_controls_summary.csv", warnings)
    if not strong.empty and "model" in strong.columns and (strong["model"] == "cnn3d").any():
        r = strong[strong["model"] == "cnn3d"].iloc[0]
        rows.append(
            {
                "model": "3D CNN",
                "accuracy": float(r["mean_offdiag_accuracy"]),
                "ci_low": float(r["ci_low"]),
                "ci_high": float(r["ci_high"]),
            }
        )
    graph_rows = []
    for name in ["real optic-lobe graph", "degree-matched graph", "weight-shuffled graph", "type-shuffled graph"]:
        est, lo, hi = ci_lookup(bootstrap, name)
        if est is not None:
            graph_rows.append((est, lo, hi))
    if graph_rows:
        arr = np.asarray(graph_rows)
        rows.append({"model": "graph controls", "accuracy": float(arr[:, 0].mean()), "ci_low": float(arr[:, 1].min()), "ci_high": float(arr[:, 2].max())})
    destructive_rows = []
    for name in [
        "response-shuffled FlyVis",
        "time-shuffled FlyVis",
        "cell-identity-shuffled FlyVis",
        "Gaussian response noise",
        "direction-label permutation",
        "random cell dropout mismatch",
    ]:
        est, lo, hi = ci_lookup(bootstrap, name)
        if est is not None:
            destructive_rows.append((est, lo, hi))
    if destructive_rows:
        arr = np.asarray(destructive_rows)
        rows.append({"model": "destructive controls", "accuracy": float(arr[:, 0].mean()), "ci_low": float(arr[:, 1].min()), "ci_high": float(arr[:, 2].max())})
    df = pd.DataFrame(rows).drop_duplicates("model", keep="last")
    return df.sort_values("accuracy", ascending=False)


def figure3_controls(out: Path, outputs: Path, warnings: list[str]) -> None:
    import matplotlib.pyplot as plt

    df = main_controls_dataframe(outputs, warnings)
    fig, ax = plt.subplots(figsize=(6.8, 3.3))
    panel_label(ax, "A", x=-0.11, y=1.05)
    if df.empty:
        ax.text(0.5, 0.5, "TODO: missing controls table", ha="center", va="center")
    else:
        def group(model: str) -> str:
            if "graph" in model.lower():
                return "graph"
            if "nuisance" in model.lower() or "destructive" in model.lower():
                return "null"
            return "trained"

        order = {"trained": 0, "graph": 1, "null": 2}
        df = df.assign(group=df["model"].map(group))
        df = df.sort_values(["group", "accuracy"], key=lambda s: s.map(order) if s.name == "group" else -s)
        colors = [model_color(m) for m in df["model"]]
        yerr = np.vstack([df["accuracy"] - df["ci_low"], df["ci_high"] - df["accuracy"]])
        x = np.arange(len(df))
        ax.bar(x, df["accuracy"], yerr=yerr, color=colors, edgecolor="black", linewidth=0.4, capsize=2)
        annotate_bars(ax, df["accuracy"], fmt="{:.2f}")
        ax.set_xticks(x, df["model"], rotation=28, ha="right")
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0, 1.05)
        groups = df["group"].tolist()
        changes = [i - 0.5 for i in range(1, len(groups)) if groups[i] != groups[i - 1]]
        for c in changes:
            ax.axvline(c, color="0.35", lw=0.75)
        from matplotlib.patches import Patch

        legend_items = [Patch(facecolor=color, edgecolor="black", linewidth=0.3, label=label) for label, color in GROUP_LEGEND.items()]
        ax.legend(handles=legend_items, loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False, title="Color key", title_fontsize=7)
        clean_axis(ax, grid=True)
    fig.subplots_adjust(right=0.73, bottom=0.28)
    ax.set_title("Main held-out-scale direction result", pad=4)
    save_pub(fig, out / "fig3_controls")
    plt.close(fig)


def figure4_feature_family(out: Path, outputs: Path, warnings: list[str]) -> None:
    import matplotlib.pyplot as plt

    df = read_csv(outputs / "final_hardening" / "final_tables" / "table2_feature_family_results.csv", warnings)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.35), sharey=True)
    panel_label(axes[0], "A", x=-0.16, y=1.05)
    panel_label(axes[1], "B", x=-0.16, y=1.05)
    if df.empty:
        axes[0].text(0.5, 0.5, "TODO: missing feature-family table", ha="center", va="center")
    else:
        plot = pd.DataFrame(
            {
                "feature_family": df["feature_family_task"]
                .str.replace("small translating target direction", "small target")
                .str.replace("moving edge direction", "moving edge")
                .str.replace("moving bar direction", "moving bar")
                .str.replace("looming angle-position", "looming")
                .str.replace("static appendix shape", "static shapes"),
                "FlyVis": df["FlyVis"],
                "pixel": df["pixel"],
                "CNN": df["CNN"],
                "local RNN": df["local RNN"],
                "graph controls": df[["real optic-lobe graph", "degree-matched graph", "weight-shuffled graph", "type-shuffled graph"]].mean(axis=1),
            }
        )
        x = np.arange(len(plot))
        panel_specs = [
            (axes[0], ["FlyVis", "CNN", "pixel"], "Primary comparison"),
            (axes[1], ["local RNN", "graph controls"], "Structural controls"),
        ]
        for ax, models, title in panel_specs:
            width = 0.22 if len(models) == 3 else 0.28
            offsets = (np.arange(len(models)) - (len(models) - 1) / 2) * width
            for off, model in zip(offsets, models):
                ax.bar(x + off, plot[model], width=width, label=model, color=model_color(model), edgecolor="black", linewidth=0.3)
            ax.axvspan(3.5, 4.5, color=PALETTE["light_gray"], alpha=0.35, zorder=0)
            ax.text(4, 0.98, "static", ha="center", va="top", fontsize=7)
            ax.set_xticks(x, plot["feature_family"], rotation=25, ha="right")
            ax.set_ylim(0, 1.05)
            ax.set_title(title, pad=4)
            ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
            clean_axis(ax, grid=True)
        axes[0].set_ylabel("Accuracy")
    fig.subplots_adjust(right=0.82, wspace=0.42)
    save_pub(fig, out / "fig4_feature_family")
    plt.close(fig)


def figure5_temporal(out: Path, outputs: Path, warnings: list[str]) -> None:
    import matplotlib.pyplot as plt

    lesion = read_csv(outputs / "final_hardening" / "temporal_lesions" / "table_temporal_lesion_accuracy.csv", warnings)
    time_df = read_csv(outputs / "final_hardening" / "temporal_lesions" / "table_time_resolved_accuracy.csv", warnings)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2), gridspec_kw={"width_ratios": [1.0, 1.35]})
    panel_label(axes[0], "A", x=-0.22, y=1.05)
    panel_label(axes[1], "B", x=-0.14, y=1.05)
    order = ["early_0_20pct", "middle_33_66pct", "late_66_100pct", "full_time_mean", "temporal_bins_5"]
    labels = ["early", "middle", "late", "full mean", "5 bins"]
    if lesion.empty:
        axes[0].text(0.5, 0.5, "TODO", ha="center", va="center")
    else:
        sub = lesion.set_index("feature_variant").loc[[o for o in order if o in set(lesion["feature_variant"])]]
        yerr = np.vstack([sub["accuracy"] - sub["ci_low"], sub["ci_high"] - sub["accuracy"]])
        axes[0].bar(np.arange(len(sub)), sub["accuracy"], yerr=yerr, color=PALETTE["flyvis"], edgecolor="black", linewidth=0.4, capsize=2)
        axes[0].set_xticks(np.arange(len(sub)), labels[: len(sub)], rotation=25, ha="right")
        axes[0].set_ylim(0, 1.05)
        axes[0].set_ylabel("Accuracy")
        clean_axis(axes[0], grid=True)
    if time_df.empty:
        axes[1].text(0.5, 0.5, "TODO", ha="center", va="center")
    else:
        all_dyn = time_df[time_df["feature_family"] == "all_dynamic"].sort_values("frame")
        smooth = all_dyn["accuracy"].rolling(3, center=True, min_periods=1).mean()
        axes[1].plot(all_dyn["frame"], smooth, color=PALETTE["flyvis"], lw=1.8)
        axes[1].axvspan(55, 120, color=PALETTE["light_gray"], alpha=0.24, lw=0)
        axes[1].set_xlabel("Frame")
        axes[1].set_ylabel("Accuracy")
        axes[1].set_xticks([0, 40, 80, 120, 160])
        axes[1].set_ylim(0, 1.05)
        clean_axis(axes[1], grid=True)
    axes[1].set_title("Time-resolved decoding", pad=4)
    fig.suptitle("Direction decoding strengthens after early frames", y=1.02, fontsize=9)
    save_pub(fig, out / "fig5_temporal")
    plt.close(fig)


def figure6_ablation(out: Path, outputs: Path, warnings: list[str]) -> None:
    import matplotlib.pyplot as plt

    df = read_csv(outputs / "final_hardening" / "final_tables" / "table4_group_ablation.csv", warnings)
    fig, ax = plt.subplots(figsize=(5.7, 3.4))
    panel_label(ax, "A", x=-0.18, y=1.05)
    if df.empty:
        ax.text(0.5, 0.5, "TODO", ha="center", va="center")
    else:
        keep = [
            "remove all T4",
            "remove all T5",
            "remove T4+T5",
            "remove top-5 ablation-sensitive types",
            "remove random matched top-5 count",
        ]
        labels = {
            "remove all T4": "T4",
            "remove all T5": "T5",
            "remove T4+T5": "T4+T5",
            "remove top-5 ablation-sensitive types": "top-5",
            "remove random matched top-5 count": "random",
        }
        df = df[df["ablation"].isin(keep)].copy()
        df["short_label"] = df["ablation"].map(labels)
        df = df.sort_values("drop_accuracy")
        colors = [PALETTE["flyvis"] if "T4" in a or "T5" in a else PALETTE["nuisance"] for a in df["ablation"]]
        xerr = np.vstack([df["drop_accuracy"] - df["drop_ci_low"], df["drop_ci_high"] - df["drop_accuracy"]])
        ax.barh(df["short_label"], df["drop_accuracy"], xerr=xerr, color=colors, edgecolor="black", linewidth=0.35, capsize=2)
        ax.set_xlabel("Accuracy drop")
        ax.set_xlim(0, max(0.015, float(df["drop_ci_high"].max()) * 1.3))
        clean_axis(ax, grid=True)
    ax.set_title("Group ablation drops", pad=4)
    save_pub(fig, out / "fig6_ablation")
    plt.close(fig)


def figure7_efficiency(out: Path, outputs: Path, warnings: list[str]) -> None:
    import matplotlib.pyplot as plt

    df = read_csv(outputs / "final_hardening" / "final_tables" / "table6_representation_metrics.csv", warnings)
    fig, ax = plt.subplots(figsize=(5.9, 3.4))
    panel_label(ax, "A", x=-0.11, y=1.05)
    if df.empty:
        ax.text(0.5, 0.5, "TODO", ha="center", va="center")
    else:
        graph = df[df["model"].str.contains("graph", case=False, na=False) | df["model"].str.contains("optic", case=False, na=False)]
        keep = df[df["model"].isin(["FlyVis", "pixel", "CNN", "local RNN"])].copy()
        if not graph.empty:
            keep = pd.concat(
                [
                    keep,
                    pd.DataFrame(
                        [
                            {
                                "model": "graph controls",
                                "retained_direction_bits_per_activity": graph["retained_direction_bits_per_activity"].mean(),
                            }
                        ]
                    ),
                ],
                ignore_index=True,
            )
        df = keep.sort_values("retained_direction_bits_per_activity", ascending=False)
        colors = [model_color(m) for m in df["model"]]
        ax.bar(np.arange(len(df)), df["retained_direction_bits_per_activity"], color=colors, edgecolor="black", linewidth=0.35)
        annotate_bars(ax, df["retained_direction_bits_per_activity"], fmt="{:.1f}")
        ax.set_xticks(np.arange(len(df)), df["model"], rotation=25, ha="right")
        ax.set_ylabel("Bits / activity")
        clean_axis(ax, grid=True)
    ax.set_title("Efficiency proxy favors FlyVis", pad=4)
    save_pub(fig, out / "fig7_efficiency")
    plt.close(fig)


def write_manifest(out: Path, warnings: list[str]) -> None:
    entries = [
        ("fig1_concept", "Concept schematic and stimulus examples", "Pilot v2 stimuli, metadata, hex coordinates", "Main Figure 1"),
        ("fig2_scale_matrix", "FlyVis train-scale/test-scale direction matrix", "flyvis_pilot_v2/tables/direction_scale_generalization.csv", "Main Figure 2"),
        ("fig3_controls", "FlyVis versus serious CNN and matched controls", "v4 bootstrap CI, serious CNN summary", "Main Figure 3"),
        ("fig4_feature_family", "Dynamic and static feature-family control breakdown", "final table2 feature-family results", "Main Figure 4"),
        ("fig5_temporal", "Temporal lesion and time-resolved decoding", "temporal lesion tables", "Main Figure 5"),
        ("fig6_ablation", "Cell-group ablation drops", "final table4 group ablation", "Main Figure 6"),
        ("fig7_efficiency", "Direction bits per activity proxy", "final table6 representation metrics", "Main Figure 7"),
    ]
    lines = ["# Publication Figure Manifest", ""]
    for name, shows, source, use in entries:
        lines.extend([f"## {name}", f"- Shows: {shows}", f"- Data source: `{source}`", f"- Manuscript use: {use}", ""])
    if warnings:
        lines.extend(["## Warnings", ""])
        lines.extend([f"- {w}" for w in warnings])
    else:
        lines.extend(["## Warnings", "", "- None."])
    (out / "figures_pub_manifest.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs-dir", default="scalebreak_flyvis/outputs")
    parser.add_argument("--out-dir", default="scalebreak_flyvis/outputs/final_hardening/figures_pub")
    args = parser.parse_args()
    set_pub_style()
    outputs = Path(args.outputs_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    warnings: list[str] = []
    figure1_concept(out, outputs, warnings)
    figure2_matrix(out, outputs, warnings)
    figure3_controls(out, outputs, warnings)
    figure4_feature_family(out, outputs, warnings)
    figure5_temporal(out, outputs, warnings)
    figure6_ablation(out, outputs, warnings)
    figure7_efficiency(out, outputs, warnings)
    write_manifest(out, warnings)
    print(f"Wrote publication figures to {out}")


if __name__ == "__main__":
    main()
