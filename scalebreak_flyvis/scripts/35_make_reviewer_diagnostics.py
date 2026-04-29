#!/usr/bin/env python
"""Build reviewer-facing diagnostics from existing ScaleBreak-FlyVis outputs.

This script does not run new experiments. It consolidates existing predictions
and tables into diagnostics that address common reviewer questions:

* LOSO held-out-scale accuracies, distinct from pairwise train-one-scale matrices.
* Pixel/nuisance feature definitions.
* Graph-control interpretation when variants produce identical aggregate scores.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


COLORS = {
    "FlyVis": "#1f77b4",
    "TemporalResNet18Small": "#ff7f0e",
    "pixel": "#2ca02c",
    "CNN": "#ff7f0e",
    "local RNN": "#9467bd",
    "nuisance": "#7f7f7f",
}


def save_table(df: pd.DataFrame, path: Path) -> None:
    """Save CSV plus Markdown sidecar."""

    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    try:
        md = df.to_markdown(index=False)
    except Exception:
        md = "```csv\n" + df.to_csv(index=False) + "```\n"
    path.with_suffix(".md").write_text(md, encoding="utf-8")


def build_loso_table(outputs: Path) -> pd.DataFrame:
    """Construct model x held-out-scale LOSO accuracy table from existing outputs."""

    pred_path = outputs / "flyvis_pilot_v3" / "tables" / "direction_loso_predictions_all_models.csv"
    if not pred_path.exists():
        raise FileNotFoundError(f"Missing LOSO prediction table: {pred_path}")
    pred = pd.read_csv(pred_path)
    rename = {
        "flyvis": "FlyVis",
        "pixel": "pixel",
        "cnn": "CNN",
        "local_rnn": "local RNN",
        "nuisance_area_contrast_energy": "nuisance",
    }
    pred = pred[pred["model"].isin(rename)]
    rows: list[dict[str, object]] = []
    for (model, scale), sub in pred.groupby(["model", "heldout_scale"]):
        rows.append(
            {
                "model": rename[model],
                "heldout_scale": float(scale),
                "accuracy": float(sub["correct"].astype(bool).mean()),
                "n_predictions": int(len(sub)),
                "source": str(pred_path),
            }
        )

    serious_path = outputs / "serious_cnn_baseline" / "table_serious_cnn_by_seed_scale.csv"
    if serious_path.exists():
        serious = pd.read_csv(serious_path)
        for scale, sub in serious.groupby("heldout_scale"):
            rows.append(
                {
                    "model": "TemporalResNet18Small",
                    "heldout_scale": float(scale),
                    "accuracy": float(sub["accuracy"].mean()),
                    "n_predictions": "5 seeds x 360 held-out trials",
                    "source": str(serious_path),
                }
            )
    return pd.DataFrame(rows)


def plot_loso_heatmap(df: pd.DataFrame, out_base: Path) -> None:
    """Plot LOSO held-out-scale accuracies as a compact model x scale heatmap."""

    import matplotlib as mpl
    import matplotlib.pyplot as plt

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.linewidth": 0.75,
        }
    )

    order = ["FlyVis", "TemporalResNet18Small", "pixel", "CNN", "local RNN", "nuisance"]
    pivot = (
        df.pivot_table(index="model", columns="heldout_scale", values="accuracy", aggfunc="mean")
        .reindex(order)
        .dropna(how="all")
        .sort_index(axis=1)
    )
    fig, ax = plt.subplots(figsize=(6.3, 2.7))
    im = ax.imshow(pivot.values, vmin=0, vmax=1, cmap="viridis", aspect="auto")
    ax.set_xticks(np.arange(len(pivot.columns)), [str(int(c)) for c in pivot.columns])
    ax.set_yticks(np.arange(len(pivot.index)), pivot.index)
    ax.set_xlabel("Held-out apparent scale")
    ax.set_ylabel("Model")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.values[i, j]
            if np.isfinite(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=6, color="white" if val < 0.55 else "black")
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Accuracy")
    ax.set_title("LOSO held-out-scale accuracy, train on all other scales", pad=4)
    fig.tight_layout()
    out_base.parent.mkdir(parents=True, exist_ok=True)
    for ext in ["pdf", "svg", "png"]:
        kwargs = {"dpi": 240} if ext == "png" else {}
        fig.savefig(out_base.with_suffix(f".{ext}"), bbox_inches="tight", **kwargs)
    plt.close(fig)


def feature_definition_table() -> pd.DataFrame:
    """Describe the exact baseline feature sets from the current code."""

    return pd.DataFrame(
        [
            {
                "baseline": "pixel",
                "features": "mean frame flattened; max frame flattened; mean absolute temporal difference; temporal-difference standard deviation; total stimulus energy",
                "normalization_decoder": "StandardScaler inside the logistic-regression probe",
                "interpretation": "Strong non-neural baseline with access to raw retinal movie summaries and simple temporal-difference statistics.",
            },
            {
                "baseline": "nuisance area/contrast/energy",
                "features": "area_pixels; edge_length_pixels; contrast",
                "normalization_decoder": "StandardScaler inside the logistic-regression probe",
                "interpretation": "Tests whether decoding is explained by scalar stimulus size/edge/contrast metadata rather than representation.",
            },
            {
                "baseline": "TemporalResNet18Small",
                "features": "10 hex-to-grid temporal channels: five frames, mean frame, max frame, early-middle difference, middle-late difference, motion energy",
                "normalization_decoder": "trained end-to-end classifier after deterministic 24x24 hex-to-grid projection",
                "interpretation": "Manuscript-grade trained artificial-vision control; projection may lose hex-native information.",
            },
        ]
    )


def graph_audit_table(outputs: Path) -> pd.DataFrame:
    """Summarize graph-control aggregate scores and the interpretation of identical values."""

    path = outputs / "flyvis_pilot_v4" / "tables" / "bootstrap_ci_v4.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing graph-control bootstrap table: {path}")
    df = pd.read_csv(path)
    wanted = [
        "real optic-lobe graph offdiag direction accuracy",
        "degree-matched graph offdiag direction accuracy",
        "weight-shuffled graph offdiag direction accuracy",
        "type-shuffled graph offdiag direction accuracy",
    ]
    sub = df[df["metric"].isin(wanted)].copy()
    sub["interpretation"] = (
        "Simplified type-rate control; identical aggregate scores indicate this control family is insensitive "
        "to these graph shuffles under the current readout and should not be treated as evidence for or against exact connectome causality."
    )
    return sub


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs-dir", default="scalebreak_flyvis/outputs")
    parser.add_argument("--draft-dir", default="scalebreak_flyvis/ScaleBreak_FlyVis_full_submission_draft/ScaleBreak_FlyVis_submission_draft")
    args = parser.parse_args()

    outputs = Path(args.outputs_dir)
    draft = Path(args.draft_dir)
    out_tables = draft / "supplementary_tables"
    out_figs = draft / "supplementary_figures"

    loso = build_loso_table(outputs)
    save_table(loso, out_tables / "tableS11_loso_by_scale.csv")
    plot_loso_heatmap(loso, out_figs / "figS12_loso_by_scale")

    save_table(feature_definition_table(), out_tables / "tableS12_baseline_feature_definitions.csv")
    save_table(graph_audit_table(outputs), out_tables / "tableS13_graph_control_audit.csv")

    final_supp = outputs / "final_submission" / "supplementary"
    if final_supp.exists():
        for path in [out_tables / "tableS11_loso_by_scale.csv", out_tables / "tableS11_loso_by_scale.md", out_tables / "tableS12_baseline_feature_definitions.csv", out_tables / "tableS12_baseline_feature_definitions.md", out_tables / "tableS13_graph_control_audit.csv", out_tables / "tableS13_graph_control_audit.md"]:
            target = final_supp / "supplementary_tables" / path.name
            target.write_bytes(path.read_bytes())
        for ext in ["pdf", "svg", "png"]:
            src = out_figs / f"figS12_loso_by_scale.{ext}"
            (final_supp / "supplementary_figures" / src.name).write_bytes(src.read_bytes())

    print(f"Wrote reviewer diagnostics to {draft}")


if __name__ == "__main__":
    main()
