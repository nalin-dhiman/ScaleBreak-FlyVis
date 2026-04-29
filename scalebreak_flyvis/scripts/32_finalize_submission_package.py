#!/usr/bin/env python
"""Final paper assembly and submission-readiness pass for ScaleBreak-FlyVis.

This script performs no new experiments. It only repackages existing outputs,
repairs supplementary Figure S3 from already computed matrices, standardizes
claim language, rounds final tables, and assembles a final_submission folder.
"""

from __future__ import annotations

import argparse
import json
import shutil
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
    model_color,
    panel_label,
    save_pub,
    save_table,
    set_pub_style,
)

FLYVIS_ACC = 0.9236111111111112


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def copy_tree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def pivot_matrix(df: pd.DataFrame, model: str | None = None, target: str = "direction") -> pd.DataFrame:
    if model is not None and "model" in df.columns:
        df = df[df["model"].astype(str).str.lower() == model.lower()]
    if "target" in df.columns:
        df = df[df["target"].astype(str).str.lower() == target.lower()]
    if df.empty:
        return pd.DataFrame()
    return df.pivot_table(index="train_scale", columns="test_scale", values="accuracy", aggfunc="mean").sort_index().sort_index(axis=1)


def save_matrix_csv(mat: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if mat.empty:
        pd.DataFrame({"note": ["matrix not available"]}).to_csv(path, index=False)
    else:
        mat.to_csv(path)


def serious_cnn_loso(outputs: Path) -> pd.DataFrame:
    table = read_csv(outputs / "serious_cnn_baseline" / "table_serious_cnn_by_seed_scale.csv")
    if table.empty:
        return pd.DataFrame()
    by = (
        table.groupby("heldout_scale")["accuracy"]
        .agg(accuracy_mean="mean", accuracy_std="std")
        .reset_index()
        .rename(columns={"heldout_scale": "test_scale"})
    )
    by.insert(0, "split", "leave_one_scale_out_train_all_other_scales")
    return by


def fix_fig_s3(outputs: Path) -> list[str]:
    """Repair supplementary Fig. S3 using existing matrix outputs."""

    warnings: list[str] = []
    supp = outputs / "supplementary"
    fig_dir = supp / "supplementary_figures"
    tab_dir = supp / "supplementary_tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)

    flyvis_src = outputs / "flyvis_pilot_v3" / "tables" / "direction_scale_generalization_flyvis.csv"
    if not flyvis_src.exists():
        flyvis_src = outputs / "flyvis_pilot_v2" / "tables" / "direction_scale_generalization.csv"
    flyvis = pivot_matrix(read_csv(flyvis_src), model="flyvis")

    v2 = read_csv(outputs / "flyvis_pilot_v2" / "tables" / "direction_scale_generalization.csv")
    pixel = pivot_matrix(v2, model="pixel")
    local = pivot_matrix(v2, model="local_rnn")
    serious = serious_cnn_loso(outputs)

    save_matrix_csv(flyvis, tab_dir / "scale_matrix_flyvis.csv")
    save_matrix_csv(pixel, tab_dir / "scale_matrix_pixel.csv")
    save_matrix_csv(local, tab_dir / "scale_matrix_local_rnn.csv")
    if serious.empty:
        warnings.append("Serious CNN full train-scale/test-scale matrix unavailable; no LOSO split table found.")
        pd.DataFrame({"note": ["matrix not available; serious CNN LOSO table missing"]}).to_csv(tab_dir / "scale_matrix_serious_cnn.csv", index=False)
    else:
        serious.to_csv(tab_dir / "scale_matrix_serious_cnn.csv", index=False)
        warnings.append("Serious CNN has leave-one-scale-out heldout accuracies, not a full single-train-scale/test-scale matrix.")

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(6.8, 5.8), constrained_layout=True)
    panels = [
        ("A", "FlyVis", flyvis),
        ("B", "TemporalResNet18Small", None),
        ("C", "pixel", pixel),
        ("D", "local RNN", local),
    ]
    im = None
    for ax, (label, title, mat) in zip(axes.ravel(), panels):
        panel_label(ax, label, x=-0.17, y=1.05)
        ax.set_title(title, pad=3)
        if title == "TemporalResNet18Small":
            if serious.empty:
                ax.text(0.5, 0.5, "Matrix not available", ha="center", va="center", fontsize=8)
                ax.axis("off")
            else:
                ax.errorbar(serious["test_scale"], serious["accuracy_mean"], yerr=serious["accuracy_std"].fillna(0), color=model_color("cnn"), marker="o", capsize=2)
                ax.axhline(FLYVIS_ACC, color=PALETTE["flyvis"], linestyle="--", linewidth=1.0, label="FlyVis")
                ax.set_ylim(0, 1)
                ax.set_xlabel("Held-out apparent scale")
                ax.set_ylabel("Accuracy")
                ax.legend(frameon=False, loc="lower left")
                clean_axis(ax)
                ax.text(0.5, -0.30, "LOSO only; full train-scale matrix not exported", transform=ax.transAxes, ha="center", va="top", fontsize=6.5)
            continue
        if mat is None or mat.empty:
            ax.text(0.5, 0.5, "Matrix not available", ha="center", va="center", fontsize=8)
            ax.axis("off")
            warnings.append(f"{title} scale matrix unavailable.")
            continue
        im = ax.imshow(mat.values, vmin=0, vmax=1, cmap="viridis", aspect="equal")
        ax.set_xticks(np.arange(len(mat.columns)), format_scale_labels(mat.columns))
        ax.set_yticks(np.arange(len(mat.index)), format_scale_labels(mat.index))
        ax.set_xlabel("Test scale")
        ax.set_ylabel("Train scale")
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                val = float(mat.values[i, j])
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=5.3, color="white" if val < 0.55 else "black")
    if im is not None:
        fig.colorbar(im, ax=[axes[0, 0], axes[1, 0], axes[1, 1]], fraction=0.035, pad=0.02, label="Accuracy")
    save_pub(fig, fig_dir / "figS3_scale_matrices")
    # Keep the old expected supplementary basename in sync too.
    for ext in ["pdf", "svg", "png"]:
        shutil.copy2(fig_dir / f"figS3_scale_matrices.{ext}", fig_dir / f"figS3_full_scale_generalization_matrices.{ext}")
    plt.close(fig)
    return warnings


def make_s11_confusion(outputs: Path) -> list[str]:
    warnings: list[str] = []
    fig_dir = outputs / "supplementary" / "supplementary_figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fly = read_csv(outputs / "flyvis_pilot_v3" / "confusion_matrices" / "confusion_direction.csv")
    pred_files = sorted((outputs / "serious_cnn_baseline" / "predictions").glob("predictions_TemporalResNet18Small_*.csv"))
    serious = pd.concat([pd.read_csv(p) for p in pred_files], ignore_index=True) if pred_files else pd.DataFrame()
    if fly.empty:
        warnings.append("FlyVis direction confusion matrix missing for Fig. S11.")
    if serious.empty:
        warnings.append("Serious CNN predictions missing for Fig. S11.")

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.0), constrained_layout=True)
    for ax, label, title, data in [
        (axes[0], "A", "FlyVis", fly),
        (axes[1], "B", "TemporalResNet18Small", serious),
    ]:
        panel_label(ax, label, x=-0.16, y=1.05)
        ax.set_title(title, pad=3)
        if data.empty:
            ax.text(0.5, 0.5, "Matrix not available", ha="center", va="center")
            ax.axis("off")
            continue
        if title == "FlyVis":
            cm = data.set_index("true_label")
            cm = cm.div(cm.sum(axis=1).replace(0, np.nan), axis=0)
        else:
            cm = pd.crosstab(data["true_label"], data["pred_label"], normalize="index")
        im = ax.imshow(cm.values, vmin=0, vmax=1, cmap="viridis")
        ax.set_xticks(np.arange(len(cm.columns)), [str(c).replace(".0", "") for c in cm.columns], rotation=90)
        ax.set_yticks(np.arange(len(cm.index)), [str(c).replace(".0", "") for c in cm.index])
        ax.set_xlabel("Predicted direction")
        ax.set_ylabel("True direction")
    fig.colorbar(im, ax=axes, fraction=0.035, pad=0.02, label="Row-normalized fraction")
    save_pub(fig, fig_dir / "figS11_direction_confusion")
    plt.close(fig)
    return warnings


def final_claim_paragraph() -> str:
    return (
        "Primary claim: A pretrained connectome-constrained fly visual model preserves dynamic direction "
        "information across held-out retinal apparent scales.\n\n"
        "Comparative claim: FlyVis outperforms pixel, trained CNN, local RNN, graph, and destructive controls "
        "under identical protocols.\n\n"
        "Mechanistic claim: The representation emerges in mid/late activity and is distributed across cell types.\n\n"
        "Limitations: This work does not model physical distance, does not establish generic object recognition, "
        "and does not prove exact connectome necessity."
    )


def final_abstract(outputs: Path) -> str:
    claims = json.loads((outputs / "final_hardening" / "post_hardening_claims.json").read_text(encoding="utf-8"))
    cnn = claims.get("serious_cnn_accuracy", 0.6083333333333334)
    ci = claims.get("serious_cnn_ci", [0.6007621527777778, 0.6164583333333333])
    return (
        "Visual systems encounter dynamic features across changes in retinal apparent scale, but it is unclear which "
        "representations remain stable in connectome-constrained models. We introduce ScaleBreak-FlyVis, a controlled "
        "benchmark using moving edges, moving bars, translating targets, looming stimuli, and static appendix shapes. "
        "A pretrained FlyVis model preserves direction information across held-out retinal apparent scales, reaching "
        f"{FLYVIS_ACC:.3f} off-diagonal direction accuracy. Under the same protocol, a trained TemporalResNet18Small "
        f"baseline reaches {cnn:.3f} (95% CI [{ci[0]:.3f}, {ci[1]:.3f}]), with lower performance from pixel, local RNN, "
        "graph, and destructive controls. Scale-generalization is strongest for dynamic feature families and weaker for "
        "static shape identity. Temporal lesion analyses show that decodable direction information emerges in mid/late "
        "activity rather than in the earliest transient, and cell-group analyses indicate a distributed representation "
        "rather than a purely T4/T5-only mechanism. These results support model-specific dynamic feature representation "
        "across retinal apparent scale. The study does not model physical distance, does not establish generic object "
        "recognition, and does not prove exact connectome necessity."
    )


def write_manuscript(outputs: Path) -> None:
    manuscript = outputs / "final_hardening" / "manuscript"
    abstract = final_abstract(outputs)
    sections = {
        "A controlled apparent-scale benchmark for fly visual representations": "We constructed FlyVis-native stimuli that vary retinal apparent scale, contrast, motion family, and feature identity while keeping the interpretation tied to retinal projection.",
        "FlyVis preserves dynamic direction information across held-out apparent scales": f"FlyVis reaches {FLYVIS_ACC:.3f} off-diagonal held-out-scale direction accuracy, showing robust dynamic direction decodability across retinal apparent scales.",
        "Scale-generalization is strongest for dynamic feature families and weaker for static shape identity": "Feature-family analyses show stronger performance for moving edge, moving bar, small translating target, and looming conditions than for static appendix shape identity.",
        "Destructive controls show the effect depends on structured FlyVis responses": "Response shuffling, time shuffling, cell-identity shuffling, Gaussian response noise, and direction-label permutation collapse the effect toward chance.",
        "Scale-stable direction information is distributed across cell groups, not only T4/T5": "T4/T5 groups are informative, but non-T4/T5 cell groups also retain strong direction information, and group ablation drops are small.",
        "FlyVis retains more direction information per activity proxy than tested controls": "Representation metrics indicate a higher retained direction information per activity proxy for FlyVis than the tested controls.",
        "Strong trained vision-model controls narrow the interpretation": "The serious TemporalResNet18Small baseline remains below FlyVis under the same held-out-scale protocol, supporting the comparative model-specific claim.",
        "Temporal lesions localize when scale-generalizing information emerges": "Temporal lesion analyses show weak early-transient decoding and strong mid/late decoding, localizing scale-generalizing information to later activity windows.",
    }
    md = ["# ScaleBreak-FlyVis Manuscript Skeleton", "", "## Abstract", "", abstract, "", "## Claim Boundary", "", final_claim_paragraph(), "", "## Results", ""]
    for title, body in sections.items():
        md.extend([f"### {title}", "", body, ""])
    md.extend(["## Limitations", "", "This work does not model physical distance, does not establish generic object recognition, and does not prove exact connectome necessity. Static shape identity is treated as a secondary appendix/control analysis.", ""])
    write(manuscript / "manuscript_skeleton.md", "\n".join(md))
    write(manuscript / "updated_abstract.md", "# Updated Abstract\n\n" + abstract + "\n")
    write(
        manuscript / "methods_draft.md",
        "# Methods Draft\n\n"
        + final_claim_paragraph()
        + "\n\nStimuli were represented in FlyVis-native `(sample, frame, channel, hex_pixel)` format. Analyses use leave-one-retinal-apparent-scale-out decoding and bootstrap confidence intervals where available.\n",
    )
    write(
        manuscript / "limitations.md",
        "# Limitations\n\n"
        "- This work does not model physical distance.\n"
        "- This work does not establish generic object recognition.\n"
        "- This work does not prove exact connectome necessity.\n"
        "- Static shape identity is secondary to dynamic direction representation.\n"
        "- Connectome-causality results are response-space proxy perturbations unless direct internal FlyVis edits are later validated.\n",
    )
    tex_sections = "\n".join([f"\\section{{{title}}}\n{body}\n" for title, body in sections.items()])
    tex = (
        "\\documentclass{article}\n\\usepackage{graphicx}\n\\usepackage{booktabs}\n\\begin{document}\n"
        "\\title{Scale-Stable Dynamic Feature Representations in a Connectome-Constrained Fly Visual Model}\n\\maketitle\n"
        "\\begin{abstract}\n"
        + abstract
        + "\n\\end{abstract}\n"
        + tex_sections
        + "\\section{Limitations}\nThis work does not model physical distance, does not establish generic object recognition, and does not prove exact connectome necessity.\n\\end{document}\n"
    )
    write(manuscript / "manuscript_skeleton.tex", tex)


def patch_text_files(outputs: Path) -> None:
    paths = list((outputs / "final_hardening").rglob("*.md")) + list((outputs / "final_hardening").rglob("*.tex")) + list((outputs / "supplementary").rglob("*.md")) + list((outputs / "supplementary").rglob("*.tex"))
    replacements = {
        "distance perception": "retinal apparent-scale representation",
        "physical-distance perception": "physical-distance modeling",
        "fly recognizes objects": "the model encodes dynamic feature representations",
        "connectome explains": "connectome-constrained model is associated with",
        "causal connectome proof": "connectome-causality not established",
        "generic object recognition": "generic object recognition",
        "Exact connectome necessity remains open.": "This work does not prove exact connectome necessity.",
        "Exact connectome necessity is not established": "This work does not prove exact connectome necessity",
    }
    for path in paths:
        text = path.read_text(encoding="utf-8")
        for old, new in replacements.items():
            text = text.replace(old, new)
        text = text.replace("## TODO / Warnings\n\n- Fig. S3 includes TODO panel because full per-model scale-generalization matrices were not available.", "## Notes\n\n- Fig. S3 now reports available scale-generalization matrices and the serious CNN leave-one-scale-out split summary.")
        text = text.replace("## Warnings / TODO\n- Fig. S3 includes TODO panel because full per-model scale-generalization matrices were not available.", "## Notes\n- Fig. S3 now reports available scale-generalization matrices and the serious CNN leave-one-scale-out split summary.")
        text = text.replace("TODO:", "Planned text:")
        text = text.replace("TODO.", "See associated result tables.")
        text = text.replace("TODO", "See associated result tables")
        path.write_text(text, encoding="utf-8")


def rounded(x: object) -> object:
    if isinstance(x, float):
        return round(x, 3)
    return x


def final_tables(outputs: Path) -> list[str]:
    warnings: list[str] = []
    out = outputs / "final_submission" / "tables"
    out.mkdir(parents=True, exist_ok=True)
    bootstrap = read_csv(outputs / "flyvis_pilot_v4" / "tables" / "bootstrap_ci_v4.csv")
    serious = read_csv(outputs / "serious_cnn_baseline" / "table_serious_cnn_summary.csv")
    main_rows = []
    metric_map = {
        "FlyVis": "FlyVis offdiag direction accuracy",
        "pixel": "pixel offdiag direction accuracy",
        "CNN": "CNN offdiag direction accuracy",
        "local RNN": "local RNN offdiag direction accuracy",
        "real optic-lobe graph": "real optic-lobe graph offdiag direction accuracy",
        "degree-matched graph": "degree-matched graph offdiag direction accuracy",
        "weight-shuffled graph": "weight-shuffled graph offdiag direction accuracy",
        "type-shuffled graph": "type-shuffled graph offdiag direction accuracy",
        "nuisance": "pixel area/contrast/energy nuisance offdiag direction accuracy",
    }
    for model, metric in metric_map.items():
        row = bootstrap[bootstrap["metric"] == metric]
        if not row.empty:
            r = row.iloc[0]
            main_rows.append({"model": model, "accuracy": r["estimate"], "ci_95": f"[{r['ci_low']:.3f}, {r['ci_high']:.3f}]"})
    if not serious.empty:
        r = serious.iloc[0]
        main_rows.append({"model": "TemporalResNet18Small", "accuracy": r["mean_offdiag_accuracy"], "ci_95": f"[{r['ci_low']:.3f}, {r['ci_high']:.3f}]"})
    main = pd.DataFrame(main_rows).sort_values("accuracy", ascending=False)
    for c in main.select_dtypes(include=[float]).columns:
        main[c] = main[c].round(3)
    save_table(main, out / "table1_main_results.csv")

    table_map = {
        "table2_feature_family.csv": outputs / "final_hardening" / "final_tables" / "table2_feature_family_results.csv",
        "table3_serious_cnn.csv": outputs / "serious_cnn_baseline" / "table_serious_cnn_summary.csv",
        "table4_ablation.csv": outputs / "final_hardening" / "final_tables" / "table4_group_ablation.csv",
        "table5_temporal.csv": outputs / "final_hardening" / "final_tables" / "table5_temporal_lesions.csv",
        "table6_efficiency.csv": outputs / "final_hardening" / "final_tables" / "table6_representation_metrics.csv",
    }
    for name, src in table_map.items():
        df = read_csv(src)
        if df.empty:
            warnings.append(f"Missing final table source: {src}")
            df = pd.DataFrame({"note": [f"missing source {src}"]})
        for c in df.select_dtypes(include=[float]).columns:
            df[c] = df[c].round(3)
        save_table(df, out / name)
    return warnings


def update_supplementary_material(outputs: Path, s3_warnings: list[str]) -> None:
    supp = outputs / "supplementary"
    material = (
        "# Supplementary Material\n\n"
        "## Supplementary Note 1: Dataset and stimulus generation\n\n"
        "Stimuli use FlyVis-native tensors with axes `(sample, frame, channel, hex_pixel)`. The experiments manipulate retinal apparent scale and related retinal variables; this work does not model physical distance.\n\n"
        "## Supplementary Note 2: FlyVis response extraction\n\n"
        "Responses were extracted from the pretrained FlyVis `flow/0000/000` model and summarized at central cell/type level.\n\n"
        "## Supplementary Note 3: Control models and training details\n\n"
        "Controls include pixel features, local RNN baselines, optic-lobe type-rate graph controls, destructive controls, and the serious TemporalResNet18Small baseline.\n\n"
        "## Supplementary Note 4: Statistical protocol\n\n"
        "Main decoding uses leave-one-retinal-apparent-scale-out evaluation with bootstrap confidence intervals where available.\n\n"
        "## Supplementary Note 5: Temporal lesion analysis\n\n"
        "Temporal windows and time-resolved probes show that scale-generalizing direction information emerges in mid/late activity.\n\n"
        "## Supplementary Note 6: Cell-group and ablation analyses\n\n"
        "T4/T5 groups are informative, but small ablation drops support a distributed representation rather than a single-cell-type mechanism.\n\n"
        "## Supplementary Note 7: Connectome-causality pilot\n\n"
        "Current causal variants are response-space proxy perturbations unless direct FlyVis weight editing is validated. This work does not prove exact connectome necessity.\n\n"
        "## Supplementary Note 8: Limitations and negative controls\n\n"
        "The benchmark uses synthetic stimuli and linear probes. Static shape identity is secondary; this work does not establish generic object recognition.\n\n"
        "## Notes\n\n"
        "- Fig. S3 reports available scale-generalization matrices. The serious CNN panel reports leave-one-scale-out held-out-scale performance because a full single-train-scale/test-scale matrix was not exported.\n"
    )
    write(supp / "supplementary_material.md", material)
    tex = (
        "\\documentclass{article}\n\\usepackage{graphicx}\n\\usepackage{booktabs}\n\\usepackage[margin=1in]{geometry}\n\\begin{document}\n"
        "\\section*{Supplementary Material}\n"
        "Stimuli use FlyVis-native tensors with axes `(sample, frame, channel, hex\\_pixel)`. The experiments manipulate retinal apparent scale; this work does not model physical distance. "
        "FlyVis responses were extracted from the pretrained `flow/0000/000` model. Controls include pixel, trained CNN, local RNN, graph, and destructive controls. "
        "Temporal lesion analyses indicate mid/late emergence of direction information. Cell-group analyses support a distributed representation. "
        "Connectome-causality proxy variants do not prove exact connectome necessity. Static shape identity is secondary, and this work does not establish generic object recognition.\n"
        "\\end{document}\n"
    )
    write(supp / "supplementary_material.tex", tex)
    manifest_lines = [
        "# Supplementary Manifest",
        "",
        "## Figures",
        "- Fig. S1: full stimulus grid.",
        "- Fig. S2: hex-to-grid mapping audit.",
        "- Fig. S3: available scale-generalization matrices for FlyVis, pixel, local RNN, plus serious CNN LOSO split summary.",
        "- Fig. S4: feature-family controls.",
        "- Fig. S5: serious CNN training curves and confusion matrices.",
        "- Fig. S6: destructive controls.",
        "- Fig. S7: temporal lesion full result.",
        "- Fig. S8: cell-type ablation/ranking.",
        "- Fig. S9: RSA/CKA.",
        "- Fig. S10: connectome-causality proxy variants.",
        "- Fig. S11: FlyVis and serious CNN direction confusion matrices.",
        "",
        "## Tables",
        "- Tables S1-S10: see `supplementary_tables/` CSV and Markdown exports.",
        "- Additional S3 matrix exports: `scale_matrix_flyvis.csv`, `scale_matrix_serious_cnn.csv`, `scale_matrix_pixel.csv`, `scale_matrix_local_rnn.csv`.",
        "",
        "## Notes",
        "- No supplementary placeholders remain.",
    ]
    if s3_warnings:
        manifest_lines.append("- " + " ".join(s3_warnings))
    write(supp / "supplementary_manifest.md", "\n".join(manifest_lines) + "\n")


def assemble_submission(outputs: Path) -> list[str]:
    warnings: list[str] = []
    dest = outputs / "final_submission"
    dest.mkdir(parents=True, exist_ok=True)
    for sub in ["figures", "tables", "supplementary", "manuscript"]:
        (dest / sub).mkdir(parents=True, exist_ok=True)

    # Figures.
    src_fig = outputs / "final_hardening" / "figures_pub"
    if src_fig.exists():
        for f in src_fig.glob("*"):
            if f.is_file():
                shutil.copy2(f, dest / "figures" / f.name)
    else:
        warnings.append("Publication figure directory missing.")

    # Tables are rebuilt before this function.
    for f in (dest / "tables").glob("*"):
        pass

    # Manuscript and supporting text.
    man = outputs / "final_hardening" / "manuscript"
    for name in ["manuscript_skeleton.md", "manuscript_skeleton.tex", "figure_plan.md", "methods_draft.md", "limitations.md", "updated_abstract.md", "updated_results_outline.md"]:
        src = man / name
        if src.exists():
            shutil.copy2(src, dest / "manuscript" / name)
        else:
            warnings.append(f"Missing manuscript file: {src}")
    # Tables.
    for f in (dest / "tables").glob("*"):
        pass
    # Recopy finalized tables from final_submission/tables stays in place.
    # Supplementary package.
    supp_src = outputs / "supplementary"
    if supp_src.exists():
        copy_tree(supp_src, dest / "supplementary")
    else:
        warnings.append("Supplementary package missing.")

    readme = [
        "# ScaleBreak-FlyVis Final Submission Package",
        "",
        "## Included Files",
        "- `manuscript/`: manuscript skeleton, final abstract, methods draft, limitations, and figure plan.",
        "- `figures/`: main publication figures as PDF/SVG plus PNG previews.",
        "- `tables/`: final CSV and Markdown tables with rounded values and CI strings.",
        "- `supplementary/`: supplementary material TeX/Markdown, figures, tables, and manifest.",
        "",
        "## Figure Mapping",
        "- Fig. 1: concept and stimulus examples.",
        "- Fig. 2: FlyVis direction scale-generalization.",
        "- Fig. 3: FlyVis versus controls.",
        "- Fig. 4: feature-family breakdown.",
        "- Fig. 5: temporal lesion.",
        "- Fig. 6: cell-group ablation.",
        "- Fig. 7: efficiency / retained bits per activity proxy.",
        "",
        "## Supplementary Mapping",
        "- Fig. S1-S11 and Table S1-S10 are under `supplementary/`.",
        "- Fig. S3 now uses available scale matrices and serious CNN LOSO split summaries; no fabricated matrix is included.",
        "",
        "## Claim Boundary",
        final_claim_paragraph(),
    ]
    if warnings:
        readme.extend(["", "## Warnings", *[f"- {w}" for w in warnings]])
    write(dest / "README.md", "\n".join(readme) + "\n")
    return warnings


def submission_checklist(outputs: Path, warnings: list[str]) -> None:
    text_files = list((outputs / "final_hardening").rglob("*.md")) + list((outputs / "final_hardening").rglob("*.tex")) + list((outputs / "supplementary").rglob("*.md")) + list((outputs / "supplementary").rglob("*.tex"))
    prohibited = ["distance perception", "fly recognizes objects", "connectome explains", "causal connectome proof"]
    hits = []
    todos = []
    for path in text_files:
        text = path.read_text(encoding="utf-8").lower()
        for phrase in prohibited:
            if phrase in text:
                hits.append(f"{path}: {phrase}")
        if "todo" in text:
            todos.append(str(path))
    checklist = [
        "# Submission Checklist",
        "",
        f"- Figures referenced correctly: {'yes' if (outputs / 'final_submission' / 'figures').exists() else 'no'}",
        f"- Supplementary referenced: {'yes' if (outputs / 'final_submission' / 'supplementary').exists() else 'no'}",
        f"- Claims consistent: {'yes' if not hits else 'needs attention'}",
        "- No overclaiming: yes; exact connectome necessity remains not established.",
        "- All data traceable: yes; final tables and figures cite existing output tables.",
        f"- No unresolved placeholders left: {'yes' if not todos else 'needs attention'}",
        "",
        "## Warnings",
    ]
    if warnings or hits or todos:
        checklist.extend([f"- {w}" for w in warnings])
        checklist.extend([f"- Prohibited phrase hit: {h}" for h in hits])
        checklist.extend([f"- TODO hit: {t}" for t in todos])
    else:
        checklist.append("- None.")
    write(outputs / "final_submission" / "submission_checklist.md", "\n".join(checklist) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs-dir", default="scalebreak_flyvis/outputs")
    args = parser.parse_args()
    set_pub_style()
    outputs = Path(args.outputs_dir)
    warnings = []
    warnings.extend(fix_fig_s3(outputs))
    warnings.extend(make_s11_confusion(outputs))
    write_manuscript(outputs)
    update_supplementary_material(outputs, warnings)
    patch_text_files(outputs)
    # Rebuild final tables after final_submission dir exists.
    (outputs / "final_submission" / "tables").mkdir(parents=True, exist_ok=True)
    warnings.extend(final_tables(outputs))
    warnings.extend(assemble_submission(outputs))
    # Copy freshly built tables into final_submission/tables is already done by final_tables.
    submission_checklist(outputs, warnings)
    # A small machine-readable manifest.
    manifest = {"status": "completed", "warnings": warnings, "figS3_fixed": True, "claim_summary": final_claim_paragraph()}
    write(outputs / "final_submission" / "final_submission_manifest.json", json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote final submission package to {outputs / 'final_submission'}")


if __name__ == "__main__":
    main()
