#!/usr/bin/env python
"""Write manuscript scaffold for ScaleBreak-FlyVis final hardening."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd


TITLE_OPTIONS = [
    "Scale-Stable Dynamic Feature Representations in a Connectome-Constrained Fly Visual Model",
    "Apparent-Scale Generalization in a Connectome-Constrained Model of the Fly Visual System",
    "Dynamic Visual Features Remain Decodable Across Apparent Scale in FlyVis",
    "Representational Breakdown Across Retinal Scale in a Fly Connectome-Constrained Visual Network",
    "Scale-Dependent Dynamic Feature Coding in a Fly Visual System Model",
]


RESULT_HEADINGS = [
    "1. A controlled apparent-scale benchmark for fly visual representations",
    "2. FlyVis preserves dynamic direction information across held-out apparent scales",
    "3. Scale-generalization is strongest for dynamic feature families and weaker for static shape identity",
    "4. Destructive controls show the effect depends on structured FlyVis responses",
    "5. Scale-stable direction information is distributed across cell groups, not only T4/T5",
    "6. FlyVis retains more direction information per activity proxy than tested controls",
    "7. Strong trained vision-model controls narrow the interpretation",
    "8. Temporal lesions localize when scale-generalizing information emerges",
]


def read_table(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def metric_snippets(base: Path) -> dict[str, str]:
    v4 = base / "outputs" / "flyvis_pilot_v4" / "tables"
    final = base / "outputs" / "final_hardening"
    main = read_table(v4 / "table_v4_main_results.csv")
    strong = read_table(final / "strong_controls" / "table_strong_vision_controls_summary.csv")
    temporal = read_table(final / "temporal_lesions" / "table_temporal_lesion_accuracy.csv")
    snippets = {
        "flyvis_accuracy": "not yet available",
        "pixel_accuracy": "not yet available",
        "best_strong": "TODO: run strong trained controls",
        "best_temporal": "TODO: run temporal lesions",
    }
    if len(main):
        for model in ["FlyVis", "pixel"]:
            row = main.loc[main["model"] == model]
            if len(row):
                snippets[f"{model.lower()}_accuracy"] = f"{float(row['offdiag_accuracy'].iloc[0]):.3f}"
    if len(strong):
        row = strong.sort_values("mean_offdiag_accuracy", ascending=False).iloc[0]
        snippets["best_strong"] = f"{row['model']} at {float(row['mean_offdiag_accuracy']):.3f}"
    if len(temporal):
        row = temporal.sort_values("accuracy", ascending=False).iloc[0]
        snippets["best_temporal"] = f"{row['feature_variant']} at {float(row['accuracy']):.3f}"
    return snippets


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def manuscript_md(snips: dict[str, str]) -> str:
    titles = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(TITLE_OPTIONS))
    results = "\n\n".join(f"## {h}\n\nTODO: draft result text. Current relevant outputs are in `outputs/final_hardening/final_tables` and `outputs/final_hardening/final_figures`." for h in RESULT_HEADINGS)
    return f"""# Manuscript Skeleton

## Title Options

{titles}

## Abstract Draft

Animals encounter visual features at many retinal apparent scales as object size, position, and viewing geometry change. Here we introduce ScaleBreak-FlyVis, a controlled benchmark for asking which dynamic visual representations remain stable across apparent scale in a pretrained connectome-constrained model of the fly visual system. Using FlyVis responses to moving edges, moving bars, translating small targets, looming disks, and static appendix shapes, we find strong held-out-scale direction decoding from FlyVis activity (currently `{snips['flyvis_accuracy']}`), exceeding pixel and tested model controls in the completed Pilot v4 analyses. The effect is strongest for dynamic feature families and weaker for static shape identity. Destructive controls that shuffle responses, time, cell identities, or direction labels collapse the effect, while cell-group analyses suggest a distributed mechanism: T4/T5 responses are informative, but non-T4/T5 groups also retain direction information. These results support model-specific dynamic scale-generalization across retinal projection, not physical distance perception or generic object recognition. Exact connectome necessity remains unresolved and will require causal FlyVis variants and stronger graph-constrained model comparisons.

## Results

{results}

## Discussion

- This is not a claim about physical distance perception; all experiments manipulate retinal apparent scale.
- This is not generic object recognition; dynamic direction and motion-feature representations are the central result.
- Static shapes are weaker and should remain an appendix/control analysis.
- The completed analyses support FlyVis model-specific dynamic representation relative to tested controls.
- Exact connectome necessity remains open.
- Future work should include retrained and ablated FlyVis variants, connectome mask randomization inside FlyVis, stronger full optic-lobe graph models, naturalistic optic flow, and behavior-linked looming stimuli.
"""


def manuscript_tex() -> str:
    titles = "\\\\\n".join(TITLE_OPTIONS)
    sections = "\n".join("\\section{" + h.split(". ", 1)[1] + "}\nTODO.\n" for h in RESULT_HEADINGS)
    return rf"""\documentclass{{article}}
\usepackage{{graphicx}}
\usepackage{{booktabs}}
\title{{ScaleBreak-FlyVis Manuscript Skeleton}}
\author{{TODO}}
\begin{{document}}
\maketitle
\section*{{Title Options}}
{titles}
\begin{{abstract}}
TODO: use the Markdown abstract draft and keep claims limited to retinal apparent scale, dynamic direction generalization, model-specificity, and non-causal connectome interpretation.
\end{{abstract}}
{sections}
\section{{Discussion}}
TODO: include limitations: not physical distance, not generic object recognition, exact connectome necessity not established.
\end{{document}}
"""


def figure_plan() -> str:
    return """# Figure Plan

1. `fig1_concept_scale_retinal_projection`: apparent scale concept and stimulus examples.
2. `fig2_flyvis_direction_scale_generalization`: FlyVis train-scale/test-scale direction matrix.
3. `fig3_controls_hardening`: FlyVis vs pixel, trained controls, graph controls, destructive controls.
4. `fig4_feature_family_breakdown`: dynamic family and static appendix comparison.
5. `fig5_cell_group_ablation`: distributed cell-group mechanism and small group-ablation drops.
6. `fig6_temporal_lesion`: early/middle/late/full/time-resolved decoding.
7. `fig7_bits_per_activity`: retained direction information per activity proxy.
"""


def limitations() -> str:
    return """# Limitations

- The benchmark manipulates retinal apparent scale, not physical distance.
- The analyses do not establish generic object recognition.
- Exact connectome necessity is not established; FlyVis model-specificity is the supported claim.
- Static disk/square/triangle decoding is secondary and weaker than dynamic feature coding.
- Current ablations show distributed information but do not identify a unique causal cell group.
- Synthetic stimuli simplify natural optic flow and behavior-linked visual tasks.
- Linear probes measure decodability, not necessarily behaviorally used variables.
"""


def methods(snips: dict[str, str]) -> str:
    return f"""# Methods Draft

## Stimuli

Stimuli were generated in FlyVis-native format `(sample, frame, channel, hex_pixel)` and varied retinal apparent scale, direction/angle, contrast, feature family, and repeat seed. The main dynamic families were moving edge, moving bar, small translating target, and looming disk; static disk/square/triangle stimuli were retained as appendix controls.

## FlyVis Responses

The pretrained FlyVis model `flow/0000/000` was evaluated on the native hex-pixel movies. Central-cell responses had shape `(samples, frames, cell_types)` in the Pilot v2 response store.

## Decoding

The main task was direction decoding under leave-one-apparent-scale-out train/test splits. The principal metric was off-diagonal held-out-scale accuracy. FlyVis reached `{snips['flyvis_accuracy']}` in the current v4 table; pixel reached `{snips['pixel_accuracy']}`.

## Strong Vision Controls

Controls include temporal CNN, 3D CNN, ConvRNN, and ResNet-like temporal summary models trained from scratch on the same held-out-scale protocol. Current best trained-control status: `{snips['best_strong']}`.

## Temporal Lesions

FlyVis response features were extracted from early, middle, late, full-mean, peak, and temporal-bin windows. Current best temporal feature status: `{snips['best_temporal']}`.
"""


def reviewer_risks() -> str:
    risks = [
        ("Your CNN controls are weak.", "It could mean FlyVis only beats undertrained baselines.", "Add stronger trained controls; report architecture, seeds, CIs, and failures.", "Scale model capacity, train longer, and add augmentation/regularization sweeps."),
        ("FlyVis was trained for motion; this is expected.", "The result may reflect training objective rather than emergent connectome structure.", "Frame as model-specific dynamic representation, not surprise object recognition.", "Compare against trained motion-specialist artificial models and FlyVis ablations."),
        ("This does not prove connectome causality.", "A reviewer can reject causal connectome language.", "Final claims explicitly say exact connectome necessity is not established.", "Randomize or lesion connectome masks inside retrained FlyVis variants."),
        ("Pixel controls solve some stimuli.", "Moving edge and looming can be easy from retinal energy.", "Feature-family table reports where pixels match or exceed FlyVis.", "Design matched-energy stimuli and harder ambiguous-motion controls."),
        ("Scale is not distance.", "Physical distance claims would be overreach.", "All final language uses retinal apparent scale/projection.", "Add geometry appendix clarifying the mapping."),
        ("Linear probes may not reflect behavior.", "Decodability is not behavioral readout.", "State linear probes measure available information.", "Link to downstream readouts or behavior-inspired tasks."),
        ("Ablations are tiny, so mechanism is unclear.", "Small drops weaken mechanistic claims.", "Interpret as distributed representation.", "Perform causal model lesions/retraining and pathway-restricted probes."),
        ("Stimuli are synthetic.", "Generalization to natural vision is unknown.", "Treat as controlled benchmark.", "Add naturalistic optic-flow and behavior-linked looming sequences."),
    ]
    lines = ["# Reviewer Risks", ""]
    for i, (risk, why, defense, experiment) in enumerate(risks, 1):
        lines.extend([f"## {i}. {risk}", "", f"Why it matters: {why}", "", f"Current defense: {defense}", "", f"Additional experiment if needed: {experiment}", ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default="scalebreak_flyvis")
    parser.add_argument("--out-dir", default="scalebreak_flyvis/outputs/final_hardening/manuscript")
    args = parser.parse_args()

    base = Path(args.project_root)
    snips = metric_snippets(base)
    out = Path(args.out_dir)
    write(out / "manuscript_skeleton.md", manuscript_md(snips))
    write(out / "manuscript_skeleton.tex", manuscript_tex())
    write(out / "figure_plan.md", figure_plan())
    write(out / "limitations.md", limitations())
    write(out / "methods_draft.md", methods(snips))
    write(out / "reviewer_risks.md", reviewer_risks())
    write(
        out / "run_info.json",
        json.dumps({"status": "completed", "elapsed_seconds": time.time(), "python": sys.version}, indent=2),
    )
    print(f"Wrote manuscript scaffold to {out}")


if __name__ == "__main__":
    main()
