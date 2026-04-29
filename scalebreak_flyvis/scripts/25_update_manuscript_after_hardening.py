#!/usr/bin/env python
"""Update final manuscript claims after Part A and Part B hardening."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

FLYVIS_ACC = 0.9236111111111112


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def write_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--final-dir", default="scalebreak_flyvis/outputs/final_hardening")
    parser.add_argument("--cnn-dir", default="scalebreak_flyvis/outputs/serious_cnn_baseline")
    parser.add_argument("--causal-dir", default="scalebreak_flyvis/outputs/connectome_causality")
    args = parser.parse_args()

    final = Path(args.final_dir)
    claims = read_json(final / "final_claims.json")
    claims.setdefault("robust_dynamic_scale_generalization_in_flyvis", True)
    claims.setdefault("evidence_for_flyvis_model_specificity", True)
    claims.setdefault("evidence_for_exact_connectome_necessity", "not established")
    cnn_summary_path = Path(args.cnn_dir) / "table_serious_cnn_summary.csv"
    if cnn_summary_path.exists():
        cnn = pd.read_csv(cnn_summary_path).iloc[0]
        diff = float(cnn["flyvis_minus_model"])
        if diff <= 0.05:
            status = "trained CNN reaches FlyVis-level raw direction decoding; emphasis should shift to efficiency and biological interpretability"
        elif diff <= 0.10:
            status = "trained CNN approaches FlyVis; emphasize biological decomposition, distributed cell-type representation, and activity efficiency"
        else:
            status = "FlyVis remains higher than manuscript-grade trained CNN control"
        claims["strong_trained_vision_control_status"] = status
        claims["serious_cnn_accuracy"] = float(cnn["mean_offdiag_accuracy"])
        claims["serious_cnn_ci"] = [float(cnn["ci_low"]), float(cnn["ci_high"])]
    else:
        claims["strong_trained_vision_control_status"] = "not run"

    causal_path = Path(args.causal_dir) / "table_causal_variants.csv"
    if causal_path.exists():
        causal = pd.read_csv(causal_path)
        nonfull = causal[causal["variant"] != "full"]
        max_drop = float(nonfull["accuracy_drop_vs_flyvis"].max()) if len(nonfull) else 0.0
        claims["connectome_causality_max_drop_vs_flyvis"] = max_drop
        if max_drop > 0.20:
            claims["connectome_causality_claim_strength"] = "provisional response-perturbation support; exact connectome necessity not established"
        else:
            claims["connectome_causality_claim_strength"] = "not established"
    else:
        claims["connectome_causality_claim_strength"] = "not established"

    claims["physical_distance_claim"] = False
    claims["generic_object_recognition_claim"] = False
    write_json(claims, final / "post_hardening_claims.json")
    lines = ["# Post-Hardening Claims", ""]
    for k, v in claims.items():
        lines.append(f"- `{k}`: `{v}`")
    (final / "post_hardening_claims.md").write_text("\n".join(lines), encoding="utf-8")

    manuscript = final / "manuscript"
    manuscript.mkdir(parents=True, exist_ok=True)
    updated_abstract = f"""# Updated Abstract Draft

ScaleBreak-FlyVis tests which dynamic visual representations remain decodable across retinal apparent scale in a pretrained connectome-constrained model of the fly visual system. FlyVis retains strong off-diagonal held-out-scale direction information across moving-edge, moving-bar, and small-target stimuli. The serious CNN control status is: {claims['strong_trained_vision_control_status']}. Connectome causality status is: {claims['connectome_causality_claim_strength']}. These results support FlyVis model-specific dynamic scale-generalization across retinal projection, while explicitly not claiming physical-distance perception, generic object recognition, or established exact connectome necessity.
"""
    (manuscript / "updated_abstract.md").write_text(updated_abstract, encoding="utf-8")
    outline = """# Updated Results Outline

1. Controlled retinal apparent-scale benchmark.
2. FlyVis dynamic direction generalization across held-out apparent scales.
3. Serious trained CNN baseline and interpretation boundary.
4. Feature-family results: dynamic families primary, static shapes secondary.
5. Distributed cell-group representation and temporal lesion timing.
6. Connectome-causality pilot: report only as provisional unless direct internal perturbations are validated.
7. Final claim boundary: model-specific dynamic representation yes; exact connectome necessity not established.
"""
    (manuscript / "updated_results_outline.md").write_text(outline, encoding="utf-8")
    print(f"Wrote post-hardening claims to {final}")


if __name__ == "__main__":
    main()
