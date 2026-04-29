#!/usr/bin/env python
"""Generate non-retrained FlyVis causal-variant proxy features.

This script uses saved FlyVis responses. Unless the audit has manually
certified safe in-place weight editing, perturbations are explicitly labeled as
response-space proxy perturbations, not true internal FlyVis rewiring.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def write_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)


def parse_list(text: str) -> list[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def parse_float_list(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def feature_matrix(resp: np.ndarray, pre_frames: int, bins: int = 5) -> np.ndarray:
    base = resp[:, :pre_frames].mean(axis=1, keepdims=True)
    delta = resp - base
    chunks = np.array_split(delta, bins, axis=1)
    x = np.concatenate([c.mean(axis=1) for c in chunks], axis=1).astype(np.float32)
    x = np.nan_to_num(x)
    return (x - x.mean(axis=0, keepdims=True)) / np.where(x.std(axis=0, keepdims=True) > 1e-8, x.std(axis=0, keepdims=True), 1.0)


def activity_metrics(resp: np.ndarray, pre_frames: int) -> dict[str, float]:
    base = resp[:, :pre_frames].mean(axis=1, keepdims=True)
    delta = resp - base
    return {
        "mean_abs_activity_proxy": float(np.abs(delta).mean()),
        "peak_abs_activity_proxy": float(np.abs(delta).max()),
        "temporal_variance_proxy": float(delta.var(axis=1).mean()),
    }


def cell_sets(cell_meta: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    types = cell_meta["cell_type"].astype(str).to_numpy()
    t4t5 = np.array([t.startswith("T4") or t.startswith("T5") for t in types])
    return t4t5, ~t4t5


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v2-dir", default="scalebreak_flyvis/outputs/flyvis_pilot_v2")
    parser.add_argument("--out-dir", default="scalebreak_flyvis/outputs/connectome_causality")
    parser.add_argument("--variants", default="full,weight_shuffle,edge_dropout,t4t5_attenuation,non_t4t5_attenuation")
    parser.add_argument("--dropout-levels", default="0.05,0.10,0.20")
    parser.add_argument("--attenuation-levels", default="0.0,0.25,0.5,0.75")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        write_json({"status": "dry_run", "variants": args.variants}, out / "causal_variant_run_info.json")
        print(f"Dry-run wrote plan to {out}")
        return

    rng = np.random.default_rng(args.seed)
    v2 = Path(args.v2_dir)
    resp = np.load(v2 / "responses" / "flyvis_central_cell_responses.npy", mmap_mode="r")
    meta = pd.read_csv(v2 / "responses" / "metadata.csv")
    cell_meta = pd.read_csv(v2 / "responses" / "cell_metadata.csv")
    pre_frames = int(round(float(meta["t_pre"].iloc[0]) / float(meta["dt"].iloc[0])))
    response = np.asarray(resp)
    variants = parse_list(args.variants)
    dropout_levels = parse_float_list(args.dropout_levels)
    attenuation_levels = parse_float_list(args.attenuation_levels)
    if args.quick:
        dropout_levels = dropout_levels[:1]
        attenuation_levels = [0.5]

    t4t5, non = cell_sets(cell_meta)
    feature_dict: dict[str, np.ndarray] = {}
    rows = []

    def add(name: str, perturbed: np.ndarray, perturbation: str, level: float | str) -> None:
        feature_dict[name] = feature_matrix(perturbed, pre_frames)
        rows.append(
            {
                "variant": name,
                "perturbation": perturbation,
                "level": level,
                "implementation": "response-space proxy summary feature" if name != "full" else "saved original FlyVis responses",
                **activity_metrics(perturbed, pre_frames),
            }
        )

    if "full" in variants:
        add("full", response, "none", "none")
    if "weight_shuffle" in variants:
        shuffled = response.copy()
        for c in range(shuffled.shape[2]):
            scale = shuffled[:, :, c].copy()
            shuffled[:, :, c] = scale[rng.permutation(scale.shape[0])]
        add("weight_shuffle_proxy", shuffled, "trial-wise per-cell response shuffle preserving cell marginal traces", "all")
    if "edge_dropout" in variants:
        for level in dropout_levels:
            dropped = response.copy()
            n_drop = max(1, int(level * dropped.shape[2]))
            drop = rng.choice(dropped.shape[2], size=n_drop, replace=False)
            dropped[:, :, drop] = 0.0
            add(f"edge_dropout_proxy_{level:.2f}", dropped, "random central-cell response dropout proxy", level)
    if "t4t5_attenuation" in variants:
        for level in attenuation_levels:
            atten = response.copy()
            atten[:, :, t4t5] *= level
            add(f"t4t5_attenuation_{level:.2f}", atten, "T4/T5 response attenuation proxy", level)
    if "non_t4t5_attenuation" in variants:
        for level in attenuation_levels:
            atten = response.copy()
            atten[:, :, non] *= level
            add(f"non_t4t5_attenuation_{level:.2f}", atten, "non-T4/T5 response attenuation proxy", level)

    np.savez_compressed(out / "causal_variant_features.npz", **feature_dict)
    pd.DataFrame(rows).to_csv(out / "table_causal_activity_metrics.csv", index=False)
    meta.to_csv(out / "metadata.csv", index=False)
    write_json(
        {
            "status": "completed",
            "response_shape": tuple(resp.shape),
            "pre_frames": pre_frames,
            "variants": list(feature_dict.keys()),
            "note": "These are non-retrained response-space perturbation proxies unless audit certifies internal weight editing.",
        },
        out / "causal_variant_run_info.json",
    )
    print(f"Wrote causal variant features to {out}")


if __name__ == "__main__":
    main()
