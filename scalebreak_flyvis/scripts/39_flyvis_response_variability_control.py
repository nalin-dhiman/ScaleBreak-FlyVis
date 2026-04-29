#!/usr/bin/env python
"""Small FlyVis response-variability diagnostic.

This is not a new FlyVis checkpoint. It tests whether the documented LOSO
direction readout is stable to small additive response perturbations, as a
lightweight proxy for run-to-run response variability when additional
pretrained checkpoints are unavailable.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler


def temporal_bins(a: np.ndarray, bins: int = 5) -> np.ndarray:
    edges = np.linspace(0, a.shape[1], bins + 1, dtype=int)
    return np.concatenate([a[:, edges[i] : edges[i + 1]].mean(axis=1) for i in range(bins)], axis=1)


def loso_accuracy(x: np.ndarray, meta: pd.DataFrame, y: np.ndarray, dynamic: np.ndarray, seed: int) -> float:
    correct: list[np.ndarray] = []
    for heldout in sorted(meta.loc[dynamic, "scale"].unique()):
        train = dynamic & (meta["scale"].to_numpy() != heldout)
        test = dynamic & (meta["scale"].to_numpy() == heldout)
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed, n_jobs=1),
        )
        clf.fit(x[train], y[train])
        pred = clf.predict(x[test])
        correct.append(pred == y[test])
    return float(np.concatenate(correct).mean())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs-dir", default="scalebreak_flyvis/outputs")
    parser.add_argument("--noise-levels", default="0,0.01,0.05")
    parser.add_argument("--seeds", default="42,84,96")
    args = parser.parse_args()

    outputs = Path(args.outputs_dir)
    out = outputs / "flyvis_variability_control"
    out.mkdir(parents=True, exist_ok=True)

    resp = np.load(outputs / "flyvis_pilot_v2" / "responses" / "flyvis_central_cell_responses.npy", mmap_mode="r")
    meta = pd.read_csv(outputs / "flyvis_pilot_v2" / "responses" / "metadata.csv")
    dynamic = meta["feature_family"].isin(["moving_edge", "moving_bar", "small_translating_target"]).to_numpy()
    x0 = temporal_bins(np.asarray(resp), bins=5).astype(np.float32)
    scale = np.maximum(x0.std(axis=0, keepdims=True), 1e-6)
    le = LabelEncoder()
    y = np.full(len(meta), -1, dtype=int)
    y[dynamic] = le.fit_transform(meta.loc[dynamic, "direction"].astype(str))

    rows = []
    levels = [float(v) for v in args.noise_levels.split(",") if v.strip()]
    seeds = [int(v) for v in args.seeds.split(",") if v.strip()]
    for level in levels:
        for seed in seeds:
            rng = np.random.default_rng(seed)
            if level == 0:
                x = x0
            else:
                x = x0 + rng.normal(0.0, level, size=x0.shape).astype(np.float32) * scale
            acc = loso_accuracy(x, meta, y, dynamic, seed)
            rows.append({"variant": f"response_noise_{level:g}", "noise_fraction_of_feature_std": level, "seed": seed, "accuracy": acc})

    df = pd.DataFrame(rows)
    df.to_csv(out / "table_flyvis_response_noise_variability.csv", index=False)
    summary = (
        df.groupby(["variant", "noise_fraction_of_feature_std"], as_index=False)
        .agg(mean_accuracy=("accuracy", "mean"), min_accuracy=("accuracy", "min"), max_accuracy=("accuracy", "max"), n=("accuracy", "size"))
        .sort_values("noise_fraction_of_feature_std")
    )
    summary.to_csv(out / "table_flyvis_response_noise_variability_summary.csv", index=False)
    try:
        summary.to_markdown(out / "table_flyvis_response_noise_variability_summary.md", index=False)
    except Exception:
        (out / "table_flyvis_response_noise_variability_summary.md").write_text(
            "```csv\n" + summary.to_csv(index=False) + "```\n", encoding="utf-8"
        )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
