#!/usr/bin/env python
"""Compute calibration diagnostics for available probability outputs.

This script is deliberately diagnostic-only. It reconstructs FlyVis LOSO
probabilities from existing response tensors with the documented linear probe,
and reads STN-CNN probability outputs from the bounded scale-aware baseline.
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


def calibration(y_true: np.ndarray, prob: np.ndarray, n_bins: int = 10) -> dict[str, float]:
    onehot = np.eye(prob.shape[1])[y_true]
    brier = float(np.mean(np.sum((prob - onehot) ** 2, axis=1)))
    conf = prob.max(axis=1)
    pred = prob.argmax(axis=1)
    correct = (pred == y_true).astype(float)
    ece = 0.0
    for lo in np.linspace(0, 1, n_bins, endpoint=False):
        hi = lo + 1 / n_bins
        mask = (conf >= lo) & (conf < hi if hi < 1 else conf <= hi)
        if mask.any():
            ece += float(mask.mean()) * abs(float(correct[mask].mean()) - float(conf[mask].mean()))
    return {
        "accuracy": float(correct.mean()),
        "ece": float(ece),
        "brier": brier,
        "mean_confidence": float(conf.mean()),
        "n_predictions": int(len(y_true)),
    }


def flyvis_probs(outputs: Path) -> tuple[np.ndarray, np.ndarray]:
    resp = np.load(outputs / "flyvis_pilot_v2" / "responses" / "flyvis_central_cell_responses.npy", mmap_mode="r")
    meta = pd.read_csv(outputs / "flyvis_pilot_v2" / "responses" / "metadata.csv")
    dynamic = meta["feature_family"].isin(["moving_edge", "moving_bar", "small_translating_target"]).to_numpy()
    x = temporal_bins(np.asarray(resp), bins=5)
    le = LabelEncoder()
    y = np.full(len(meta), -1, dtype=int)
    y[dynamic] = le.fit_transform(meta.loc[dynamic, "direction"].astype(str))
    probs = []
    truth = []
    for heldout in sorted(meta.loc[dynamic, "scale"].unique()):
        train = dynamic & (meta["scale"].to_numpy() != heldout)
        test = dynamic & (meta["scale"].to_numpy() == heldout)
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42, n_jobs=1),
        )
        clf.fit(x[train], y[train])
        probs.append(clf.predict_proba(x[test]))
        truth.append(y[test])
    return np.concatenate(truth), np.concatenate(probs)


def stn_probs(outputs: Path) -> tuple[np.ndarray, np.ndarray] | None:
    path = outputs / "stn_cnn_baseline" / "predictions_stn_cnn.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    prob_cols = [c for c in df.columns if c.startswith("prob_")]
    labels = [c.replace("prob_", "") for c in prob_cols]
    label_to_i = {label: i for i, label in enumerate(labels)}
    y = np.array([label_to_i[str(v)] for v in df["true_label"]], dtype=int)
    return y, df[prob_cols].to_numpy(dtype=float)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs-dir", default="scalebreak_flyvis/outputs")
    args = parser.parse_args()
    outputs = Path(args.outputs_dir)
    out = outputs / "calibration_reliability"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    y, p = flyvis_probs(outputs)
    rows.append({"model": "FlyVis linear probe", **calibration(y, p)})
    stn = stn_probs(outputs)
    if stn is not None:
        y, p = stn
        rows.append({"model": "STN-CNN", **calibration(y, p)})
    df = pd.DataFrame(rows)
    df.to_csv(out / "table_calibration_reliability.csv", index=False)
    try:
        df.to_markdown(out / "table_calibration_reliability.md", index=False)
    except Exception:
        (out / "table_calibration_reliability.md").write_text("```csv\n" + df.to_csv(index=False) + "```\n", encoding="utf-8")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
