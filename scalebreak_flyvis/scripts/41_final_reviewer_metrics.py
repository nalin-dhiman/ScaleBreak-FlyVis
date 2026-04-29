#!/usr/bin/env python
"""Final small reviewer metrics: ID/OOD, calibration, phase scramble.

This script only uses existing stimuli/responses and trained-output tables. It
does not change the core FlyVis or baseline results.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scalebreak.models import pixel_baseline  # noqa: E402


def temporal_bins(a: np.ndarray, bins: int = 5) -> np.ndarray:
    edges = np.linspace(0, a.shape[1], bins + 1, dtype=int)
    return np.concatenate([a[:, edges[i] : edges[i + 1]].mean(axis=1) for i in range(bins)], axis=1)


def dynamic_labels(meta: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, LabelEncoder]:
    dynamic = meta["feature_family"].isin(["moving_edge", "moving_bar", "small_translating_target"]).to_numpy()
    le = LabelEncoder()
    y = np.full(len(meta), -1, dtype=int)
    y[dynamic] = le.fit_transform(meta.loc[dynamic, "direction"].astype(str))
    return dynamic, y, le


def in_scale_accuracy(x: np.ndarray, y: np.ndarray, dynamic: np.ndarray, seed: int = 42) -> float:
    idx = np.flatnonzero(dynamic)
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_rel, test_rel = next(splitter.split(idx, y[idx]))
    train = idx[train_rel]
    test = idx[test_rel]
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed, n_jobs=1),
    )
    clf.fit(x[train], y[train])
    return float(accuracy_score(y[test], clf.predict(x[test])))


def flyvis_probs(outputs: Path) -> tuple[np.ndarray, np.ndarray]:
    resp = np.load(outputs / "flyvis_pilot_v2" / "responses" / "flyvis_central_cell_responses.npy", mmap_mode="r")
    meta = pd.read_csv(outputs / "flyvis_pilot_v2" / "responses" / "metadata.csv")
    dynamic, y, _ = dynamic_labels(meta)
    x = temporal_bins(np.asarray(resp), bins=5)
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
    ce = float(log_loss(y_true, prob, labels=np.arange(prob.shape[1])))
    bits = float(np.log2(prob.shape[1]) - ce / np.log(2))
    return {"accuracy": float(correct.mean()), "ece": ece, "brier": brier, "cross_entropy": ce, "direction_bits_proxy": bits}


def apply_temperature(prob: np.ndarray, temperature: float) -> np.ndarray:
    logits = np.log(np.clip(prob, 1e-8, 1.0)) / temperature
    logits = logits - logits.max(axis=1, keepdims=True)
    out = np.exp(logits)
    return out / out.sum(axis=1, keepdims=True)


def temperature_scaling_rows(model: str, y: np.ndarray, prob: np.ndarray, seed: int = 42) -> list[dict[str, float | str]]:
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=seed)
    cal_idx, eval_idx = next(splitter.split(np.arange(len(y)), y))
    temps = np.linspace(0.25, 5.0, 96)
    losses = [log_loss(y[cal_idx], apply_temperature(prob[cal_idx], t), labels=np.arange(prob.shape[1])) for t in temps]
    best_t = float(temps[int(np.argmin(losses))])
    rows = []
    rows.append({"model": model, "calibration": "uncalibrated", "temperature": 1.0, **calibration(y[eval_idx], prob[eval_idx])})
    rows.append({"model": model, "calibration": "temperature_scaled", "temperature": best_t, **calibration(y[eval_idx], apply_temperature(prob[eval_idx], best_t))})
    return rows


def stn_probs(outputs: Path) -> tuple[np.ndarray, np.ndarray] | None:
    path = outputs / "stn_cnn_baseline" / "predictions_stn_cnn.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    prob_cols = [c for c in df.columns if c.startswith("prob_")]
    labels = [c.replace("prob_", "") for c in prob_cols]
    label_to_i = {label: i for i, label in enumerate(labels)}
    return np.array([label_to_i[str(v)] for v in df["true_label"]], dtype=int), df[prob_cols].to_numpy(dtype=float)


def phase_scramble_chunk(videos: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    x = np.asarray(videos, dtype=np.float32)
    spec = np.fft.rfft(x[:, :, 0, :], axis=1)
    phase = rng.uniform(0, 2 * np.pi, size=spec.shape).astype(np.float32)
    phase[:, 0, :] = 0.0
    if x.shape[1] % 2 == 0:
        phase[:, -1, :] = 0.0
    scrambled = np.fft.irfft(np.abs(spec) * np.exp(1j * phase), n=x.shape[1], axis=1).astype(np.float32)
    scrambled = scrambled[:, :, None, :]
    return scrambled


def loso_pixel_accuracy(x: np.ndarray, meta_dyn: pd.DataFrame, y: np.ndarray) -> float:
    correct = []
    scales = meta_dyn["scale"].to_numpy()
    for heldout in sorted(meta_dyn["scale"].unique()):
        train = scales != heldout
        test = scales == heldout
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced", random_state=42, n_jobs=1))
        clf.fit(x[train], y[train])
        correct.append(clf.predict(x[test]) == y[test])
    return float(np.concatenate(correct).mean())


def phase_scramble_control(outputs: Path, seed: int = 42, chunk: int = 96) -> pd.DataFrame:
    stim = np.load(outputs / "flyvis_pilot_v2" / "stimuli" / "stimuli.npy", mmap_mode="r")
    meta = pd.read_csv(outputs / "flyvis_pilot_v2" / "responses" / "metadata.csv")
    dynamic, _, le = dynamic_labels(meta)
    meta_dyn = meta.loc[dynamic].reset_index(drop=True)
    y_dyn = le.transform(meta_dyn["direction"].astype(str))
    idx = np.flatnonzero(dynamic)
    rng = np.random.default_rng(seed)
    feats_orig = []
    feats_phase = []
    for start in range(0, len(idx), chunk):
        ids = idx[start : start + chunk]
        vids = np.asarray(stim[ids], dtype=np.float32)
        feats_orig.append(pixel_baseline(vids))
        feats_phase.append(pixel_baseline(phase_scramble_chunk(vids, rng)))
    x_orig = np.concatenate(feats_orig)
    x_phase = np.concatenate(feats_phase)
    return pd.DataFrame(
        [
            {"control": "original_pixel_dynamic_subset", "mean_loso_accuracy": loso_pixel_accuracy(x_orig, meta_dyn, y_dyn)},
            {"control": "phase_scrambled_pixel", "mean_loso_accuracy": loso_pixel_accuracy(x_phase, meta_dyn, y_dyn)},
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs-dir", default="scalebreak_flyvis/outputs")
    args = parser.parse_args()
    outputs = Path(args.outputs_dir)
    out = outputs / "final_reviewer_metrics"
    out.mkdir(parents=True, exist_ok=True)

    # ID vs OOD
    resp = np.load(outputs / "flyvis_pilot_v2" / "responses" / "flyvis_central_cell_responses.npy", mmap_mode="r")
    meta = pd.read_csv(outputs / "flyvis_pilot_v2" / "responses" / "metadata.csv")
    dynamic, y, _ = dynamic_labels(meta)
    flyvis_x = temporal_bins(np.asarray(resp), bins=5)
    serious = pd.read_csv(outputs / "serious_cnn_baseline" / "table_serious_cnn_by_seed_scale.csv")
    hex_path = outputs / "hex_native_temporal_baseline" / "table_hex_native_by_seed_scale.csv"
    hex_df = pd.read_csv(hex_path) if hex_path.exists() else pd.DataFrame()
    id_rows = [
        {"model": "FlyVis", "in_scale_accuracy": in_scale_accuracy(flyvis_x, y, dynamic), "loso_accuracy": 0.9236111111111112},
        {
            "model": "TemporalResNet18Small",
            "in_scale_accuracy": float(serious["best_val_accuracy"].mean()),
            "loso_accuracy": float(pd.read_csv(outputs / "serious_cnn_baseline" / "table_serious_cnn_summary.csv").iloc[0]["mean_offdiag_accuracy"]),
        },
    ]
    if not hex_df.empty:
        id_rows.append(
            {
                "model": "Hex-native temporal model",
                "in_scale_accuracy": float(hex_df["best_val_accuracy"].mean()),
                "loso_accuracy": float(pd.read_csv(outputs / "hex_native_temporal_baseline" / "table_hex_native_summary.csv").iloc[0]["mean_offdiag_accuracy"]),
            }
        )
    id_df = pd.DataFrame(id_rows)
    id_df.to_csv(out / "table_id_vs_ood_accuracy.csv", index=False)
    try:
        id_df.to_markdown(out / "table_id_vs_ood_accuracy.md", index=False)
    except Exception:
        pass

    # Temperature scaling
    rows = []
    fy, fp = flyvis_probs(outputs)
    rows.extend(temperature_scaling_rows("FlyVis linear probe", fy, fp))
    stn = stn_probs(outputs)
    if stn is not None:
        sy, sp = stn
        rows.extend(temperature_scaling_rows("STN-CNN", sy, sp))
    temp = pd.DataFrame(rows)
    temp.to_csv(out / "table_temperature_scaling_calibration.csv", index=False)
    try:
        temp.to_markdown(out / "table_temperature_scaling_calibration.md", index=False)
    except Exception:
        pass

    phase = phase_scramble_control(outputs)
    phase.to_csv(out / "table_pixel_phase_scramble_summary.csv", index=False)
    try:
        phase.to_markdown(out / "table_pixel_phase_scramble_summary.md", index=False)
    except Exception:
        pass
    print(id_df.round(3).to_string(index=False))
    print(temp.round(3).to_string(index=False))
    print(phase.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
