"""Feature construction from activity/rate-proxy tensors."""

from __future__ import annotations

import numpy as np
import pandas as pd


def as_time_unit(activations: np.ndarray) -> np.ndarray:
    if activations.ndim == 2:
        return activations[:, None, :]
    if activations.ndim > 3:
        return activations.reshape(activations.shape[0], activations.shape[1], -1)
    return activations


def temporal_bins(a: np.ndarray, bins: int = 5) -> np.ndarray:
    bins = max(1, min(int(bins), a.shape[1]))
    chunks = np.array_split(a, bins, axis=1)
    return np.concatenate([c.mean(axis=1) for c in chunks], axis=1)


def normalize_features(x: np.ndarray) -> dict[str, np.ndarray]:
    x = np.nan_to_num(x.astype(np.float32))
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    z = (x - mean) / np.where(std > 1e-8, std, 1.0)
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    l2 = x / np.where(norm > 1e-8, norm, 1.0)
    mag = norm.astype(np.float32)
    direction = l2
    return {"raw": x, "zscore": z.astype(np.float32), "l2": l2.astype(np.float32), "direction_plus_magnitude": np.concatenate([direction, mag], axis=1).astype(np.float32)}


def make_feature_matrices(activations: np.ndarray, bins: int = 5) -> dict[str, np.ndarray]:
    a = as_time_unit(activations)
    return {
        "mean_time": a.mean(axis=1),
        "peak_time": a.max(axis=1),
        "final": a[:, -1],
        "temporal_bins": temporal_bins(a, bins=bins),
    }


def activity_metrics(activations: np.ndarray) -> pd.DataFrame:
    a = as_time_unit(activations)
    abs_a = np.abs(a)
    peak_by_t = abs_a.mean(axis=2)
    return pd.DataFrame(
        {
            "mean_abs_activity": abs_a.mean(axis=(1, 2)),
            "peak_abs_activity": abs_a.max(axis=(1, 2)),
            "l2_activity_norm": np.linalg.norm(a.reshape(len(a), -1), axis=1),
            "temporal_variance": a.var(axis=1).mean(axis=1),
            "latency_to_peak": peak_by_t.argmax(axis=1),
        }
    )
