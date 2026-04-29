"""Model adapters and activity/rate-proxy baselines."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import ndimage

from .connectome import type_adjacency_matrix


def pixel_baseline(videos: np.ndarray) -> np.ndarray:
    mean_frame = videos.mean(axis=1).reshape(len(videos), -1)
    max_frame = videos.max(axis=1).reshape(len(videos), -1)
    diff = np.diff(videos, axis=1)
    diff_stats = np.stack([np.abs(diff).mean(axis=(1, 2, 3)), diff.std(axis=(1, 2, 3)), videos.sum(axis=(1, 2, 3))], axis=1)
    return np.concatenate([mean_frame, max_frame, diff_stats], axis=1).astype(np.float32)


def local_rnn(videos: np.ndarray, seed: int = 0, channels: int = 8, alpha: float = 0.35) -> np.ndarray:
    try:
        import torch
        import torch.nn.functional as F

        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        n, t, h, w = videos.shape
        x = torch.from_numpy(videos.astype(np.float32))
        h_state = torch.zeros((n, channels, h, w), dtype=torch.float32)
        k_input = torch.from_numpy(rng.normal(0, 0.25, size=(channels, 1, 3, 3)).astype(np.float32))
        k_rec = torch.from_numpy(rng.normal(0, 0.08, size=(channels, 1, 3, 3)).astype(np.float32))
        outs = []
        for ti in range(t):
            drive = F.conv2d(x[:, ti : ti + 1], k_input, padding=1)
            rec = F.conv2d(h_state, k_rec, padding=1, groups=channels)
            h_state = (1 - alpha) * h_state + alpha * torch.relu(drive + rec)
            outs.append(h_state.mean(dim=(2, 3)).numpy())
        return np.stack(outs, axis=1).astype(np.float32)
    except Exception:
        rng = np.random.default_rng(seed)
        n, t, h, w = videos.shape
        h_state = np.zeros((n, channels, h, w), dtype=np.float32)
        k_input = rng.normal(0, 0.25, size=(channels, 3, 3)).astype(np.float32)
        k_rec = rng.normal(0, 0.08, size=(channels, 3, 3)).astype(np.float32)
        out = np.zeros((n, t, channels), dtype=np.float32)
        for ti in range(t):
            frame = videos[:, ti]
            drive = np.stack([ndimage.convolve(frame[i], k_input[c], mode="nearest") for i in range(n) for c in range(channels)])
            drive = drive.reshape(n, channels, h, w)
            rec = np.stack([ndimage.convolve(h_state[i, c], k_rec[c], mode="nearest") for i in range(n) for c in range(channels)])
            rec = rec.reshape(n, channels, h, w)
            h_state = (1 - alpha) * h_state + alpha * np.maximum(0, drive + rec)
            out[:, ti] = h_state.mean(axis=(2, 3))
        return out


def small_cnn_random(videos: np.ndarray, seed: int = 0, channels: int = 12) -> np.ndarray:
    rng = np.random.default_rng(seed)
    kernels = rng.normal(0, 0.3, size=(channels, 5, 5)).astype(np.float32)
    summary = videos.mean(axis=1)
    feats = []
    for k in kernels:
        conv = np.stack([ndimage.convolve(frame, k, mode="nearest") for frame in summary])
        feats.append(np.maximum(0, conv).mean(axis=(1, 2)))
        feats.append(np.maximum(0, conv).max(axis=(1, 2)))
    return np.stack(feats, axis=1).astype(np.float32)[:, None, :]


def optic_lobe_type_rate(videos: np.ndarray, type_edges_path: str | Path | None, seed: int = 0, n_units: int = 64) -> tuple[np.ndarray, dict[str, Any]]:
    rng = np.random.default_rng(seed)
    if type_edges_path and Path(type_edges_path).exists():
        type_edges = pd.read_parquet(type_edges_path)
        w, types = type_adjacency_matrix(type_edges, max_types=n_units)
        if w.shape[0] == 0:
            w = rng.normal(0, 0.05, size=(n_units, n_units)).astype(np.float32)
            types = [f"generic_type_{i}" for i in range(n_units)]
    else:
        w = rng.normal(0, 0.05, size=(n_units, n_units)).astype(np.float32)
        types = [f"generic_type_{i}" for i in range(n_units)]
    n_units = w.shape[0]
    input_w = rng.normal(0, 0.4, size=(n_units, 6)).astype(np.float32)
    n, t, _, _ = videos.shape
    h = np.zeros((n, n_units), dtype=np.float32)
    out = np.zeros((n, t, n_units), dtype=np.float32)
    for ti in range(t):
        x = videos[:, ti]
        gx = np.gradient(x, axis=2)
        gy = np.gradient(x, axis=1)
        drive_features = np.stack(
            [
                x.mean(axis=(1, 2)),
                x.max(axis=(1, 2)),
                x.std(axis=(1, 2)),
                np.abs(gx).mean(axis=(1, 2)),
                np.abs(gy).mean(axis=(1, 2)),
                (x > 0).mean(axis=(1, 2)),
            ],
            axis=1,
        ).astype(np.float32)
        h = 0.8 * h + 0.2 * np.maximum(0, h @ w.T + drive_features @ input_w.T)
        out[:, ti] = h
    return out, {"unit_types": types, "uses_real_type_graph": bool(type_edges_path and Path(type_edges_path).exists())}


def flyvis_available() -> tuple[bool, str]:
    try:
        import flyvis  # type: ignore  # noqa: F401

        return True, "flyvis import succeeded; adapter stub needs project-specific pretrained API selection."
    except Exception as exc:
        return False, f"flyvis unavailable: {type(exc).__name__}: {exc}"
