"""Plotting helpers for final figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def save_heatmap(matrix: pd.DataFrame, path: str | Path, title: str = "", cmap: str = "viridis") -> None:
    plt.figure(figsize=(7, 6))
    plt.imshow(matrix.values, aspect="auto", cmap=cmap)
    plt.colorbar()
    plt.xticks(range(len(matrix.columns)), matrix.columns, rotation=90, fontsize=6)
    plt.yticks(range(len(matrix.index)), matrix.index, fontsize=6)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def save_stimulus_montage(videos: np.ndarray, meta: pd.DataFrame, path: str | Path, n: int = 24) -> None:
    idx = np.linspace(0, len(videos) - 1, min(n, len(videos))).astype(int)
    cols = 6
    rows = int(np.ceil(len(idx) / cols))
    plt.figure(figsize=(cols * 2, rows * 2))
    for k, i in enumerate(idx, 1):
        ax = plt.subplot(rows, cols, k)
        ax.imshow(videos[i, videos.shape[1] // 2], cmap="gray", vmin=0, vmax=1)
        r = meta.iloc[i]
        ax.set_title(f"{r.shape} s={r.scale:g}", fontsize=7)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
