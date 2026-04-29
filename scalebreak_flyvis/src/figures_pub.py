"""Publication figure utilities for ScaleBreak-FlyVis.

The helpers here centralize typography, palette, panel labels, and vector
export so the main and supplementary figure builders stay consistent.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


PALETTE = {
    "flyvis": "#1f77b4",
    "cnn": "#ff7f0e",
    "pixel": "#2ca02c",
    "local_rnn": "#9467bd",
    "graph": "#8c564b",
    "nuisance": "#7f7f7f",
    "destructive": "#d62728",
    "black": "#222222",
    "light_gray": "#d9d9d9",
}


def set_pub_style() -> None:
    """Apply deterministic publication plotting style."""

    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.linewidth": 0.75,
            "lines.linewidth": 1.8,
            "xtick.major.width": 0.75,
            "ytick.major.width": 0.75,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "savefig.transparent": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def model_color(model: str) -> str:
    """Return a consistent color for a model/control label."""

    m = model.lower()
    if "flyvis" in m or m == "full":
        return PALETTE["flyvis"]
    if "cnn" in m or "resnet" in m or "temporalresnet" in m or "hex-native" in m:
        return PALETTE["cnn"]
    if "pixel" in m and "area" not in m:
        return PALETTE["pixel"]
    if "rnn" in m:
        return PALETTE["local_rnn"]
    if "graph" in m or "optic" in m or "dropout" in m or "attenuation" in m:
        return PALETTE["graph"]
    if "nuisance" in m or "area" in m or "pixel" in m:
        return PALETTE["nuisance"]
    if "shuffle" in m or "noise" in m or "permutation" in m or "destructive" in m:
        return PALETTE["destructive"]
    return PALETTE["black"]


def panel_label(ax, label: str, x: float = -0.15, y: float = 1.05) -> None:
    """Add a bold multi-panel label in axes coordinates."""

    ax.text(x, y, label, transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom", ha="left")


def clean_axis(ax, grid: bool = False) -> None:
    """Apply consistent axis cleanup."""

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid:
        ax.grid(axis="y", alpha=0.15, linewidth=0.5)
        ax.set_axisbelow(True)


def save_pub(fig, out_base: Path, preview_png: bool = True) -> list[Path]:
    """Save a figure as PDF, SVG, and optional PNG preview."""

    out_base.parent.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for ext in ["pdf", "svg"]:
        path = out_base.with_suffix(f".{ext}")
        fig.savefig(path, bbox_inches="tight")
        paths.append(path)
    if preview_png:
        path = out_base.with_suffix(".png")
        fig.savefig(path, dpi=240, bbox_inches="tight")
        paths.append(path)
    return paths


def format_scale_labels(values: Iterable[float]) -> list[str]:
    """Format scale labels without noisy floating point suffixes."""

    labels = []
    for v in values:
        fv = float(v)
        labels.append(str(int(fv)) if fv.is_integer() else f"{fv:g}")
    return labels


def annotate_bars(ax, values: Iterable[float], orientation: str = "vertical", fmt: str = "{:.2f}") -> None:
    """Annotate bar values cleanly."""

    vals = list(values)
    if orientation == "vertical":
        for patch, value in zip(ax.patches, vals):
            ax.text(
                patch.get_x() + patch.get_width() / 2,
                patch.get_height() + 0.015,
                fmt.format(value),
                ha="center",
                va="bottom",
                fontsize=7,
                rotation=0,
            )
    else:
        for patch, value in zip(ax.patches, vals):
            ax.text(
                patch.get_width() + 0.015,
                patch.get_y() + patch.get_height() / 2,
                fmt.format(value),
                ha="left",
                va="center",
                fontsize=7,
            )


def save_table(df: pd.DataFrame, path_csv: Path) -> None:
    """Save CSV and Markdown sidecar with a no-tabulate fallback."""

    path_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path_csv, index=False)
    try:
        md = df.to_markdown(index=False)
    except Exception:
        md = "```\n" + df.to_csv(index=False) + "```\n"
    path_csv.with_suffix(".md").write_text(md, encoding="utf-8")


def hex_mapping(coords: pd.DataFrame, grid_size: int = 32) -> pd.DataFrame:
    """Deterministically project FlyVis hex coordinates to a square grid."""

    x = coords["x"].to_numpy(dtype=float)
    y = coords["y"].to_numpy(dtype=float)
    gx = np.round((x - x.min()) / max(x.max() - x.min(), 1e-8) * (grid_size - 1)).astype(int)
    gy = np.round((y - y.min()) / max(y.max() - y.min(), 1e-8) * (grid_size - 1)).astype(int)
    return pd.DataFrame({"hex_pixel": np.arange(len(coords)), "grid_x": gx, "grid_y": gy, "source_x": x, "source_y": y})


def project_hex_frame(values: np.ndarray, mapping: pd.DataFrame, grid_size: int = 32, fill: float = 0.5) -> np.ndarray:
    """Project one hex-pixel frame into a square grid for visualization."""

    acc = np.zeros((grid_size, grid_size), dtype=np.float32)
    count = np.zeros((grid_size, grid_size), dtype=np.float32)
    for hp, row in mapping.iterrows():
        y = int(row["grid_y"])
        x = int(row["grid_x"])
        acc[y, x] += float(values[int(row["hex_pixel"])])
        count[y, x] += 1.0
    out = np.full((grid_size, grid_size), fill, dtype=np.float32)
    mask = count > 0
    out[mask] = acc[mask] / count[mask]
    return out
