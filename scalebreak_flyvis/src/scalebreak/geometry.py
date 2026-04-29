"""Retinal stimulus geometry helpers."""

from __future__ import annotations

import numpy as np
from scipy import ndimage


def coordinate_grid(height: int, width: int, cx: float, cy: float) -> tuple[np.ndarray, np.ndarray]:
    yy, xx = np.mgrid[0:height, 0:width]
    return xx - cx, yy - cy


def binary_shape(
    shape: str,
    height: int,
    width: int,
    scale: float,
    cx: float,
    cy: float,
    orientation: float = 0.0,
) -> np.ndarray:
    x, y = coordinate_grid(height, width, cx, cy)
    theta = np.deg2rad(orientation)
    xr = x * np.cos(theta) + y * np.sin(theta)
    yr = -x * np.sin(theta) + y * np.cos(theta)

    if shape == "disk":
        mask = x**2 + y**2 <= (scale / 2.0) ** 2
    elif shape == "square":
        mask = (np.abs(xr) <= scale / 2.0) & (np.abs(yr) <= scale / 2.0)
    elif shape == "bar":
        mask = (np.abs(xr) <= scale * 1.5) & (np.abs(yr) <= max(1.0, scale / 6.0))
    elif shape == "triangle":
        half = scale / 2.0
        y_top = -half
        y_bot = half
        # Upright isosceles triangle in rotated coordinates.
        row_frac = np.clip((yr - y_top) / max(scale, 1e-6), 0, 1)
        half_width = row_frac * half
        mask = (yr >= y_top) & (yr <= y_bot) & (np.abs(xr) <= half_width)
    elif shape == "annulus":
        r = np.sqrt(x**2 + y**2)
        mask = (r <= scale / 2.0) & (r >= scale / 3.0)
    else:
        raise ValueError(f"Unknown shape: {shape}")
    return mask.astype(np.float32)


def edge_length(mask: np.ndarray) -> float:
    gx = ndimage.sobel(mask.astype(float), axis=1)
    gy = ndimage.sobel(mask.astype(float), axis=0)
    return float(np.count_nonzero(np.hypot(gx, gy) > 0))


def bbox(mask: np.ndarray) -> tuple[int, int]:
    ys, xs = np.nonzero(mask > 0)
    if len(xs) == 0:
        return 0, 0
    return int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)


def apply_blur(frame: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return frame
    return ndimage.gaussian_filter(frame, sigma=sigma)
