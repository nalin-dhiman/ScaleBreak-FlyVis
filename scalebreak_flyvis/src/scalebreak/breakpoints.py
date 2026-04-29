"""Breakpoint estimation for scale-dependent representational breakdown."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .probes import stratified_accuracy


def accuracy_by_scale(x: np.ndarray, meta: pd.DataFrame, target: str = "shape", seed: int = 0) -> pd.DataFrame:
    rows = []
    for scale in sorted(meta["scale"].unique()):
        idx = meta["scale"].to_numpy() == scale
        acc, std = stratified_accuracy(x[idx], meta.loc[idx, target].astype(str).to_numpy(), seed=seed)
        rows.append({"scale": scale, "target": target, "accuracy": acc, "accuracy_std": std, "n": int(idx.sum())})
    return pd.DataFrame(rows)


def estimate_breakpoint(curve: pd.DataFrame, chance: float, margin: float = 0.05) -> float | None:
    reliable = curve.sort_values("scale")
    ok = reliable["accuracy"].to_numpy() >= chance + margin
    scales = reliable["scale"].to_numpy()
    if not ok.any():
        return None
    return float(scales[np.argmax(ok)])


def bootstrap_breakpoints(
    x: np.ndarray,
    meta: pd.DataFrame,
    target: str = "shape",
    seed: int = 0,
    n_boot: int = 100,
    margin: float = 0.05,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    classes = meta[target].nunique()
    chance = 1.0 / max(classes, 1)
    curves = []
    bps = []
    for b in range(n_boot):
        idx = rng.integers(0, len(meta), size=len(meta))
        curve = accuracy_by_scale(x[idx], meta.iloc[idx].reset_index(drop=True), target=target, seed=seed + b)
        curve["bootstrap"] = b
        curves.append(curve)
        bps.append({"bootstrap": b, "breakpoint_scale": estimate_breakpoint(curve, chance, margin), "chance": chance})
    return pd.DataFrame(bps), pd.concat(curves, ignore_index=True)
