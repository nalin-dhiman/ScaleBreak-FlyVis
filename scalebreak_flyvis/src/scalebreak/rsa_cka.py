"""Representational similarity analysis and linear CKA."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def class_means(x: np.ndarray, meta: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    keys = meta[["shape", "scale"]].astype(str).agg(" scale=".join, axis=1)
    labels = sorted(keys.unique())
    means = np.stack([x[keys.to_numpy() == k].mean(axis=0) for k in labels])
    label_df = pd.DataFrame([{"label": k, "shape": k.split(" scale=")[0], "scale": float(k.split(" scale=")[1])} for k in labels])
    return means, label_df


def rsa_summary(x: np.ndarray, meta: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    means, labels = class_means(x, meta)
    sim = cosine_similarity(means)
    rsa = pd.DataFrame(sim, index=labels["label"], columns=labels["label"])
    rows = []
    for i, li in labels.iterrows():
        for j, lj in labels.iterrows():
            if i >= j:
                continue
            rows.append(
                {
                    "label_a": li["label"],
                    "label_b": lj["label"],
                    "same_shape": li["shape"] == lj["shape"],
                    "same_scale": li["scale"] == lj["scale"],
                    "similarity": float(sim[i, j]),
                }
            )
    pairs = pd.DataFrame(rows)
    within = pairs[(pairs.same_shape) & (~pairs.same_scale)]["similarity"].mean()
    between = pairs[(~pairs.same_shape)]["similarity"].mean()
    summary = pd.DataFrame(
        [{"within_shape_cross_scale_similarity": within, "between_shape_similarity": between, "scale_invariance_margin": within - between}]
    )
    return rsa, summary


def linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    x = x - x.mean(axis=0, keepdims=True)
    y = y - y.mean(axis=0, keepdims=True)
    if x.shape[0] <= x.shape[1] or y.shape[0] <= y.shape[1]:
        hsic = np.linalg.norm(x @ y.T, ord="fro") ** 2
        denom = np.linalg.norm(x @ x.T, ord="fro") * np.linalg.norm(y @ y.T, ord="fro")
    else:
        hsic = np.linalg.norm(x.T @ y, ord="fro") ** 2
        denom = np.linalg.norm(x.T @ x, ord="fro") * np.linalg.norm(y.T @ y, ord="fro")
    return float(hsic / denom) if denom > 1e-12 else float("nan")


def cka_by_scale(x: np.ndarray, meta: pd.DataFrame) -> pd.DataFrame:
    scales = sorted(meta["scale"].unique())
    rows = []
    for a in scales:
        xa = x[meta["scale"].to_numpy() == a]
        for b in scales:
            xb = x[meta["scale"].to_numpy() == b]
            n = min(len(xa), len(xb))
            rows.append({"scale_a": a, "scale_b": b, "cka": linear_cka(xa[:n], xb[:n])})
    return pd.DataFrame(rows)
