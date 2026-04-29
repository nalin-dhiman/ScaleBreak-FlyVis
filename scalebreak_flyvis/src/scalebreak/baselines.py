"""Control and graph-ablation helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd


def nuisance_features(meta: pd.DataFrame, kind: str) -> np.ndarray:
    if kind == "area":
        cols = ["area_pixels"]
    elif kind == "edge":
        cols = ["edge_length_pixels"]
    elif kind == "area_contrast":
        cols = ["area_pixels", "contrast"]
    else:
        cols = ["area_pixels", "edge_length_pixels", "contrast"]
    return meta[cols].to_numpy(dtype=np.float32)


def random_sparse_type_edges(type_edges: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    types = sorted(set(type_edges["pre_type"]) | set(type_edges["post_type"]))
    weights = type_edges["total_weight"].sample(frac=1.0, random_state=seed).to_numpy()
    out = type_edges.copy()
    out["pre_type"] = rng.choice(types, size=len(out))
    out["post_type"] = rng.choice(types, size=len(out))
    out["total_weight"] = weights
    return out


def shuffled_type_labels(type_edges: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    types = np.array(sorted(set(type_edges["pre_type"]) | set(type_edges["post_type"])))
    shuffled = types.copy()
    rng.shuffle(shuffled)
    mapping = dict(zip(types, shuffled))
    out = type_edges.copy()
    out["pre_type"] = out["pre_type"].map(mapping)
    out["post_type"] = out["post_type"].map(mapping)
    return out
