#!/usr/bin/env python
"""Run a fast pixel robustness control using spatially scrambled stimuli.

The control preserves per-frame intensity distributions but randomly permutes
hex-pixel positions independently per trial/frame. It tests whether the pixel
baseline is relying on simple intensity/edge-energy summaries rather than
organized motion structure. This does not run FlyVis.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scalebreak.models import pixel_baseline  # noqa: E402


def loso_accuracy(x: np.ndarray, meta: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    dynamic = meta["feature_family"].isin(["moving_edge", "moving_bar", "small_translating_target"]).to_numpy()
    le = LabelEncoder()
    y = np.full(len(meta), -1, dtype=int)
    y[dynamic] = le.fit_transform(meta.loc[dynamic, "direction"].astype(str))
    rows = []
    for heldout in sorted(meta.loc[dynamic, "scale"].unique()):
        train = dynamic & (meta["scale"].to_numpy() != heldout)
        test = dynamic & (meta["scale"].to_numpy() == heldout)
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed, n_jobs=1),
        )
        clf.fit(x[train], y[train])
        pred = clf.predict(x[test])
        rows.append({"heldout_scale": heldout, "accuracy": float(accuracy_score(y[test], pred)), "n_test": int(test.sum())})
    return pd.DataFrame(rows)


def scramble_frames(stimuli: np.ndarray, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = np.asarray(stimuli).copy()
    n, t, c, p = out.shape
    flat = out.reshape(n * t * c, p)
    for i in range(flat.shape[0]):
        flat[i] = flat[i, rng.permutation(p)]
    return flat.reshape(out.shape)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs-dir", default="scalebreak_flyvis/outputs")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    outputs = Path(args.outputs_dir)
    out = outputs / "pixel_robustness_control"
    out.mkdir(parents=True, exist_ok=True)
    stim_path = outputs / "flyvis_pilot_v2" / "stimuli" / "stimuli.npy"
    meta_path = outputs / "flyvis_pilot_v2" / "responses" / "metadata.csv"
    stimuli = np.load(stim_path, mmap_mode="r")
    meta = pd.read_csv(meta_path)
    original_x = pixel_baseline(np.asarray(stimuli))
    scrambled = scramble_frames(stimuli, seed=args.seed)
    scrambled_x = pixel_baseline(scrambled)
    rows = []
    for label, x in [("original_pixel", original_x), ("spatially_scrambled_pixel", scrambled_x)]:
        df = loso_accuracy(x, meta, seed=args.seed)
        df.insert(0, "control", label)
        rows.append(df)
    table = pd.concat(rows, ignore_index=True)
    table.to_csv(out / "table_pixel_spatial_scramble_loso.csv", index=False)
    summary = table.groupby("control")["accuracy"].mean().reset_index(name="mean_loso_accuracy")
    summary.to_csv(out / "table_pixel_spatial_scramble_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
