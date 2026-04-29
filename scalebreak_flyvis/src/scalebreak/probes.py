"""Linear probe protocols and nuisance controls."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler


def classifier(seed: int = 0) -> Any:
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed, n_jobs=1),
    )


def stratified_accuracy(x: np.ndarray, y: np.ndarray, seed: int = 0, n_splits: int = 5) -> tuple[float, float]:
    le = LabelEncoder()
    yy = le.fit_transform(y)
    counts = np.bincount(yy)
    splits = max(2, min(n_splits, int(counts.min())))
    if splits < 2:
        return float("nan"), float("nan")
    scores = []
    cv = StratifiedKFold(n_splits=splits, shuffle=True, random_state=seed)
    for tr, te in cv.split(x, yy):
        clf = classifier(seed)
        clf.fit(x[tr], yy[tr])
        scores.append(accuracy_score(yy[te], clf.predict(x[te])))
    return float(np.mean(scores)), float(np.std(scores))


def scale_generalization_matrix(x: np.ndarray, meta: pd.DataFrame, target: str = "shape", seed: int = 0) -> pd.DataFrame:
    scales = sorted(meta["scale"].unique())
    le = LabelEncoder()
    y = le.fit_transform(meta[target].astype(str))
    rows = []
    for train_scale in scales:
        tr = meta["scale"].to_numpy() == train_scale
        if len(np.unique(y[tr])) < 2:
            continue
        clf = classifier(seed)
        clf.fit(x[tr], y[tr])
        for test_scale in scales:
            te = meta["scale"].to_numpy() == test_scale
            rows.append(
                {
                    "train_scale": train_scale,
                    "test_scale": test_scale,
                    "target": target,
                    "accuracy": accuracy_score(y[te], clf.predict(x[te])),
                    "n_train": int(tr.sum()),
                    "n_test": int(te.sum()),
                }
            )
    return pd.DataFrame(rows)


def protocol_metrics(x: np.ndarray, meta: pd.DataFrame, targets: list[str], seed: int = 0) -> tuple[pd.DataFrame, dict[str, Any], dict[str, np.ndarray]]:
    rows = []
    reports: dict[str, Any] = {}
    cms: dict[str, np.ndarray] = {}
    for target in targets:
        y_str = meta[target].astype(str).to_numpy()
        acc, std = stratified_accuracy(x, y_str, seed=seed)
        rows.append({"protocol": "stratified_cv", "target": target, "accuracy": acc, "accuracy_std": std, "n": len(y_str)})
        le = LabelEncoder()
        y = le.fit_transform(y_str)
        tr, te = train_test_split(np.arange(len(y)), test_size=0.25, stratify=y, random_state=seed)
        clf = classifier(seed)
        clf.fit(x[tr], y[tr])
        pred = clf.predict(x[te])
        reports[target] = classification_report(y[te], pred, target_names=le.classes_, output_dict=True, zero_division=0)
        cms[target] = confusion_matrix(y[te], pred)
        if target == "shape" and "scale" in meta:
            scales = np.array(sorted(meta["scale"].unique()))
            for held in scales:
                train = meta["scale"].to_numpy() != held
                test = ~train
                if len(np.unique(y[train])) >= 2 and len(np.unique(y[test])) >= 2:
                    clf = classifier(seed)
                    clf.fit(x[train], y[train])
                    rows.append(
                        {
                            "protocol": "leave_one_scale_out",
                            "target": target,
                            "heldout_scale": held,
                            "accuracy": accuracy_score(y[test], clf.predict(x[test])),
                            "accuracy_std": np.nan,
                            "n": int(test.sum()),
                        }
                    )
            small = scales[: max(1, len(scales) // 3)]
            medium = scales[len(scales) // 3 : max(len(scales) // 3 + 1, 2 * len(scales) // 3)]
            large = scales[max(1, 2 * len(scales) // 3) :]
            transfers = {
                "train_medium_test_small_large": (medium, np.concatenate([small, large])),
                "train_large_test_small": (large, small),
                "train_small_test_large": (small, large),
            }
            for name, (train_scales, test_scales) in transfers.items():
                train = meta["scale"].isin(train_scales).to_numpy()
                test = meta["scale"].isin(test_scales).to_numpy()
                if train.any() and test.any() and len(np.unique(y[train])) >= 2 and len(np.unique(y[test])) >= 2:
                    clf = classifier(seed)
                    clf.fit(x[train], y[train])
                    rows.append(
                        {
                            "protocol": name,
                            "target": target,
                            "accuracy": accuracy_score(y[test], clf.predict(x[test])),
                            "accuracy_std": np.nan,
                            "n": int(test.sum()),
                        }
                    )
    return pd.DataFrame(rows), reports, cms


def cap_features_by_variance(x: np.ndarray, max_features: int | None = None) -> np.ndarray:
    if max_features is None or x.shape[1] <= max_features:
        return x
    variances = np.nanvar(x, axis=0)
    keep = np.argsort(variances)[-max_features:]
    keep.sort()
    return x[:, keep]
