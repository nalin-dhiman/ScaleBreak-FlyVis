#!/usr/bin/env python
"""Train a lightweight hex-native temporal baseline for LOSO direction decoding.

The model consumes raw FlyVis-native sequences with shape (T, 721). It avoids
hex-to-square-grid projection by using six-neighbor message passing on the
retinal hex coordinates, followed by temporal 1D convolutions and a linear
classifier. This is a bounded reviewer-facing control, not an exhaustive
architecture search.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder

FLYVIS_ACCURACY = 0.9236111111111112


def parse_ints(text: str) -> list[int]:
    return [int(v) for v in text.split(",") if v.strip()]


def nearest_neighbors(coords: pd.DataFrame, k: int = 6) -> np.ndarray:
    """Return k nearest hex neighbors for each pixel using coordinate distance."""

    xy = coords[["x", "y"]].to_numpy(dtype=np.float32)
    dist = ((xy[:, None, :] - xy[None, :, :]) ** 2).sum(axis=-1)
    order = np.argsort(dist, axis=1)
    return order[:, 1 : k + 1].astype(np.int64)


def bootstrap_ci(correct: np.ndarray, seed: int = 42, n_boot: int = 1000) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    boot = np.array([correct[rng.integers(0, len(correct), len(correct))].mean() for _ in range(n_boot)])
    return float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def make_model(n_pixels: int, n_classes: int, neighbor_idx: np.ndarray, hidden: int, dropout: float):
    import torch
    import torch.nn as nn

    class HexConvTemporalModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.register_buffer("neighbor_idx", torch.tensor(neighbor_idx, dtype=torch.long))
            self.temporal = nn.Sequential(
                nn.Conv1d(n_pixels * 2, hidden, kernel_size=7, padding=3, bias=False),
                nn.GroupNorm(8, hidden),
                nn.ReLU(inplace=True),
                nn.Conv1d(hidden, hidden, kernel_size=5, padding=2, bias=False),
                nn.GroupNorm(8, hidden),
                nn.ReLU(inplace=True),
                nn.Conv1d(hidden, hidden, kernel_size=3, padding=1, bias=False),
                nn.GroupNorm(8, hidden),
                nn.ReLU(inplace=True),
            )
            self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden * 2, n_classes))

        def forward(self, x):
            # x: (batch, time, hex_pixel). The neighbor average is a one-step
            # graph-convolution message on the native retinal lattice.
            neigh = x[:, :, self.neighbor_idx].mean(dim=-1)
            z = torch.cat([x, neigh], dim=-1).transpose(1, 2)
            h = self.temporal(z)
            pooled = torch.cat([h.mean(dim=-1), h.amax(dim=-1)], dim=1)
            return self.head(pooled)

    return HexConvTemporalModel()


def split_train_val(train_idx: np.ndarray, y: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=seed)
    a, b = next(splitter.split(train_idx, y[train_idx]))
    return train_idx[a], train_idx[b]


def train_fold(x: np.ndarray, y: np.ndarray, train_idx: np.ndarray, test_idx: np.ndarray, seed: int, args, neighbor_idx: np.ndarray):
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

    torch.manual_seed(seed)
    model = make_model(x.shape[-1], int(y.max() + 1), neighbor_idx, args.hidden, args.dropout)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    tr_idx, val_idx = split_train_val(train_idx, y, seed)
    counts = np.bincount(y[tr_idx], minlength=int(y.max() + 1)).astype(np.float32)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = weights / weights.mean()
    sampler = WeightedRandomSampler(weights[y[tr_idx]], len(tr_idx), replacement=True)
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32))
    loader = DataLoader(
        TensorDataset(torch.tensor(x[tr_idx], dtype=torch.float32), torch.tensor(y[tr_idx], dtype=torch.long)),
        batch_size=args.batch_size,
        sampler=sampler,
    )
    val_x = torch.tensor(x[val_idx], dtype=torch.float32)
    val_y = torch.tensor(y[val_idx], dtype=torch.long)
    best_acc = -1.0
    best_state = None
    stale = 0
    curves = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for xb, yb in loader:
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():
            val_pred = model(val_x).argmax(1).cpu().numpy()
        val_acc = float((val_pred == val_y.cpu().numpy()).mean())
        curves.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "val_accuracy": val_acc})
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= args.patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    probs = []
    with torch.no_grad():
        for start in range(0, len(test_idx), args.batch_size):
            batch = torch.tensor(x[test_idx[start : start + args.batch_size]], dtype=torch.float32)
            probs.append(torch.softmax(model(batch), dim=1).cpu().numpy())
    prob = np.concatenate(probs)
    pred = prob.argmax(axis=1)
    return pred, prob, {"best_val_accuracy": best_acc, "epochs_ran": len(curves), "early_stopped": len(curves) < args.epochs, "n_val": len(val_idx)}, curves


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs-dir", default="scalebreak_flyvis/outputs")
    parser.add_argument("--out-dir", default="scalebreak_flyvis/outputs/hex_native_temporal_baseline")
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=96)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=4)
    args = parser.parse_args()

    outputs = Path(args.outputs_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    stimuli = np.load(outputs / "flyvis_pilot_v2" / "stimuli" / "stimuli.npy", mmap_mode="r")
    meta = pd.read_csv(outputs / "flyvis_pilot_v2" / "responses" / "metadata.csv")
    coords = pd.read_csv(outputs / "flyvis_pilot_v2" / "stimuli" / "hex_coordinates.csv")
    dynamic = meta["feature_family"].isin(["moving_edge", "moving_bar", "small_translating_target"]).to_numpy()
    dyn_idx = np.flatnonzero(dynamic)
    x = np.asarray(stimuli[dyn_idx, :, 0, :], dtype=np.float32) - 0.5
    le = LabelEncoder()
    y = le.fit_transform(meta.loc[dynamic, "direction"].astype(str)).astype(np.int64)
    scales = meta.loc[dynamic, "scale"].to_numpy()
    neighbor_idx = nearest_neighbors(coords, k=6)
    pd.DataFrame(neighbor_idx).to_csv(out / "hex_neighbor_indices.csv", index=False)

    rows, pred_rows, curve_rows = [], [], []
    for seed in parse_ints(args.seeds):
        for heldout in sorted(np.unique(scales)):
            train_idx = np.flatnonzero(scales != heldout)
            test_idx = np.flatnonzero(scales == heldout)
            pred, prob, info, curves = train_fold(x, y, train_idx, test_idx, seed + int(heldout), args, neighbor_idx)
            row = {
                "model": "Hex-native temporal model",
                "seed": seed,
                "heldout_scale": float(heldout),
                "accuracy": float(accuracy_score(y[test_idx], pred)),
                "balanced_accuracy": float(balanced_accuracy_score(y[test_idx], pred)),
                "macro_f1": float(f1_score(y[test_idx], pred, average="macro", zero_division=0)),
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
                **info,
            }
            rows.append(row)
            for c in curves:
                curve_rows.append({"seed": seed, "heldout_scale": float(heldout), **c})
            for idx, true, pp in zip(test_idx, y[test_idx], prob):
                rec = {
                    "seed": seed,
                    "sample": int(dyn_idx[idx]),
                    "heldout_scale": float(heldout),
                    "true_label": str(le.inverse_transform([true])[0]),
                    "pred_label": str(le.inverse_transform([int(pp.argmax())])[0]),
                    "correct": bool(int(pp.argmax()) == true),
                }
                for i, label in enumerate(le.classes_):
                    rec[f"prob_{label}"] = float(pp[i])
                pred_rows.append(rec)

    by_scale = pd.DataFrame(rows)
    by_scale.to_csv(out / "table_hex_native_by_seed_scale.csv", index=False)
    pd.DataFrame(curve_rows).to_csv(out / "training_curves.csv", index=False)
    preds = pd.DataFrame(pred_rows)
    preds.to_csv(out / "predictions_hex_native.csv", index=False)
    correct = preds["correct"].astype(float).to_numpy()
    lo, hi = bootstrap_ci(correct)
    summary = pd.DataFrame(
        [
            {
                "model": "Hex-native temporal model",
                "mean_offdiag_accuracy": float(correct.mean()),
                "std": float(by_scale["accuracy"].std(ddof=0)),
                "ci_low": lo,
                "ci_high": hi,
                "flyvis_accuracy": FLYVIS_ACCURACY,
                "flyvis_minus_model": FLYVIS_ACCURACY - float(correct.mean()),
                "n_seeds": len(parse_ints(args.seeds)),
            }
        ]
    )
    summary.to_csv(out / "table_hex_native_summary.csv", index=False)
    try:
        summary.to_markdown(out / "table_hex_native_summary.md", index=False)
    except Exception:
        (out / "table_hex_native_summary.md").write_text("```csv\n" + summary.to_csv(index=False) + "```\n", encoding="utf-8")
    (out / "run_info.json").write_text(
        json.dumps(
            {
                "python": platform.python_version(),
                "runtime_seconds": time.time() - t0,
                "args": vars(args),
                "classes": list(map(str, le.classes_)),
                "stimulus_shape": list(stimuli.shape),
                "n_dynamic_trials": int(len(dyn_idx)),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (out / "REPORT.md").write_text(
        f"# Hex-native Temporal Baseline Report\n\n"
        f"Mean off-diagonal accuracy: {float(correct.mean()):.3f}\n"
        f"95% bootstrap CI: [{lo:.3f}, {hi:.3f}]\n"
        f"FlyVis minus model: {FLYVIS_ACCURACY - float(correct.mean()):.3f}\n\n"
        "This baseline consumes raw FlyVis-native hex sequences and uses six-neighbor message passing plus temporal convolution. "
        "It removes the hex-to-grid projection disadvantage but remains a bounded reviewer-facing control, not a full artificial-vision model search.\n",
        encoding="utf-8",
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
