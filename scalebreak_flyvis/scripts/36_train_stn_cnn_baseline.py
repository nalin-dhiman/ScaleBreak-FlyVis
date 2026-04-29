#!/usr/bin/env python
"""Train a lightweight Spatial Transformer CNN baseline on the LOSO task.

This is a reviewer-facing scale-aware control. It reuses the existing
FlyVis-native stimuli, deterministic hex-to-grid projection, and dynamic
direction leave-one-apparent-scale-out protocol. It is intentionally small and
bounded; it does not perform hyperparameter search.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

FLYVIS_ACCURACY = 0.9236111111111112


def find_inputs(outputs: Path) -> tuple[Path, Path, Path]:
    stim = outputs / "flyvis_pilot_v2" / "stimuli" / "stimuli.npy"
    meta = outputs / "flyvis_pilot_v2" / "responses" / "metadata.csv"
    coords = outputs / "flyvis_pilot_v2" / "stimuli" / "hex_coordinates.csv"
    if stim.exists() and meta.exists() and coords.exists():
        return stim, meta, coords
    return next(outputs.rglob("stimuli.npy")), next(outputs.rglob("metadata.csv")), next(outputs.rglob("hex_coordinates.csv"))


def parse_seeds(text: str) -> list[int]:
    return [int(x) for x in text.split(",") if x.strip()]


def encode(values) -> tuple[np.ndarray, dict[int, str]]:
    labels = sorted({str(v) for v in values})
    to_i = {v: i for i, v in enumerate(labels)}
    return np.array([to_i[str(v)] for v in values], dtype=np.int64), {i: v for v, i in to_i.items()}


def build_hex_grid(coords: pd.DataFrame, grid_size: int) -> pd.DataFrame:
    x = coords["x"].to_numpy(dtype=float)
    y = coords["y"].to_numpy(dtype=float)
    gx = np.round((x - x.min()) / max(x.max() - x.min(), 1e-8) * (grid_size - 1)).astype(int)
    gy = np.round((y - y.min()) / max(y.max() - y.min(), 1e-8) * (grid_size - 1)).astype(int)
    return pd.DataFrame({"hex_pixel": np.arange(len(coords)), "grid_x": gx, "grid_y": gy, "source_x": x, "source_y": y})


def temporal_summary_grid(stimuli: np.ndarray, mapping: pd.DataFrame, grid_size: int) -> np.ndarray:
    n, t, _, _ = stimuli.shape
    frame_ids = [0, int(0.25 * (t - 1)), int(0.50 * (t - 1)), int(0.75 * (t - 1)), t - 1]
    raw = [stimuli[:, f, 0, :] for f in frame_ids]
    raw.append(stimuli[:, :, 0, :].mean(axis=1))
    raw.append(stimuli[:, :, 0, :].max(axis=1))
    raw.append(stimuli[:, frame_ids[2], 0, :] - stimuli[:, frame_ids[1], 0, :])
    raw.append(stimuli[:, frame_ids[4], 0, :] - stimuli[:, frame_ids[2], 0, :])
    raw.append(np.abs(np.diff(stimuli[:, :, 0, :], axis=1)).mean(axis=1))
    gx = mapping["grid_x"].to_numpy()
    gy = mapping["grid_y"].to_numpy()
    counts = np.zeros((grid_size, grid_size), dtype=np.float32)
    for y, x in zip(gy, gx):
        counts[y, x] += 1.0
    counts = np.maximum(counts, 1.0)
    out = np.zeros((n, len(raw), grid_size, grid_size), dtype=np.float32)
    for ci, values in enumerate(raw):
        acc = np.zeros((n, grid_size, grid_size), dtype=np.float32)
        for hp, (y, x) in enumerate(zip(gy, gx)):
            acc[:, y, x] += values[:, hp]
        out[:, ci] = acc / counts[None]
    return out


def make_model(n_classes: int, in_channels: int = 10, dropout: float = 0.2):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class STNCNN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.localization = nn.Sequential(
                nn.Conv2d(in_channels, 16, kernel_size=5, padding=2),
                nn.ReLU(True),
                nn.MaxPool2d(2),
                nn.Conv2d(16, 24, kernel_size=3, padding=1),
                nn.ReLU(True),
                nn.AdaptiveAvgPool2d(1),
            )
            self.fc_loc = nn.Sequential(nn.Flatten(), nn.Linear(24, 32), nn.ReLU(True), nn.Linear(32, 6))
            self.fc_loc[-1].weight.data.zero_()
            self.fc_loc[-1].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float32))
            self.features = nn.Sequential(
                nn.Conv2d(in_channels, 32, 3, padding=1, bias=False),
                nn.BatchNorm2d(32),
                nn.ReLU(True),
                nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(True),
                nn.Conv2d(64, 96, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(96),
                nn.ReLU(True),
                nn.Conv2d(96, 128, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(True),
            )
            self.drop = nn.Dropout(dropout)
            self.head = nn.Linear(128, n_classes)

        def stn(self, x):
            theta = self.fc_loc(self.localization(x)).view(-1, 2, 3)
            grid = F.affine_grid(theta, x.size(), align_corners=False)
            return F.grid_sample(x, grid, align_corners=False)

        def forward(self, x):
            x = self.stn(x)
            z = self.features(x).mean(dim=(2, 3))
            return self.head(self.drop(z))

    return STNCNN()


def split_train_val(train_idx: np.ndarray, y: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    from sklearn.model_selection import StratifiedShuffleSplit

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=seed)
    a, b = next(splitter.split(train_idx, y[train_idx]))
    return train_idx[a], train_idx[b]


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def train_fold(x: np.ndarray, y: np.ndarray, train_idx: np.ndarray, test_idx: np.ndarray, seed: int, args, n_classes: int) -> tuple[np.ndarray, np.ndarray, dict, list[dict]]:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

    torch.manual_seed(seed)
    model = make_model(n_classes, in_channels=x.shape[1], dropout=args.dropout)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    tr_idx, val_idx = split_train_val(train_idx, y, seed)
    counts = np.bincount(y[tr_idx], minlength=n_classes).astype(np.float32)
    class_weights = counts.sum() / np.maximum(counts, 1.0)
    class_weights = class_weights / class_weights.mean()
    sampler = WeightedRandomSampler(class_weights[y[tr_idx]], num_samples=len(tr_idx), replacement=True)
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32))
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
    curves: list[dict] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for xb, yb in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
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
            idx = test_idx[start : start + args.batch_size]
            logits = model(torch.tensor(x[idx], dtype=torch.float32))
            probs.append(torch.softmax(logits, dim=1).cpu().numpy())
    prob = np.concatenate(probs, axis=0)
    pred = prob.argmax(axis=1)
    info = {"best_val_accuracy": best_acc, "epochs_ran": len(curves), "early_stopped": len(curves) < args.epochs, "n_val": len(val_idx)}
    return pred, prob, info, curves


def calibration(y_true: np.ndarray, prob: np.ndarray, n_bins: int = 10) -> dict[str, float]:
    onehot = np.eye(prob.shape[1])[y_true]
    brier = float(np.mean(np.sum((prob - onehot) ** 2, axis=1)))
    conf = prob.max(axis=1)
    correct = (prob.argmax(axis=1) == y_true).astype(float)
    ece = 0.0
    for lo in np.linspace(0, 1, n_bins, endpoint=False):
        hi = lo + 1 / n_bins
        mask = (conf >= lo) & (conf < hi if hi < 1 else conf <= hi)
        if mask.any():
            ece += float(mask.mean()) * abs(float(correct[mask].mean()) - float(conf[mask].mean()))
    return {"ece": float(ece), "brier": brier, "mean_confidence": float(conf.mean())}


def bootstrap_ci(vals: np.ndarray, seed: int = 42, n_boot: int = 1000) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    boot = np.array([vals[rng.integers(0, len(vals), len(vals))].mean() for _ in range(n_boot)])
    return float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs-dir", default="scalebreak_flyvis/outputs")
    parser.add_argument("--out-dir", default="scalebreak_flyvis/outputs/stn_cnn_baseline")
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--grid-size", type=int, default=24)
    parser.add_argument("--dropout", type=float, default=0.2)
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    stim_path, meta_path, coord_path = find_inputs(Path(args.outputs_dir))
    meta = pd.read_csv(meta_path)
    coords = pd.read_csv(coord_path)
    stimuli = np.load(stim_path, mmap_mode="r")
    mapping = build_hex_grid(coords, args.grid_size)
    x = temporal_summary_grid(np.asarray(stimuli), mapping, args.grid_size)
    dynamic = meta["feature_family"].isin(["moving_edge", "moving_bar", "small_translating_target"]).to_numpy()
    y_dyn, inv = encode(meta.loc[dynamic, "direction"].to_numpy())
    y = np.full(len(meta), -1, dtype=np.int64)
    y[np.flatnonzero(dynamic)] = y_dyn
    rows, pred_rows, curve_rows = [], [], []
    for seed in parse_seeds(args.seeds):
        for heldout in sorted(meta.loc[dynamic, "scale"].unique()):
            train_idx = np.flatnonzero(dynamic & (meta["scale"].to_numpy() != heldout))
            test_idx = np.flatnonzero(dynamic & (meta["scale"].to_numpy() == heldout))
            pred, prob, info, curves = train_fold(x, y, train_idx, test_idx, seed + int(heldout), args, n_classes=len(inv))
            m = metrics(y[test_idx], pred)
            cal = calibration(y[test_idx], prob)
            rows.append({"model": "STN-CNN", "seed": seed, "heldout_scale": heldout, **m, **cal, "n_train": len(train_idx), "n_test": len(test_idx), **info})
            for c in curves:
                curve_rows.append({"model": "STN-CNN", "seed": seed, "heldout_scale": heldout, **c})
            for sample, yt, yp, pr in zip(test_idx, y[test_idx], pred, prob):
                row = {"model": "STN-CNN", "seed": seed, "heldout_scale": heldout, "sample": int(sample), "true_label": inv[int(yt)], "pred_label": inv[int(yp)], "correct": bool(yt == yp), "confidence": float(pr.max())}
                for i, p in enumerate(pr):
                    row[f"prob_{inv[i]}"] = float(p)
                pred_rows.append(row)
    table = pd.DataFrame(rows)
    preds = pd.DataFrame(pred_rows)
    curves = pd.DataFrame(curve_rows)
    table.to_csv(out / "table_stn_cnn_by_seed_scale.csv", index=False)
    preds.to_csv(out / "predictions_stn_cnn.csv", index=False)
    curves.to_csv(out / "training_curves.csv", index=False)
    mean_by_seed = preds.groupby("seed")["correct"].mean()
    lo, hi = bootstrap_ci(preds["correct"].astype(float).to_numpy())
    prob_cols = [c for c in preds.columns if c.startswith("prob_")]
    label_to_i = {label: i for i, label in inv.items()}
    y_true = np.array([label_to_i[str(v)] for v in preds["true_label"]], dtype=int)
    cal = calibration(y_true, preds[prob_cols].to_numpy())
    summary = pd.DataFrame(
        [
            {
                "model": "STN-CNN",
                "mean_offdiag_accuracy": float(mean_by_seed.mean()),
                "std": float(mean_by_seed.std(ddof=1)) if len(mean_by_seed) > 1 else 0.0,
                "ci_low": lo,
                "ci_high": hi,
                **cal,
                "flyvis_accuracy": FLYVIS_ACCURACY,
                "flyvis_minus_model": FLYVIS_ACCURACY - float(mean_by_seed.mean()),
                "n_seeds": int(len(mean_by_seed)),
            }
        ]
    )
    summary.to_csv(out / "table_stn_cnn_summary.csv", index=False)
    (out / "REPORT.md").write_text(
        "\n".join(
            [
                "# STN-CNN Baseline Report",
                "",
                "This is a bounded reviewer-facing scale-aware baseline. It uses the same LOSO dynamic direction protocol and deterministic hex-to-grid projection as the serious CNN baseline.",
                "",
                f"Mean off-diagonal accuracy: {float(summary['mean_offdiag_accuracy'].iloc[0]):.3f}",
                f"95% bootstrap CI: [{lo:.3f}, {hi:.3f}]",
                f"ECE: {cal['ece']:.3f}",
                f"Brier score: {cal['brier']:.3f}",
                f"FlyVis minus STN-CNN: {float(summary['flyvis_minus_model'].iloc[0]):.3f}",
                "",
                "This does not resolve the need for future hex-native or fully scale-equivariant baselines.",
            ]
        ),
        encoding="utf-8",
    )
    (out / "run_info.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "python": sys.version,
                "platform": platform.platform(),
                "stimuli_path": str(stim_path),
                "metadata_path": str(meta_path),
                "coords_path": str(coord_path),
                "stimulus_shape": list(stimuli.shape),
                "input_tensor_shape": list(x.shape),
                "elapsed_seconds": time.time() - t0,
                "args": vars(args),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote STN-CNN baseline to {out}")


if __name__ == "__main__":
    main()
