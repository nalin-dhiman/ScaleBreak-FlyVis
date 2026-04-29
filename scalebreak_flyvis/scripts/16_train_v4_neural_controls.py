#!/usr/bin/env python
"""Train lightweight neural controls for FlyVis Pilot v4.

This script uses the already-generated FlyVis Pilot v2 stimuli. It does not
regenerate stimuli. The controls are intentionally small CPU-friendly models
trained with the same leave-one-scale-out direction decoding protocol used by
the v4 report.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def make_binned_square_movies(stimuli: np.ndarray, coords: pd.DataFrame, size: int = 16, bins: int = 11) -> np.ndarray:
    """Map hex-pixel movies to small square temporal-bin tensors."""

    x = coords["x"].to_numpy()
    y = coords["y"].to_numpy()
    xi = np.round((x - x.min()) / (x.max() - x.min()) * (size - 1)).astype(int)
    yi = np.round((y - y.min()) / (y.max() - y.min()) * (size - 1)).astype(int)
    frame_bins = np.array_split(np.arange(stimuli.shape[1]), bins)
    out = np.full((stimuli.shape[0], bins, size, size), 0.5, dtype=np.float32)
    for bi, frames in enumerate(frame_bins):
        vals = stimuli[:, frames, 0, :].mean(axis=1)
        for hp in range(len(xi)):
            out[:, bi, yi[hp], xi[hp]] = vals[:, hp]
    return out


def encode_labels(values: np.ndarray) -> tuple[np.ndarray, dict[int, str]]:
    values_str = np.array([str(v) for v in values], dtype=object)
    labels = sorted(set(values_str.tolist()))
    mapping = {label: i for i, label in enumerate(labels)}
    inv = {i: label for label, i in mapping.items()}
    return np.array([mapping[v] for v in values_str], dtype=np.int64), inv


def train_one_fold(
    x: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    model_kind: str,
    n_classes: int,
    seed: int,
    epochs: int,
    batch_size: int,
) -> np.ndarray:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(seed)

    class TinyTemporalCNN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv1 = nn.Conv3d(1, 8, kernel_size=(3, 3, 3), padding=1)
            self.conv2 = nn.Conv3d(8, 16, kernel_size=(3, 3, 3), padding=1)
            self.head = nn.Linear(16, n_classes)

        def forward(self, z: torch.Tensor) -> torch.Tensor:
            z = z[:, None]
            z = F.relu(self.conv1(z))
            z = F.relu(self.conv2(z))
            z = z.mean(dim=(2, 3, 4))
            return self.head(z)

    class TinyLocalConvRNN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.inp = nn.Conv2d(1, 12, kernel_size=3, padding=1)
            self.rec = nn.Conv2d(12, 12, kernel_size=3, padding=1, groups=12)
            self.head = nn.Linear(12, n_classes)

        def forward(self, z: torch.Tensor) -> torch.Tensor:
            h = torch.zeros((z.shape[0], 12, z.shape[2], z.shape[3]), dtype=z.dtype, device=z.device)
            for ti in range(z.shape[1]):
                drive = self.inp(z[:, ti : ti + 1])
                h = 0.65 * h + 0.35 * F.relu(drive + self.rec(h))
            return self.head(h.mean(dim=(2, 3)))

    model = TinyTemporalCNN() if model_kind == "trained small CNN" else TinyLocalConvRNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    counts = np.bincount(y[train_idx], minlength=n_classes).astype(np.float32)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = weights / weights.mean()
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32))
    train_ds = TensorDataset(torch.tensor(x[train_idx], dtype=torch.float32), torch.tensor(y[train_idx], dtype=torch.long))
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
    model.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, len(test_idx), batch_size):
            idx = test_idx[start : start + batch_size]
            logits = model(torch.tensor(x[idx], dtype=torch.float32))
            preds.append(logits.argmax(dim=1).cpu().numpy())
    return np.concatenate(preds)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v2-dir", default="scalebreak_flyvis/outputs/flyvis_pilot_v2")
    parser.add_argument("--v4-dir", default="scalebreak_flyvis/outputs/flyvis_pilot_v4")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    root = Path.cwd()
    v2 = root / args.v2_dir
    v4 = root / args.v4_dir
    tables = v4 / "tables"
    tables.mkdir(parents=True, exist_ok=True)

    meta = pd.read_csv(v2 / "responses" / "metadata.csv")
    stimuli = np.load(v2 / "stimuli" / "stimuli.npy", mmap_mode="r")
    coords = pd.read_csv(v2 / "stimuli" / "hex_coordinates.csv")
    subset = meta["feature_family"].isin(["moving_edge", "moving_bar", "small_translating_target"]).to_numpy()
    x = make_binned_square_movies(np.asarray(stimuli), coords)
    y, inv = encode_labels(meta["direction"].to_numpy())

    pred_rows = []
    summary_rows = []
    for model_kind in ["trained small CNN", "trained local ConvRNN"]:
        for heldout in sorted(meta.loc[subset, "scale"].unique()):
            train_idx = np.flatnonzero(subset & (meta["scale"].to_numpy() != heldout))
            test_idx = np.flatnonzero(subset & (meta["scale"].to_numpy() == heldout))
            pred = train_one_fold(
                x,
                y,
                train_idx,
                test_idx,
                model_kind,
                len(inv),
                args.seed + int(heldout),
                args.epochs,
                args.batch_size,
            )
            true = y[test_idx]
            for sample, yt, yp in zip(test_idx, true, pred):
                pred_rows.append(
                    {
                        "model": model_kind,
                        "target": "direction",
                        "sample": int(sample),
                        "heldout_scale": float(heldout),
                        "true_label": inv[int(yt)],
                        "pred_label": inv[int(yp)],
                        "correct": bool(yt == yp),
                    }
                )
        model_pred = [r for r in pred_rows if r["model"] == model_kind]
        summary_rows.append(
            {
                "model": model_kind,
                "offdiag_accuracy": float(np.mean([r["correct"] for r in model_pred])),
                "n": len(model_pred),
                "epochs": args.epochs,
                "input_tensor_shape": f"{tuple(x.shape)}",
            }
        )

    pred_df = pd.DataFrame(pred_rows)
    summary = pd.DataFrame(summary_rows)
    pred_df.to_csv(tables / "trained_neural_control_predictions.csv", index=False)
    summary.to_csv(tables / "table_v4_trained_neural_controls.csv", index=False)

    main_path = tables / "table_v4_main_results.csv"
    if main_path.exists():
        main = pd.read_csv(main_path)
        main = pd.concat([main[~main["model"].isin(summary["model"])], summary[["model", "offdiag_accuracy", "n"]]], ignore_index=True)
        main.sort_values("offdiag_accuracy", ascending=False).to_csv(main_path, index=False)

    report_path = v4 / "REPORT.md"
    if report_path.exists():
        with report_path.open("a", encoding="utf-8") as f:
            f.write("\n## Additional Trained Neural Controls\n")
            f.write(summary.to_csv(index=False))
            f.write(
                "\nThese are lightweight temporal CNN/local ConvRNN classifiers trained on the same held-out-scale "
                "direction protocol. They complement the fixed-representation CNN/local-RNN controls used in the main "
                "feature-family and representation tables.\n"
            )
    print(f"Wrote trained neural controls to {tables / 'table_v4_trained_neural_controls.csv'}")


if __name__ == "__main__":
    main()
