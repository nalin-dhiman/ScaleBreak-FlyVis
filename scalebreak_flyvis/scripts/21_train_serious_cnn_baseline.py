#!/usr/bin/env python
"""Manuscript-grade TemporalResNet18Small baseline for ScaleBreak-FlyVis.

This trains one serious artificial vision baseline on the same dynamic
leave-one-apparent-scale-out direction task used for FlyVis. Quick mode is only
for smoke testing; the default settings are the manuscript-grade run.
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

FLYVIS_ACCURACY = 0.9236111111111112


def setup_logging(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(out / "run.log", mode="a")],
    )


def write_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)


def parse_seeds(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def find_inputs(outputs: Path) -> tuple[Path, Path, Path]:
    stim = outputs / "flyvis_pilot_v2" / "stimuli" / "stimuli.npy"
    meta = outputs / "flyvis_pilot_v2" / "responses" / "metadata.csv"
    coords = outputs / "flyvis_pilot_v2" / "stimuli" / "hex_coordinates.csv"
    if stim.exists() and meta.exists() and coords.exists():
        return stim, meta, coords
    return next(outputs.rglob("stimuli.npy")), next(outputs.rglob("metadata.csv")), next(outputs.rglob("hex_coordinates.csv"))


def encode(values: Iterable) -> tuple[np.ndarray, dict[int, str]]:
    labels = sorted({str(v) for v in values})
    to_i = {v: i for i, v in enumerate(labels)}
    inv = {i: v for v, i in to_i.items()}
    return np.array([to_i[str(v)] for v in values], dtype=np.int64), inv


def build_hex_grid(coords: pd.DataFrame, grid_size: int) -> pd.DataFrame:
    x = coords["x"].to_numpy(dtype=float)
    y = coords["y"].to_numpy(dtype=float)
    gx = np.round((x - x.min()) / max(x.max() - x.min(), 1e-8) * (grid_size - 1)).astype(int)
    gy = np.round((y - y.min()) / max(y.max() - y.min(), 1e-8) * (grid_size - 1)).astype(int)
    return pd.DataFrame({"hex_pixel": np.arange(len(coords)), "grid_x": gx, "grid_y": gy, "source_x": x, "source_y": y})


def temporal_summary_grid(stimuli: np.ndarray, mapping: pd.DataFrame, grid_size: int) -> np.ndarray:
    """Convert `(N,T,1,Hhex)` to 10-channel image tensor `(N,10,G,G)`."""

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


def plot_grid_audit(mapping: pd.DataFrame, out: Path, grid_size: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    occ = np.zeros((grid_size, grid_size), dtype=float)
    for y, x in zip(mapping["grid_y"], mapping["grid_x"]):
        occ[int(y), int(x)] += 1
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(occ, cmap="magma")
    fig.colorbar(im, ax=ax, label="hex pixels per grid cell")
    ax.set_title("Hex-to-grid audit")
    fig.tight_layout()
    fig.savefig(out / "hex_to_grid_audit.png", dpi=180)
    plt.close(fig)


def make_model(n_classes: int, dropout: float = 0.2):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class Block(nn.Module):
        def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
            super().__init__()
            self.a = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
            self.b = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
            self.n1 = nn.BatchNorm2d(out_ch)
            self.n2 = nn.BatchNorm2d(out_ch)
            self.skip = nn.Identity() if in_ch == out_ch and stride == 1 else nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False)

        def forward(self, x):
            y = F.relu(self.n1(self.a(x)))
            y = self.n2(self.b(y))
            return F.relu(y + self.skip(x))

    class TemporalResNet18Small(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.stem = nn.Sequential(nn.Conv2d(10, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32), nn.ReLU())
            self.body = nn.Sequential(
                Block(32, 32),
                Block(32, 32),
                Block(32, 64, stride=2),
                Block(64, 64),
                Block(64, 96, stride=2),
                Block(96, 96),
                Block(96, 128, stride=2),
                Block(128, 128),
            )
            self.drop = nn.Dropout(dropout)
            self.head = nn.Linear(128, n_classes)

        def forward(self, x):
            z = self.body(self.stem(x)).mean(dim=(2, 3))
            return self.head(self.drop(z))

    return TemporalResNet18Small()


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def split_train_val(train_idx: np.ndarray, y: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    from sklearn.model_selection import StratifiedShuffleSplit

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=seed)
    a, b = next(splitter.split(train_idx, y[train_idx]))
    return train_idx[a], train_idx[b]


def train_fold(x: np.ndarray, y: np.ndarray, train_idx: np.ndarray, test_idx: np.ndarray, seed: int, args) -> tuple[np.ndarray, dict, list[dict], object]:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

    torch.manual_seed(seed)
    model = make_model(len(np.unique(y)))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    tr_idx, val_idx = split_train_val(train_idx, y, seed)
    counts = np.bincount(y[tr_idx], minlength=len(np.unique(y))).astype(np.float32)
    class_weights = counts.sum() / np.maximum(counts, 1.0)
    class_weights = class_weights / class_weights.mean()
    sample_weights = class_weights[y[tr_idx]]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(tr_idx), replacement=True)
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32))
    loader = DataLoader(
        TensorDataset(torch.tensor(x[tr_idx], dtype=torch.float32), torch.tensor(y[tr_idx], dtype=torch.long)),
        batch_size=args.batch_size,
        sampler=sampler,
    )
    val_x = torch.tensor(x[val_idx], dtype=torch.float32)
    val_y = torch.tensor(y[val_idx], dtype=torch.long)
    best = -np.inf
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
            losses.append(float(loss.detach().cpu()))
        scheduler.step()
        model.eval()
        with torch.no_grad():
            val_pred = model(val_x).argmax(dim=1).cpu().numpy()
        val_acc = float((val_pred == val_y.cpu().numpy()).mean())
        curves.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "val_accuracy": val_acc, "lr": float(scheduler.get_last_lr()[0])})
        if val_acc > best:
            best = val_acc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= args.patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    preds = []
    import torch

    with torch.no_grad():
        for start in range(0, len(test_idx), args.batch_size):
            idx = test_idx[start : start + args.batch_size]
            preds.append(model(torch.tensor(x[idx], dtype=torch.float32)).argmax(dim=1).cpu().numpy())
    info = {"n_val": len(val_idx), "best_val_accuracy": best, "epochs_ran": len(curves), "early_stopped": len(curves) < args.epochs}
    return np.concatenate(preds), info, curves, model


def bootstrap_summary(preds: pd.DataFrame, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    vals = preds["correct"].astype(float).to_numpy()
    boots = np.array([vals[rng.integers(0, len(vals), len(vals))].mean() for _ in range(1000)])
    return float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))


def plot_outputs(out: Path, table: pd.DataFrame, summary: pd.DataFrame, curves: pd.DataFrame, all_preds: pd.DataFrame) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    acc = float(summary["mean_offdiag_accuracy"].iloc[0])
    lo = float(summary["ci_low"].iloc[0])
    hi = float(summary["ci_high"].iloc[0])
    ax.bar(["TemporalResNet18Small", "FlyVis"], [acc, FLYVIS_ACCURACY], yerr=[[acc - lo, 0], [hi - acc, 0]])
    ax.set_ylabel("Held-out apparent-scale direction accuracy")
    fig.tight_layout()
    fig.savefig(fig_dir / "fig_serious_cnn_vs_flyvis.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    by = table.groupby("heldout_scale")["accuracy"].agg(["mean", "std"]).reset_index()
    ax.errorbar(by["heldout_scale"], by["mean"], yerr=by["std"].fillna(0), marker="o")
    ax.axhline(FLYVIS_ACCURACY, color="black", linestyle="--", label="FlyVis")
    ax.set_xlabel("Held-out apparent scale")
    ax.set_ylabel("Accuracy")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "fig_serious_cnn_by_scale.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    for seed, df in curves.groupby("seed"):
        ax.plot(df.groupby("epoch")["val_accuracy"].mean(), label=f"seed {seed}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation accuracy")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig_serious_cnn_training_curves.png", dpi=180)
    plt.close(fig)

    cm = pd.crosstab(all_preds["true_label"], all_preds["pred_label"], normalize="index")
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm.values, vmin=0, vmax=1, cmap="viridis")
    ax.set_xticks(range(len(cm.columns)), cm.columns, rotation=90)
    ax.set_yticks(range(len(cm.index)), cm.index)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig_serious_cnn_confusion.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs-dir", default="scalebreak_flyvis/outputs")
    parser.add_argument("--out-dir", default="scalebreak_flyvis/outputs/serious_cnn_baseline")
    parser.add_argument("--seeds", default="42,84,96,123,777")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--grid-size", type=int, default=32)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    if args.quick:
        args.seeds = "42"
        args.epochs = 5
        args.patience = 3
        args.grid_size = 24
        args.batch_size = min(args.batch_size, 128)

    out = Path(args.out_dir)
    setup_logging(out)
    (out / "predictions").mkdir(parents=True, exist_ok=True)
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    stim_path, meta_path, coord_path = find_inputs(Path(args.outputs_dir))
    meta = pd.read_csv(meta_path)
    coords = pd.read_csv(coord_path)
    stimuli = np.load(stim_path, mmap_mode="r")
    mapping = build_hex_grid(coords, args.grid_size)
    mapping_records = mapping.to_dict(orient="records")
    write_json({"method": "coordinate_minmax_projection", "grid_size": args.grid_size, "mapping": mapping_records}, out / "hex_to_grid_mapping.json")
    plot_grid_audit(mapping, out, args.grid_size)
    logging.info("Building TemporalResNet18Small tensor from %s", tuple(stimuli.shape))
    x = temporal_summary_grid(np.asarray(stimuli), mapping, args.grid_size)
    dynamic = meta["feature_family"].isin(["moving_edge", "moving_bar", "small_translating_target"]).to_numpy()
    y, inv = encode(meta["direction"].to_numpy())
    seeds = parse_seeds(args.seeds)
    scale_rows: list[dict] = []
    pred_rows: list[dict] = []
    curve_rows: list[dict] = []
    for seed in seeds:
        for heldout in sorted(meta.loc[dynamic, "scale"].unique()):
            train_idx = np.flatnonzero(dynamic & (meta["scale"].to_numpy() != heldout))
            test_idx = np.flatnonzero(dynamic & (meta["scale"].to_numpy() == heldout))
            pred, info, curves, model = train_fold(x, y, train_idx, test_idx, seed + int(heldout), args)
            m = metrics(y[test_idx], pred)
            scale_rows.append({"model": "TemporalResNet18Small", "seed": seed, "heldout_scale": heldout, **m, "n_train": len(train_idx), "n_test": len(test_idx), "test_accuracy": m["accuracy"], **info})
            import torch

            torch.save(model.state_dict(), out / "checkpoints" / f"TemporalResNet18Small_seed{seed}_scale{heldout}.pt")
            for c in curves:
                curve_rows.append({"model": "TemporalResNet18Small", "seed": seed, "heldout_scale": heldout, **c})
            rows = []
            for sample, yt, yp in zip(test_idx, y[test_idx], pred):
                row = {"model": "TemporalResNet18Small", "seed": seed, "heldout_scale": heldout, "sample": int(sample), "true_label": inv[int(yt)], "pred_label": inv[int(yp)], "correct": bool(yt == yp)}
                rows.append(row)
                pred_rows.append(row)
            pd.DataFrame(rows).to_csv(out / "predictions" / f"predictions_TemporalResNet18Small_seed{seed}_scale{heldout}.csv", index=False)

    table = pd.DataFrame(scale_rows)
    curves = pd.DataFrame(curve_rows)
    all_preds = pd.DataFrame(pred_rows)
    table.to_csv(out / "table_serious_cnn_by_seed_scale.csv", index=False)
    curves.to_csv(out / "training_curves.csv", index=False)
    by_seed = all_preds.groupby("seed")["correct"].mean()
    ci_low, ci_high = bootstrap_summary(all_preds, 42)
    summary = pd.DataFrame(
        [
            {
                "model": "TemporalResNet18Small",
                "mean_offdiag_accuracy": float(by_seed.mean()),
                "std": float(by_seed.std(ddof=1)) if len(by_seed) > 1 else 0.0,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "best_seed": int(by_seed.idxmax()),
                "worst_seed": int(by_seed.idxmin()),
                "flyvis_accuracy": FLYVIS_ACCURACY,
                "flyvis_minus_model": FLYVIS_ACCURACY - float(by_seed.mean()),
            }
        ]
    )
    summary.to_csv(out / "table_serious_cnn_summary.csv", index=False)
    plot_outputs(out, table, summary, curves, all_preds)
    acc = float(summary["mean_offdiag_accuracy"].iloc[0])
    if FLYVIS_ACCURACY - acc > 0.10:
        outcome = "Outcome A: The FlyVis dynamic scale-generalization result is robust to a manuscript-grade trained CNN control."
    elif FLYVIS_ACCURACY - acc > 0.05:
        outcome = "Outcome B: FlyVis is not uniquely superior in raw direction decoding; manuscript should emphasize biological decomposition, distributed cell-type representation, and activity efficiency."
    else:
        outcome = "Outcome C: The raw scale-generalization claim is not FlyVis-specific. Reframe around efficiency, biological interpretability, and representation geometry rather than performance."
    report = [
        "# Serious CNN Baseline Report",
        "",
        f"Mode: `{'quick' if args.quick else 'full'}`.",
        f"Stimulus tensor: `{tuple(stimuli.shape)}`.",
        f"Control tensor: `{tuple(x.shape)}`.",
        f"Seeds: `{seeds}`.",
        f"Mean off-diagonal held-out-scale direction accuracy: `{acc:.3f}`.",
        f"95% bootstrap CI over predictions: `[{ci_low:.3f}, {ci_high:.3f}]`.",
        f"FlyVis reference: `{FLYVIS_ACCURACY:.3f}`.",
        "",
        "## Interpretation",
        outcome,
        "",
        "No physical-distance claim, no generic object-recognition claim, and no exact connectome-necessity claim follows from this control.",
    ]
    (out / "REPORT.md").write_text("\n".join(report), encoding="utf-8")
    write_json(
        {
            "status": "completed",
            "quick": args.quick,
            "python": sys.version,
            "platform": platform.platform(),
            "stimuli_path": stim_path,
            "metadata_path": meta_path,
            "coords_path": coord_path,
            "stimulus_shape": tuple(stimuli.shape),
            "input_tensor_shape": tuple(x.shape),
            "elapsed_seconds": time.time() - t0,
        },
        out / "run_info.json",
    )
    print(f"Wrote serious CNN baseline to {out}")


if __name__ == "__main__":
    main()
