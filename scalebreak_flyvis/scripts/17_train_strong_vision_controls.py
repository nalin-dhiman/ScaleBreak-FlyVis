#!/usr/bin/env python
"""Train stronger vision-model controls for the ScaleBreak FlyVis benchmark.

The controls use the same leave-one-apparent-scale-out direction decoding task
as the FlyVis Pilot v4 analysis. They are deliberately not pretrained and do not
download weights. The hex retina is projected to a deterministic square grid
control representation; the mapping is saved with the outputs.
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

FLYVIS_REFERENCE = 0.9236111111111112


@dataclass
class RunConfig:
    seeds: list[int]
    epochs: int
    batch_size: int
    lr: float
    weight_decay: float
    patience: int
    quick: bool
    grid_size: int
    temporal_bins: int
    models: list[str]
    seed_validation_fraction: float = 0.15


def setup_logging(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(out_dir / "run.log", mode="a")],
    )


def write_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)


def write_markdown_table(df: pd.DataFrame, path: Path) -> None:
    try:
        text = df.to_markdown(index=False)
    except Exception:
        text = "```\n" + df.to_csv(index=False) + "```\n"
    path.write_text(text, encoding="utf-8")


def parse_seeds(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def find_input_files(outputs_dir: Path) -> tuple[Path, Path, Path]:
    candidates = [
        outputs_dir / "flyvis_pilot_v2" / "stimuli" / "stimuli.npy",
        outputs_dir / "flyvis_pilot_v2" / "responses" / "metadata.csv",
        outputs_dir / "flyvis_pilot_v2" / "stimuli" / "hex_coordinates.csv",
    ]
    if all(p.exists() for p in candidates):
        return candidates[0], candidates[1], candidates[2]
    stim = next(outputs_dir.rglob("stimuli.npy"))
    meta = next(outputs_dir.rglob("metadata.csv"))
    coords = next(outputs_dir.rglob("hex_coordinates.csv"))
    return stim, meta, coords


def build_grid_mapping(coords: pd.DataFrame, grid_size: int) -> pd.DataFrame:
    x = coords["x"].to_numpy(dtype=float)
    y = coords["y"].to_numpy(dtype=float)
    xi = np.round((x - x.min()) / max(x.max() - x.min(), 1e-8) * (grid_size - 1)).astype(int)
    yi = np.round((y - y.min()) / max(y.max() - y.min(), 1e-8) * (grid_size - 1)).astype(int)
    return pd.DataFrame({"hex_pixel": np.arange(len(coords)), "grid_x": xi, "grid_y": yi, "source_x": x, "source_y": y})


def project_to_grid(stimuli: np.ndarray, mapping: pd.DataFrame, grid_size: int, temporal_bins: int) -> np.ndarray:
    frame_bins = np.array_split(np.arange(stimuli.shape[1]), temporal_bins)
    out = np.full((stimuli.shape[0], temporal_bins, grid_size, grid_size), 0.5, dtype=np.float32)
    gx = mapping["grid_x"].to_numpy()
    gy = mapping["grid_y"].to_numpy()
    counts = np.zeros((grid_size, grid_size), dtype=np.float32)
    for y, x in zip(gy, gx):
        counts[y, x] += 1.0
    counts = np.maximum(counts, 1.0)
    for bi, frames in enumerate(frame_bins):
        vals = stimuli[:, frames, 0, :].mean(axis=1)
        acc = np.zeros((stimuli.shape[0], grid_size, grid_size), dtype=np.float32)
        for hp, (y, x) in enumerate(zip(gy, gx)):
            acc[:, y, x] += vals[:, hp]
        out[:, bi] = acc / counts[None]
    return out


def encode_labels(values: Iterable) -> tuple[np.ndarray, dict[int, str]]:
    labels = sorted({str(v) for v in values})
    to_i = {v: i for i, v in enumerate(labels)}
    inv = {i: v for v, i in to_i.items()}
    return np.array([to_i[str(v)] for v in values], dtype=np.int64), inv


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def bootstrap_ci(vals: np.ndarray, seed: int, n_boot: int = 1000) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    vals = np.asarray(vals, dtype=float)
    if len(vals) == 0:
        return float("nan"), float("nan")
    boots = np.array([vals[rng.integers(0, len(vals), len(vals))].mean() for _ in range(n_boot)])
    return float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))


def make_model(name: str, n_classes: int, width: int):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class TemporalCNN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.enc = nn.Sequential(
                nn.Conv2d(1, width, 3, padding=1),
                nn.BatchNorm2d(width),
                nn.ReLU(),
                nn.Conv2d(width, width * 2, 3, stride=2, padding=1),
                nn.BatchNorm2d(width * 2),
                nn.ReLU(),
            )
            self.temporal = nn.Conv1d(width * 2, width * 2, 3, padding=1)
            self.head = nn.Linear(width * 2, n_classes)

        def forward(self, x):
            b, t, h, w = x.shape
            z = self.enc(x.reshape(b * t, 1, h, w)).mean(dim=(2, 3)).reshape(b, t, -1)
            z = F.relu(self.temporal(z.transpose(1, 2))).mean(dim=2)
            return self.head(z)

    class CNN3D(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv3d(1, width, 3, padding=1),
                nn.BatchNorm3d(width),
                nn.ReLU(),
                nn.MaxPool3d((1, 2, 2)),
                nn.Conv3d(width, width * 2, 3, padding=1),
                nn.BatchNorm3d(width * 2),
                nn.ReLU(),
                nn.MaxPool3d((2, 2, 2)),
            )
            self.head = nn.Linear(width * 2, n_classes)

        def forward(self, x):
            z = self.net(x[:, None]).mean(dim=(2, 3, 4))
            return self.head(z)

    class ConvRNN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.inp = nn.Conv2d(1, width, 3, padding=1)
            self.rec = nn.Conv2d(width, width, 3, padding=1)
            self.head = nn.Linear(width, n_classes)

        def forward(self, x):
            h = torch.zeros((x.shape[0], width, x.shape[2], x.shape[3]), dtype=x.dtype, device=x.device)
            for ti in range(x.shape[1]):
                h = 0.65 * h + 0.35 * torch.relu(self.inp(x[:, ti : ti + 1]) + self.rec(h))
            return self.head(h.mean(dim=(2, 3)))

    class ResidualBlock(nn.Module):
        def __init__(self, channels: int) -> None:
            super().__init__()
            self.a = nn.Conv2d(channels, channels, 3, padding=1)
            self.b = nn.Conv2d(channels, channels, 3, padding=1)
            self.bn1 = nn.BatchNorm2d(channels)
            self.bn2 = nn.BatchNorm2d(channels)

        def forward(self, z):
            y = F.relu(self.bn1(self.a(z)))
            return F.relu(z + self.bn2(self.b(y)))

    class ResSummaryCNN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.stem = nn.Sequential(nn.Conv2d(6, width * 2, 3, padding=1), nn.BatchNorm2d(width * 2), nn.ReLU())
            self.blocks = nn.Sequential(ResidualBlock(width * 2), ResidualBlock(width * 2))
            self.down = nn.Conv2d(width * 2, width * 3, 3, stride=2, padding=1)
            self.head = nn.Linear(width * 3, n_classes)

        def forward(self, x):
            mean = x.mean(dim=1)
            maxv = x.max(dim=1).values
            first = x[:, 0]
            last = x[:, -1]
            diff = x[:, 1:] - x[:, :-1]
            motion = diff.abs().mean(dim=1)
            signed = diff.mean(dim=1)
            z = torch.stack([mean, maxv, first, last, motion, signed], dim=1)
            z = F.relu(self.down(self.blocks(self.stem(z)))).mean(dim=(2, 3))
            return self.head(z)

    return {
        "temporal_cnn": TemporalCNN,
        "cnn3d": CNN3D,
        "conv_rnn": ConvRNN,
        "resnet_like_temporal": ResSummaryCNN,
    }[name]()


def split_train_val(train_idx: np.ndarray, y: np.ndarray, seed: int, val_fraction: float) -> tuple[np.ndarray, np.ndarray]:
    from sklearn.model_selection import StratifiedShuffleSplit

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_fraction, random_state=seed)
    local_train, local_val = next(splitter.split(train_idx, y[train_idx]))
    return train_idx[local_train], train_idx[local_val]


def train_fold(
    x: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    model_name: str,
    seed: int,
    cfg: RunConfig,
    n_classes: int,
) -> tuple[np.ndarray, dict, list[dict]]:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(seed)
    width = 8 if cfg.quick else 16
    model = make_model(model_name, n_classes, width)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    counts = np.bincount(y[train_idx], minlength=n_classes).astype(np.float32)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = weights / weights.mean()
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32))
    tr_idx, val_idx = split_train_val(train_idx, y, seed, cfg.seed_validation_fraction)
    train_ds = TensorDataset(torch.tensor(x[tr_idx], dtype=torch.float32), torch.tensor(y[tr_idx], dtype=torch.long))
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_x = torch.tensor(x[val_idx], dtype=torch.float32)
    val_y = torch.tensor(y[val_idx], dtype=torch.long)
    best_state = None
    best_val = -np.inf
    best_epoch = 0
    stale = 0
    curves = []
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        losses = []
        for xb, yb in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            val_pred = model(val_x).argmax(dim=1).cpu().numpy()
        val_acc = float((val_pred == val_y.cpu().numpy()).mean())
        curves.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "val_accuracy": val_acc})
        if val_acc > best_val:
            best_val = val_acc
            best_epoch = epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= cfg.patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, len(test_idx), cfg.batch_size):
            idx = test_idx[start : start + cfg.batch_size]
            pred = model(torch.tensor(x[idx], dtype=torch.float32)).argmax(dim=1).cpu().numpy()
            preds.append(pred)
    info = {"epochs_ran": len(curves), "best_val_accuracy": best_val, "early_stopped": len(curves) < cfg.epochs, "best_epoch": best_epoch}
    return np.concatenate(preds), info, curves


def plot_outputs(out: Path, summary: pd.DataFrame, fold_table: pd.DataFrame, curves: pd.DataFrame) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = out
    fig_dir.mkdir(parents=True, exist_ok=True)
    s = summary.sort_values("mean_offdiag_accuracy")
    plt.figure(figsize=(8, 4.5))
    plt.barh(s["model"], s["mean_offdiag_accuracy"], xerr=[s["mean_offdiag_accuracy"] - s["ci_low"], s["ci_high"] - s["mean_offdiag_accuracy"]])
    plt.axvline(FLYVIS_REFERENCE, color="black", linestyle="--", label="FlyVis v4")
    plt.xlabel("Held-out apparent-scale direction accuracy")
    plt.legend()
    plt.tight_layout()
    for ext in ["png", "pdf"]:
        plt.savefig(fig_dir / f"fig_strong_controls_vs_flyvis.{ext}", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 4.5))
    for model, df in fold_table.groupby("model"):
        by = df.groupby("heldout_scale")["accuracy"].mean()
        plt.plot(by.index, by.values, marker="o", label=model)
    plt.axhline(FLYVIS_REFERENCE, color="black", linestyle="--", linewidth=1)
    plt.xlabel("Held-out apparent scale")
    plt.ylabel("Accuracy")
    plt.legend(fontsize=7)
    plt.tight_layout()
    for ext in ["png", "pdf"]:
        plt.savefig(fig_dir / f"fig_strong_controls_by_scale.{ext}", dpi=180)
    plt.close()

    if len(curves):
        plt.figure(figsize=(8, 4.5))
        for model, df in curves.groupby("model"):
            by = df.groupby("epoch")["val_accuracy"].mean()
            plt.plot(by.index, by.values, marker=".", label=model)
        plt.xlabel("Epoch")
        plt.ylabel("Validation accuracy")
        plt.legend(fontsize=7)
        plt.tight_layout()
        for ext in ["png", "pdf"]:
            plt.savefig(fig_dir / f"fig_training_curves.{ext}", dpi=180)
        plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs-dir", default="scalebreak_flyvis/outputs")
    parser.add_argument("--out-dir", default="scalebreak_flyvis/outputs/final_hardening/strong_controls")
    parser.add_argument("--seeds", default="42,84,96,123,777")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    out = Path(args.out_dir)
    setup_logging(out)
    seeds = [42] if args.quick else parse_seeds(args.seeds)
    cfg = RunConfig(
        seeds=seeds,
        epochs=3 if args.quick else args.epochs,
        batch_size=128 if args.quick else args.batch_size,
        lr=1e-3,
        weight_decay=1e-4,
        patience=2 if args.quick else 5,
        quick=args.quick,
        grid_size=16 if args.quick else 24,
        temporal_bins=11 if args.quick else 21,
        models=["temporal_cnn", "cnn3d", "conv_rnn", "resnet_like_temporal"],
    )
    write_json(asdict(cfg), out / "config_used.yaml")
    t0 = time.time()
    stim_path, meta_path, coord_path = find_input_files(Path(args.outputs_dir))
    meta = pd.read_csv(meta_path)
    stimuli = np.load(stim_path, mmap_mode="r")
    coords = pd.read_csv(coord_path)
    dynamic = meta["feature_family"].isin(["moving_edge", "moving_bar", "small_translating_target"]).to_numpy()
    y, inv = encode_labels(meta["direction"].to_numpy())
    mapping = build_grid_mapping(coords, cfg.grid_size)
    mapping.to_csv(out / "hex_to_grid_mapping.csv", index=False)
    logging.info("Projecting %s to grid tensor", tuple(stimuli.shape))
    x = project_to_grid(np.asarray(stimuli), mapping, cfg.grid_size, cfg.temporal_bins)
    logging.info("Control tensor shape: %s", tuple(x.shape))

    fold_rows: list[dict] = []
    pred_rows: dict[tuple[str, int], list[dict]] = {}
    curve_rows: list[dict] = []
    for model_name in cfg.models:
        for seed in cfg.seeds:
            logging.info("Training %s seed=%s", model_name, seed)
            pred_rows[(model_name, seed)] = []
            for heldout in sorted(meta.loc[dynamic, "scale"].unique()):
                train_idx = np.flatnonzero(dynamic & (meta["scale"].to_numpy() != heldout))
                test_idx = np.flatnonzero(dynamic & (meta["scale"].to_numpy() == heldout))
                pred, info, curves = train_fold(x, y, train_idx, test_idx, model_name, seed + int(heldout), cfg, len(inv))
                m = metrics(y[test_idx], pred)
                fold_rows.append(
                    {
                        "model": model_name,
                        "seed": seed,
                        "heldout_scale": heldout,
                        **m,
                        "n_train": len(train_idx),
                        "n_test": len(test_idx),
                        **info,
                    }
                )
                for sample, yt, yp in zip(test_idx, y[test_idx], pred):
                    pred_rows[(model_name, seed)].append(
                        {
                            "model": model_name,
                            "seed": seed,
                            "heldout_scale": heldout,
                            "sample": int(sample),
                            "true_label": inv[int(yt)],
                            "pred_label": inv[int(yp)],
                            "correct": bool(yt == yp),
                        }
                    )
                for c in curves:
                    curve_rows.append({"model": model_name, "seed": seed, "heldout_scale": heldout, **c})
    folds = pd.DataFrame(fold_rows)
    curves = pd.DataFrame(curve_rows)
    folds.to_csv(out / "table_strong_vision_controls.csv", index=False)
    curves.to_csv(out / "training_curves.csv", index=False)
    for (model_name, seed), rows in pred_rows.items():
        pred_df = pd.DataFrame(rows)
        safe = model_name.replace("/", "_")
        pred_df.to_csv(out / f"predictions_{safe}_seed{seed}.csv", index=False)
        cm = pd.crosstab(pred_df["true_label"], pred_df["pred_label"], dropna=False)
        cm.to_csv(out / f"confusion_{safe}_seed{seed}.csv")

    summary_rows = []
    for model, df in folds.groupby("model"):
        vals = df.groupby("seed")["accuracy"].mean().to_numpy()
        ci_low, ci_high = bootstrap_ci(vals, 42, n_boot=1000)
        by_seed = df.groupby("seed")["accuracy"].mean()
        summary_rows.append(
            {
                "model": model,
                "mean_offdiag_accuracy": float(vals.mean()),
                "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "best_seed": int(by_seed.idxmax()),
                "worst_seed": int(by_seed.idxmin()),
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values("mean_offdiag_accuracy", ascending=False)
    summary.to_csv(out / "table_strong_vision_controls_summary.csv", index=False)
    write_markdown_table(summary, out / "table_strong_vision_controls_summary.md")
    plot_outputs(out, summary, folds, curves)
    best = float(summary["mean_offdiag_accuracy"].max()) if len(summary) else float("nan")
    if best >= FLYVIS_REFERENCE - 0.05:
        interpretation = (
            "# Strong Vision Controls Interpretation\n\n"
            f"Best strong trained control reached `{best:.3f}`, within 0.05 of the FlyVis reference "
            f"`{FLYVIS_REFERENCE:.3f}`. The FlyVis advantage is therefore not unique in raw direction "
            "decoding; a manuscript should emphasize biological decomposition, temporal structure, and "
            "activity efficiency rather than raw performance alone.\n\nExact connectome necessity is not established.\n"
        )
    else:
        interpretation = (
            "# Strong Vision Controls Interpretation\n\n"
            f"Best strong trained control reached `{best:.3f}`, remaining more than 0.05 below the FlyVis "
            f"reference `{FLYVIS_REFERENCE:.3f}`. In this run, FlyVis scale-generalization is robust relative "
            "to trained vision controls.\n\nExact connectome necessity is not established.\n"
        )
    (out / "strong_controls_interpretation.md").write_text(interpretation, encoding="utf-8")
    write_json(
        {
            "status": "completed",
            "python": sys.version,
            "platform": platform.platform(),
            "stimuli_path": stim_path,
            "metadata_path": meta_path,
            "coords_path": coord_path,
            "stimulus_shape": tuple(stimuli.shape),
            "control_tensor_shape": tuple(x.shape),
            "elapsed_seconds": time.time() - t0,
            "flyvis_reference_accuracy": FLYVIS_REFERENCE,
            "best_control_accuracy": best,
        },
        out / "run_info.json",
    )
    print(f"Wrote strong controls to {out}")


if __name__ == "__main__":
    main()
