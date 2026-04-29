#!/usr/bin/env python
"""Temporal lesion analysis for FlyVis Pilot v2/v4 responses."""

from __future__ import annotations

import argparse
import json
import logging
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


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


def zscore(x: np.ndarray) -> np.ndarray:
    x = np.nan_to_num(x.astype(np.float32))
    return (x - x.mean(axis=0, keepdims=True)) / np.where(x.std(axis=0, keepdims=True) > 1e-8, x.std(axis=0, keepdims=True), 1.0)


def encode(values: np.ndarray) -> tuple[np.ndarray, dict[int, str]]:
    labels = sorted({str(v) for v in values})
    to_i = {v: i for i, v in enumerate(labels)}
    inv = {i: v for v, i in to_i.items()}
    return np.array([to_i[str(v)] for v in values], dtype=np.int64), inv


def classifier(seed: int):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed))


def loso_predictions(x: np.ndarray, meta: pd.DataFrame, target: str, subset: np.ndarray, model: str, seed: int) -> pd.DataFrame:
    y, inv = encode(meta[target].to_numpy())
    rows = []
    scales = sorted(meta.loc[subset, "scale"].unique())
    for heldout in scales:
        test = subset & (meta["scale"].to_numpy() == heldout)
        train = subset & ~test
        if train.sum() == 0 or test.sum() == 0 or len(np.unique(y[train])) < 2:
            continue
        clf = classifier(seed)
        clf.fit(x[train], y[train])
        pred = clf.predict(x[test])
        for sample, yt, yp in zip(np.flatnonzero(test), y[test], pred):
            rows.append(
                {
                    "feature_variant": model,
                    "target": target,
                    "heldout_scale": heldout,
                    "sample": int(sample),
                    "true_label": inv[int(yt)],
                    "pred_label": inv[int(yp)],
                    "correct": bool(yt == yp),
                }
            )
    return pd.DataFrame(rows)


def bootstrap_ci(vals: np.ndarray, seed: int, n_boot: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    vals = np.asarray(vals, dtype=float)
    boots = np.array([vals[rng.integers(0, len(vals), len(vals))].mean() for _ in range(n_boot)])
    return float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))


def feature_variants(resp: np.ndarray, pre_frames: int) -> dict[str, np.ndarray]:
    base = resp[:, :pre_frames].mean(axis=1, keepdims=True)
    delta = resp - base
    t = delta.shape[1]
    variants = {
        "early_0_20pct": delta[:, : max(1, int(0.20 * t))].mean(axis=1),
        "early_0_33pct": delta[:, : max(1, int(0.33 * t))].mean(axis=1),
        "middle_33_66pct": delta[:, int(0.33 * t) : int(0.66 * t)].mean(axis=1),
        "late_66_100pct": delta[:, int(0.66 * t) :].mean(axis=1),
        "full_time_mean": delta.mean(axis=1),
        "full_time_peak": delta.max(axis=1),
    }
    abs_mean = np.abs(delta).mean(axis=(0, 2))
    peak = int(abs_mean.argmax())
    lo = max(0, peak - int(0.06 * t))
    hi = min(t, peak + int(0.06 * t) + 1)
    variants["onset_peak_window"] = delta[:, lo:hi].mean(axis=1)
    for bins in [5, 11]:
        variants[f"temporal_bins_{bins}"] = np.concatenate([chunk.mean(axis=1) for chunk in np.array_split(delta, bins, axis=1)], axis=1)
    return {k: zscore(v) for k, v in variants.items()}


def time_resolved(resp: np.ndarray, pre_frames: int, frame_step: int) -> dict[str, np.ndarray]:
    base = resp[:, :pre_frames].mean(axis=1, keepdims=True)
    delta = resp - base
    out = {}
    for frame in range(0, delta.shape[1], frame_step):
        out[f"frame_{frame:03d}"] = zscore(delta[:, frame, :])
    return out


def summarize_predictions(pred: pd.DataFrame, seed: int, n_boot: int) -> dict:
    vals = pred["correct"].astype(float).to_numpy()
    ci_low, ci_high = bootstrap_ci(vals, seed, n_boot)
    return {"accuracy": float(vals.mean()), "ci_low": ci_low, "ci_high": ci_high, "n": len(vals)}


def plot_outputs(out: Path, lesion: pd.DataFrame, time_df: pd.DataFrame, family: pd.DataFrame) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(9, 4.5))
    order = lesion.sort_values("accuracy", ascending=True)
    xerr = np.vstack([order["accuracy"] - order["ci_low"], order["ci_high"] - order["accuracy"]])
    plt.barh(order["feature_variant"], order["accuracy"], xerr=xerr)
    plt.xlabel("Held-out apparent-scale direction accuracy")
    plt.tight_layout()
    for ext in ["png", "pdf"]:
        plt.savefig(out / f"fig_temporal_windows.{ext}", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 4.5))
    for fam, df in time_df.groupby("feature_family"):
        by = df.groupby("frame")["accuracy"].mean()
        plt.plot(by.index, by.values, label=fam)
    plt.xlabel("Frame")
    plt.ylabel("Held-out apparent-scale accuracy")
    plt.legend(fontsize=7)
    plt.tight_layout()
    for ext in ["png", "pdf"]:
        plt.savefig(out / f"fig_time_resolved_decoding.{ext}", dpi=180)
    plt.close()

    pivot = family.pivot(index="feature_variant", columns="feature_family", values="accuracy")
    plt.figure(figsize=(8, 5))
    plt.imshow(pivot.values, aspect="auto", vmin=0, vmax=1, cmap="viridis")
    plt.xticks(range(len(pivot.columns)), pivot.columns, rotation=45, ha="right", fontsize=8)
    plt.yticks(range(len(pivot.index)), pivot.index, fontsize=8)
    plt.colorbar(label="Accuracy")
    plt.tight_layout()
    for ext in ["png", "pdf"]:
        plt.savefig(out / f"fig_temporal_feature_family.{ext}", dpi=180)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v2-dir", default="scalebreak_flyvis/outputs/flyvis_pilot_v2")
    parser.add_argument("--out-dir", default="scalebreak_flyvis/outputs/final_hardening/temporal_lesions")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--frame-step", type=int, default=5)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    if args.quick:
        args.n_bootstrap = 200
        args.frame_step = 11

    out = Path(args.out_dir)
    setup_logging(out)
    t0 = time.time()
    v2 = Path(args.v2_dir)
    meta = pd.read_csv(v2 / "responses" / "metadata.csv")
    resp = np.load(v2 / "responses" / "flyvis_central_cell_responses.npy", mmap_mode="r")
    pre_frames = int(round(float(meta["t_pre"].iloc[0]) / float(meta["dt"].iloc[0])))
    dynamic = meta["feature_family"].isin(["moving_edge", "moving_bar", "small_translating_target"]).to_numpy()
    dynamic_plus_loom = meta["feature_family"].isin(["moving_edge", "moving_bar", "small_translating_target", "looming_disk"]).to_numpy()
    logging.info("Responses %s; pre_frames=%s", tuple(resp.shape), pre_frames)

    variants = feature_variants(np.asarray(resp), pre_frames)
    lesion_rows = []
    family_rows = []
    pred_dir = out / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    for name, x in variants.items():
        pred = loso_predictions(x, meta, "direction", dynamic, name, args.seed)
        pred.to_csv(pred_dir / f"predictions_{name}.csv", index=False)
        lesion_rows.append({"feature_variant": name, **summarize_predictions(pred, args.seed, args.n_bootstrap)})
        for family in ["moving_edge", "moving_bar", "small_translating_target"]:
            subset = (meta["feature_family"] == family).to_numpy()
            fp = loso_predictions(x, meta, "direction", subset, name, args.seed)
            family_rows.append({"feature_variant": name, "feature_family": family, **summarize_predictions(fp, args.seed, args.n_bootstrap)})
        loom = (meta["feature_family"] == "looming_disk").to_numpy()
        lp = loso_predictions(x, meta, "angle", loom, name, args.seed)
        family_rows.append({"feature_variant": name, "feature_family": "looming_angle", **summarize_predictions(lp, args.seed, args.n_bootstrap)})

    time_rows = []
    for name, x in time_resolved(np.asarray(resp), pre_frames, args.frame_step).items():
        frame = int(name.split("_")[1])
        for label, subset, target in [
            ("all_dynamic", dynamic, "direction"),
            ("moving_edge", (meta["feature_family"] == "moving_edge").to_numpy(), "direction"),
            ("moving_bar", (meta["feature_family"] == "moving_bar").to_numpy(), "direction"),
            ("small_translating_target", (meta["feature_family"] == "small_translating_target").to_numpy(), "direction"),
            ("looming_angle", (meta["feature_family"] == "looming_disk").to_numpy(), "angle"),
        ]:
            pred = loso_predictions(x, meta, target, subset, name, args.seed)
            time_rows.append({"frame": frame, "feature_family": label, **summarize_predictions(pred, args.seed, args.n_bootstrap)})

    lesion = pd.DataFrame(lesion_rows).sort_values("accuracy", ascending=False)
    time_df = pd.DataFrame(time_rows)
    family = pd.DataFrame(family_rows)
    lesion.to_csv(out / "table_temporal_lesion_accuracy.csv", index=False)
    time_df.to_csv(out / "table_time_resolved_accuracy.csv", index=False)
    family.to_csv(out / "table_temporal_feature_family.csv", index=False)
    plot_outputs(out, lesion, time_df, family)

    best = lesion.iloc[0]
    late = lesion.loc[lesion["feature_variant"] == "late_66_100pct", "accuracy"].iloc[0]
    bins5 = lesion.loc[lesion["feature_variant"] == "temporal_bins_5", "accuracy"].iloc[0]
    mean = lesion.loc[lesion["feature_variant"] == "full_time_mean", "accuracy"].iloc[0]
    early = lesion.loc[lesion["feature_variant"] == "early_0_20pct", "accuracy"].iloc[0]
    report = [
        "# Temporal Lesion Report",
        "",
        f"Responses shape: `{tuple(resp.shape)}`.",
        f"Best temporal feature variant: `{best['feature_variant']}` accuracy `{best['accuracy']:.3f}`.",
        "",
        "## Interpretation",
        f"1. Mostly early transient? {'No' if best['feature_variant'] not in {'early_0_20pct', 'onset_peak_window'} else 'Possibly'}; early 0-20% accuracy was `{early:.3f}`.",
        f"2. Late response retains direction? {'Yes' if late > 0.75 else 'Weak/partial'}; late 66-100% accuracy was `{late:.3f}`.",
        f"3. Temporal bins outperform time mean? {'Yes' if bins5 > mean else 'No'}; bins-5 `{bins5:.3f}` vs time-mean `{mean:.3f}`.",
        "4. Mechanism is dynamic rather than static: supported when direction decoding remains high for temporal windows and dynamic families; static shape identity remains secondary in v4.",
        "",
        "All wording refers to retinal apparent scale, not physical distance.",
    ]
    (out / "REPORT.md").write_text("\n".join(report), encoding="utf-8")
    write_json(
        {
            "status": "completed",
            "python": sys.version,
            "platform": platform.platform(),
            "response_shape": tuple(resp.shape),
            "pre_frames": pre_frames,
            "frame_step": args.frame_step,
            "elapsed_seconds": time.time() - t0,
        },
        out / "run_info.json",
    )
    print(f"Wrote temporal lesions to {out}")


if __name__ == "__main__":
    main()
