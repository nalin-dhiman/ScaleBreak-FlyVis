#!/usr/bin/env python
"""Analyze FlyVis connectome-causality pilot variants."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

FLYVIS_ACC = 0.9236111111111112


def write_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)


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


def loso(x: np.ndarray, meta: pd.DataFrame, target: str, subset: np.ndarray, variant: str, seed: int) -> pd.DataFrame:
    y, inv = encode(meta[target].to_numpy())
    rows = []
    for heldout in sorted(meta.loc[subset, "scale"].unique()):
        test = subset & (meta["scale"].to_numpy() == heldout)
        train = subset & ~test
        if train.sum() == 0 or test.sum() == 0 or len(np.unique(y[train])) < 2:
            continue
        clf = classifier(seed)
        clf.fit(x[train], y[train])
        pred = clf.predict(x[test])
        for sample, yt, yp in zip(np.flatnonzero(test), y[test], pred):
            rows.append({"variant": variant, "target": target, "heldout_scale": heldout, "sample": int(sample), "true_label": inv[int(yt)], "pred_label": inv[int(yp)], "correct": bool(yt == yp)})
    return pd.DataFrame(rows)


def bootstrap_acc(pred: pd.DataFrame, seed: int) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    vals = pred["correct"].astype(float).to_numpy()
    boots = np.array([vals[rng.integers(0, len(vals), len(vals))].mean() for _ in range(1000)])
    return float(vals.mean()), float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))


def representation_metrics(x: np.ndarray, meta: pd.DataFrame, subset: np.ndarray, pred: pd.DataFrame) -> dict[str, float]:
    acc = float(pred["correct"].mean())
    chance = 1.0 / meta.loc[subset, "direction"].nunique()
    bits = max(0.0, np.log2(meta.loc[subset, "direction"].nunique()) * (acc - chance) / max(1e-8, 1 - chance))
    labels = meta.loc[subset, ["direction", "scale"]].astype(str).agg("|".join, axis=1).to_numpy()
    uniq = sorted(set(labels))
    means = np.stack([x[subset][labels == u].mean(axis=0) for u in uniq])
    means = means / np.maximum(np.linalg.norm(means, axis=1, keepdims=True), 1e-8)
    sim = means @ means.T
    parts = [u.split("|") for u in uniq]
    within, between = [], []
    for i in range(len(parts)):
        for j in range(i + 1, len(parts)):
            if parts[i][0] == parts[j][0] and parts[i][1] != parts[j][1]:
                within.append(sim[i, j])
            elif parts[i][0] != parts[j][0]:
                between.append(sim[i, j])
    rsa = float(np.mean(within) - np.mean(between)) if within and between else np.nan
    activity = float(np.abs(x[subset]).mean())
    return {
        "direction_information_bits": bits,
        "rsa_same_direction_cross_scale_margin": rsa,
        "mean_activity_proxy_from_features": activity,
        "bits_per_activity": bits / max(activity, 1e-8),
    }


def plot_outputs(out: Path, main: pd.DataFrame, activity: pd.DataFrame) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    order = main.sort_values("offdiag_direction_accuracy")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(order["variant"], order["offdiag_direction_accuracy"])
    ax.axvline(FLYVIS_ACC, color="black", linestyle="--")
    ax.set_xlabel("Off-diagonal direction accuracy")
    fig.tight_layout()
    fig.savefig(fig_dir / "fig_causal_variants_accuracy.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(order["variant"], order["accuracy_drop_vs_flyvis"])
    ax.set_xlabel("Accuracy drop vs FlyVis")
    fig.tight_layout()
    fig.savefig(fig_dir / "fig_causal_variants_drop.png", dpi=180)
    plt.close(fig)

    merged = main.merge(activity[["variant", "mean_abs_activity_proxy"]], on="variant", how="left")
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(merged["mean_abs_activity_proxy"], merged["offdiag_direction_accuracy"])
    for _, r in merged.iterrows():
        ax.text(r["mean_abs_activity_proxy"], r["offdiag_direction_accuracy"], r["variant"], fontsize=6)
    ax.set_xlabel("Mean abs activity proxy")
    ax.set_ylabel("Accuracy")
    fig.tight_layout()
    fig.savefig(fig_dir / "fig_causal_activity_vs_accuracy.png", dpi=180)
    plt.close(fig)

    for key, filename in [("t4t5_attenuation", "fig_t4_t5_attenuation_curve.png"), ("edge_dropout_proxy", "fig_edge_dropout_curve.png")]:
        df = main[main["variant"].str.contains(key)]
        if len(df):
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.plot(df["level_numeric"], df["offdiag_direction_accuracy"], marker="o")
            ax.set_xlabel("Perturbation level")
            ax.set_ylabel("Accuracy")
            fig.tight_layout()
            fig.savefig(fig_dir / filename, dpi=180)
            plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="scalebreak_flyvis/outputs/connectome_causality")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out = Path(args.out_dir)
    meta = pd.read_csv(out / "metadata.csv")
    features = np.load(out / "causal_variant_features.npz")
    activity = pd.read_csv(out / "table_causal_activity_metrics.csv")
    activity_cols = [
        "variant",
        "perturbation",
        "level",
        "implementation",
        "mean_abs_activity_proxy",
        "peak_abs_activity_proxy",
        "temporal_variance_proxy",
    ]
    activity = activity[[c for c in activity_cols if c in activity.columns]]
    dynamic = meta["feature_family"].isin(["moving_edge", "moving_bar", "small_translating_target"]).to_numpy()
    pred_dir = out / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    main_rows, family_rows, rep_rows = [], [], []
    for name in features.files:
        x = features[name]
        pred = loso(x, meta, "direction", dynamic, name, args.seed)
        pred.to_csv(pred_dir / f"predictions_{name}.csv", index=False)
        acc, lo, hi = bootstrap_acc(pred, args.seed)
        level = activity.loc[activity["variant"] == name, "level"].iloc[0] if name in set(activity["variant"]) else "none"
        try:
            level_num = float(level)
        except Exception:
            level_num = np.nan
        main_rows.append({"variant": name, "offdiag_direction_accuracy": acc, "ci_low": lo, "ci_high": hi, "flyvis_reference_accuracy": FLYVIS_ACC, "accuracy_drop_vs_flyvis": FLYVIS_ACC - acc, "level_numeric": level_num, "n": len(pred)})
        rep_rows.append({"variant": name, **representation_metrics(x, meta, dynamic, pred)})
        for fam in ["moving_edge", "moving_bar", "small_translating_target"]:
            subset = (meta["feature_family"] == fam).to_numpy()
            fp = loso(x, meta, "direction", subset, name, args.seed)
            family_rows.append({"variant": name, "feature_family": fam, "accuracy": float(fp["correct"].mean()), "n": len(fp)})
    main_table = pd.DataFrame(main_rows).sort_values("offdiag_direction_accuracy", ascending=False)
    family = pd.DataFrame(family_rows)
    rep = pd.DataFrame(rep_rows)
    main_table.to_csv(out / "table_causal_variants.csv", index=False)
    family.to_csv(out / "table_causal_variant_feature_family.csv", index=False)
    activity.merge(rep, on="variant", how="left").to_csv(out / "table_causal_activity_metrics.csv", index=False)
    plot_outputs(out, main_table, activity)
    nonfull = main_table[main_table["variant"] != "full"]
    max_drop = float(nonfull["accuracy_drop_vs_flyvis"].max()) if len(nonfull) else 0.0
    if max_drop > 0.20:
        interpretation = "Perturbing FlyVis response structure substantially degrades scale-generalizing direction representations, but because these are response-space proxy perturbations this is provisional rather than proof of exact connectome causality."
    elif max_drop > 0.05:
        interpretation = "The perturbations have moderate effects, consistent with distributed or partially redundant representations; exact connectome necessity remains not established."
    else:
        interpretation = "The current experiments do not support a causal role for exact connectivity in this task. The main result should remain FlyVis model-specific representation, not connectome necessity."
    report = [
        "# Connectome-Causality Pilot Report",
        "",
        "Most variants in this run are non-retrained response-space proxy perturbations because automatic safe in-place FlyVis connectivity editing has not been certified.",
        "",
        f"Maximum drop vs FlyVis reference among variants: `{max_drop:.3f}`.",
        "",
        "## Interpretation",
        interpretation,
        "",
        "No physical-distance claim, no generic object-recognition claim, and no exact connectome-necessity claim is made from this pilot.",
    ]
    (out / "REPORT.md").write_text("\n".join(report), encoding="utf-8")
    write_json({"status": "completed", "variants": list(features.files), "max_drop_vs_flyvis": max_drop}, out / "analysis_run_info.json")
    print(f"Wrote connectome causality analysis to {out}")


if __name__ == "__main__":
    main()
