#!/usr/bin/env python
"""FlyVis Pilot v3: matched-control hardening.

Uses existing Pilot v2 stimuli/responses. No stimulus regeneration.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def write_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)


def zscore(x: np.ndarray) -> np.ndarray:
    x = np.nan_to_num(x.astype(np.float32))
    mu = x.mean(axis=0, keepdims=True)
    sd = x.std(axis=0, keepdims=True)
    return (x - mu) / np.where(sd > 1e-8, sd, 1.0)


def cap_features(x: np.ndarray, max_features: int) -> np.ndarray:
    x = np.nan_to_num(x.astype(np.float32))
    if x.shape[1] <= max_features:
        return x
    v = x.var(axis=0)
    keep = np.argsort(v)[-max_features:]
    keep.sort()
    return x[:, keep]


def temporal_bin_features(resp: np.ndarray, t_pre_frames: int, bins: int = 5) -> np.ndarray:
    base = resp[:, :t_pre_frames].mean(axis=1, keepdims=True)
    delta = resp - base
    chunks = np.array_split(delta, bins, axis=1)
    return np.concatenate([c.mean(axis=1) for c in chunks], axis=1)


def mean_features(resp: np.ndarray, t_pre_frames: int) -> np.ndarray:
    base = resp[:, :t_pre_frames].mean(axis=1, keepdims=True)
    return (resp - base).mean(axis=1)


def classifier(seed: int = 42):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed),
    )


def encode_labels(y: np.ndarray):
    from sklearn.preprocessing import LabelEncoder

    le = LabelEncoder()
    return le, le.fit_transform(pd.Series(y).astype(str).to_numpy())


def loso_predictions(
    x: np.ndarray,
    meta: pd.DataFrame,
    target: str,
    subset: np.ndarray,
    model: str,
    seed: int,
    scale_column: str = "scale",
) -> pd.DataFrame:
    y_raw = meta[target].to_numpy()
    le, y = encode_labels(y_raw)
    rows = []
    for scale in sorted(meta.loc[subset, scale_column].dropna().unique()):
        test = subset & (meta[scale_column].to_numpy() == scale)
        train = subset & ~test
        if train.sum() == 0 or test.sum() == 0 or len(np.unique(y[train])) < 2:
            continue
        clf = classifier(seed)
        clf.fit(x[train], y[train])
        pred = clf.predict(x[test])
        idx = np.flatnonzero(test)
        for sample, yt, yp in zip(idx, y[test], pred):
            rows.append(
                {
                    "model": model,
                    "target": target,
                    "sample": int(sample),
                    "heldout_scale": float(scale),
                    "true_label": le.inverse_transform([yt])[0],
                    "pred_label": le.inverse_transform([yp])[0],
                    "correct": bool(yt == yp),
                }
            )
    return pd.DataFrame(rows)


def scale_matrix_from_predictions(preds: pd.DataFrame, train_test_name: str = "heldout_scale") -> pd.DataFrame:
    return preds.groupby(["model", "target", train_test_name], as_index=False)["correct"].mean().rename(columns={"correct": "accuracy"})


def train_scale_matrix(x: np.ndarray, meta: pd.DataFrame, target: str, subset: np.ndarray, model: str, seed: int) -> pd.DataFrame:
    y_raw = meta[target].to_numpy()
    le, y = encode_labels(y_raw)
    rows = []
    scales = sorted(meta.loc[subset, "scale"].unique())
    for train_scale in scales:
        train = subset & (meta["scale"].to_numpy() == train_scale)
        if len(np.unique(y[train])) < 2:
            continue
        clf = classifier(seed)
        clf.fit(x[train], y[train])
        for test_scale in scales:
            test = subset & (meta["scale"].to_numpy() == test_scale)
            pred = clf.predict(x[test])
            rows.append(
                {
                    "model": model,
                    "target": target,
                    "train_scale": train_scale,
                    "test_scale": test_scale,
                    "accuracy": float((pred == y[test]).mean()),
                    "n_train": int(train.sum()),
                    "n_test": int(test.sum()),
                }
            )
    return pd.DataFrame(rows)


def confusion_matrix_table(preds: pd.DataFrame, out_path: Path) -> pd.DataFrame:
    labels = sorted(set(preds["true_label"].astype(str)) | set(preds["pred_label"].astype(str)))
    mat = pd.crosstab(preds["true_label"].astype(str), preds["pred_label"].astype(str)).reindex(index=labels, columns=labels, fill_value=0)
    mat.to_csv(out_path)
    return mat


def add_motion_type(meta: pd.DataFrame) -> pd.DataFrame:
    meta = meta.copy()
    mapping = {
        "moving_edge": "moving_edge",
        "moving_bar": "moving_bar",
        "looming_disk": "looming",
        "small_translating_target": "translating_target",
        "static_shape": "static_appendix",
    }
    meta["motion_type"] = meta["feature_family"].map(mapping).fillna(meta["feature_family"])
    meta["angle_position"] = meta["angle"]
    return meta


def make_square_movies(stimuli: np.ndarray, coords: pd.DataFrame, size: int = 32) -> np.ndarray:
    x = coords["x"].to_numpy()
    y = coords["y"].to_numpy()
    xi = np.round((x - x.min()) / (x.max() - x.min()) * (size - 1)).astype(int)
    yi = np.round((y - y.min()) / (y.max() - y.min()) * (size - 1)).astype(int)
    out = np.full((stimuli.shape[0], stimuli.shape[1], size, size), 0.5, dtype=np.float32)
    for hp in range(len(xi)):
        out[:, :, yi[hp], xi[hp]] = stimuli[:, :, 0, hp]
    return out


def baseline_features(stimuli: np.ndarray, coords: pd.DataFrame, meta: pd.DataFrame, max_features: int, seed: int) -> dict[str, np.ndarray]:
    from scalebreak.features import make_feature_matrices
    from scalebreak.models import local_rnn, pixel_baseline, small_cnn_random

    # Pixel baseline uses the hex movie as a 1 x hex_pixel "image" through the
    # existing summary feature extractor.
    pixel = pixel_baseline(stimuli[:, :, 0, :].reshape(stimuli.shape[0], stimuli.shape[1], 1, stimuli.shape[3]))
    square = make_square_movies(stimuli, coords)
    local = make_feature_matrices(local_rnn(square, seed=seed))["mean_time"]
    cnn = make_feature_matrices(small_cnn_random(square, seed=seed))["mean_time"]
    energy = np.abs(stimuli - 0.5).mean(axis=(1, 2, 3))
    peak_energy = np.abs(stimuli - 0.5).max(axis=(1, 2, 3))
    nuisance = meta[["mean_area_hex", "max_area_hex", "contrast"]].to_numpy(dtype=np.float32)
    nuisance = np.column_stack([nuisance, energy, peak_energy])
    return {
        "pixel": zscore(cap_features(pixel, max_features)),
        "local_rnn": zscore(cap_features(local, max_features)),
        "cnn": zscore(cap_features(cnn, max_features)),
        "nuisance_area_contrast_energy": zscore(nuisance),
    }


def bootstrap_ci(pred_by_model: dict[str, pd.DataFrame], comparisons: list[tuple[str, str]], n_boot: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    sample_ids = sorted(set.intersection(*[set(df["sample"]) for df in pred_by_model.values()]))
    idx = np.asarray(sample_ids)
    correct = {
        model: df.drop_duplicates("sample").set_index("sample").reindex(idx)["correct"].astype(float).to_numpy()
        for model, df in pred_by_model.items()
    }
    rows = []
    for model, vals in correct.items():
        boots = np.array([vals[rng.integers(0, len(vals), len(vals))].mean() for _ in range(n_boot)])
        rows.append({"metric": f"{model}_offdiag_direction_accuracy", "estimate": vals.mean(), "ci_low": np.quantile(boots, 0.025), "ci_high": np.quantile(boots, 0.975)})
    for a, b in comparisons:
        if a in correct and b in correct:
            diff = correct[a] - correct[b]
            boots = np.array([diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(n_boot)])
            rows.append({"metric": f"{a}_minus_{b}", "estimate": diff.mean(), "ci_low": np.quantile(boots, 0.025), "ci_high": np.quantile(boots, 0.975)})
    return pd.DataFrame(rows)


def cell_type_curves(type_summary: pd.DataFrame) -> pd.DataFrame:
    return type_summary.groupby(["cell_type", "feature_family", "scale"], as_index=False).agg(
        peak_delta_response=("peak_delta_response", "mean"),
        mean_delta_response=("mean_delta_response", "mean"),
        latency_to_peak=("latency_to_peak", "mean"),
    )


def make_figures(out: Path, matrices: dict[str, pd.DataFrame], ci: pd.DataFrame, importance: pd.DataFrame, t4t5: pd.DataFrame, family: pd.DataFrame, confusion: dict[str, pd.DataFrame]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fly = matrices["direction"]
    pivot = fly[(fly["model"] == "flyvis")].pivot(index="train_scale", columns="test_scale", values="accuracy")
    plt.figure(figsize=(5, 4))
    plt.imshow(pivot.values, vmin=0, vmax=1, cmap="viridis")
    plt.xticks(range(len(pivot.columns)), pivot.columns)
    plt.yticks(range(len(pivot.index)), pivot.index)
    plt.xlabel("Test apparent scale")
    plt.ylabel("Train apparent scale")
    plt.colorbar(label="Direction accuracy")
    plt.tight_layout()
    plt.savefig(fig_dir / "fig_v3_direction_scale_matrix_flyvis.png", dpi=180)
    plt.close()

    rows = ci[ci["metric"].str.endswith("_offdiag_direction_accuracy")].copy()
    rows["model"] = rows["metric"].str.replace("_offdiag_direction_accuracy", "", regex=False)
    rows = rows.sort_values("estimate")
    plt.figure(figsize=(7, 4))
    xerr = np.vstack([rows["estimate"] - rows["ci_low"], rows["ci_high"] - rows["estimate"]])
    plt.barh(rows["model"], rows["estimate"], xerr=xerr)
    plt.xlabel("Off-diagonal direction accuracy (95% CI)")
    plt.tight_layout()
    plt.savefig(fig_dir / "fig_v3_flyvis_vs_controls_ci.png", dpi=180)
    plt.close()

    top = importance.sort_values("drop_accuracy", ascending=False).head(25).iloc[::-1]
    plt.figure(figsize=(7, 6))
    plt.barh(top["cell_type"], top["drop_accuracy"])
    plt.xlabel("Drop in off-diagonal direction accuracy")
    plt.tight_layout()
    plt.savefig(fig_dir / "fig_v3_celltype_importance.png", dpi=180)
    plt.close()

    plt.figure(figsize=(6, 4))
    tt = t4t5.sort_values("offdiag_direction_accuracy")
    plt.barh(tt["group"], tt["offdiag_direction_accuracy"])
    plt.xlabel("Off-diagonal direction accuracy")
    plt.tight_layout()
    plt.savefig(fig_dir / "fig_v3_t4_t5_vs_other.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7, 4))
    ff = family.sort_values(["target_used", "offdiag_accuracy"])
    plt.barh(ff["feature_family"] + " / " + ff["target_used"], ff["offdiag_accuracy"])
    plt.xlabel("Off-diagonal scale-generalization accuracy")
    plt.tight_layout()
    plt.savefig(fig_dir / "fig_v3_feature_family_breakdown.png", dpi=180)
    plt.close()

    fig, axes = plt.subplots(2, 2, figsize=(9, 8))
    for ax, (target, mat) in zip(axes.ravel(), confusion.items()):
        arr = mat.to_numpy()
        ax.imshow(arr, cmap="magma")
        ax.set_title(target)
        ax.set_xticks(range(len(mat.columns)))
        ax.set_xticklabels(mat.columns, rotation=90, fontsize=6)
        ax.set_yticks(range(len(mat.index)))
        ax.set_yticklabels(mat.index, fontsize=6)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig_v3_confusion_matrices.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v2-dir", default="scalebreak_flyvis/outputs/flyvis_pilot_v2")
    parser.add_argument("--out-dir", default="scalebreak_flyvis/outputs/flyvis_pilot_v3")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--max-features", type=int, default=2048)
    args = parser.parse_args()

    root = Path.cwd()
    v2 = root / args.v2_dir
    out = root / args.out_dir
    tables = out / "tables"
    conf_dir = out / "confusion_matrices"
    tables.mkdir(parents=True, exist_ok=True)
    conf_dir.mkdir(parents=True, exist_ok=True)

    meta = add_motion_type(pd.read_csv(v2 / "responses" / "metadata.csv"))
    resp = np.load(v2 / "responses" / "flyvis_central_cell_responses.npy", mmap_mode="r")
    stimuli = np.load(v2 / "stimuli" / "stimuli.npy", mmap_mode="r")
    coords = pd.read_csv(v2 / "stimuli" / "hex_coordinates.csv")
    cell_meta = pd.read_csv(v2 / "responses" / "cell_metadata.csv")
    type_summary = pd.read_csv(v2 / "responses" / "type_response_summary.csv")
    type_summary = add_motion_type(type_summary)
    t_pre_frames = int(round(float(meta["t_pre"].iloc[0]) / float(meta["dt"].iloc[0])))
    rng = np.random.default_rng(args.seed)

    flyvis = zscore(temporal_bin_features(np.asarray(resp), t_pre_frames, bins=5))
    flyvis_mean = zscore(mean_features(np.asarray(resp), t_pre_frames))

    direction_subset = meta["feature_family"].isin(["moving_edge", "moving_bar", "small_translating_target"]).to_numpy()
    dynamic_subset = meta["dynamic"].to_numpy().astype(bool)
    all_subset = np.ones(len(meta), dtype=bool)

    # 1. Full confusion matrices for FlyVis LOSO where meaningful; scale uses a
    # stratified split because a held-out scale label cannot be learned.
    from sklearn.metrics import confusion_matrix
    from sklearn.model_selection import train_test_split

    confusion_tables = {}
    flyvis_preds = {}
    for target, subset in [
        ("direction", direction_subset),
        ("feature_family", dynamic_subset),
        ("motion_type", all_subset),
    ]:
        p = loso_predictions(flyvis, meta, target, subset, "flyvis", args.seed)
        flyvis_preds[target] = p
        confusion_tables[target] = confusion_matrix_table(p, conf_dir / f"confusion_{target}.csv")

    # Scale nuisance confusion.
    scale_idx = np.flatnonzero(dynamic_subset)
    le, y_scale = encode_labels(meta.loc[scale_idx, "scale"].to_numpy())
    tr, te = train_test_split(np.arange(len(scale_idx)), test_size=0.25, stratify=y_scale, random_state=args.seed)
    clf = classifier(args.seed)
    clf.fit(flyvis[scale_idx][tr], y_scale[tr])
    pred = clf.predict(flyvis[scale_idx][te])
    scale_pred = pd.DataFrame({"true_label": le.inverse_transform(y_scale[te]), "pred_label": le.inverse_transform(pred)})
    confusion_tables["scale"] = confusion_matrix_table(scale_pred, conf_dir / "confusion_scale.csv")

    # 2. Per-feature breakdown.
    family_rows = []
    for family_name in ["moving_edge", "moving_bar", "looming_disk", "small_translating_target"]:
        subset = (meta["feature_family"] == family_name).to_numpy()
        target = "direction" if family_name != "looming_disk" else "angle_position"
        if meta.loc[subset, target].nunique() < 2:
            continue
        p = loso_predictions(flyvis, meta, target, subset, "flyvis", args.seed)
        family_rows.append({"feature_family": family_name, "target_used": target, "offdiag_accuracy": p["correct"].mean(), "n": len(p)})
    static_subset = (meta["feature_family"] == "static_shape").to_numpy()
    p_static = loso_predictions(flyvis, meta, "shape", static_subset, "flyvis", args.seed)
    family_rows.append({"feature_family": "static_appendix_shapes", "target_used": "shape", "offdiag_accuracy": p_static["correct"].mean(), "n": len(p_static)})
    family_breakdown = pd.DataFrame(family_rows)
    family_breakdown.to_csv(tables / "per_feature_breakdown.csv", index=False)

    # 3. Scale generalization by cell type.
    cell_rows = []
    for i, row in cell_meta.iterrows():
        xi = flyvis[:, i::len(cell_meta)]  # one cell across all temporal bins
        pd_dir = loso_predictions(xi, meta, "direction", direction_subset, row.cell_type, args.seed)
        pd_fam = loso_predictions(xi, meta, "feature_family", dynamic_subset, row.cell_type, args.seed)
        curve = type_summary[type_summary["cell_type"] == row.cell_type].groupby("scale").agg(
            peak_delta_response=("peak_delta_response", "mean"),
            mean_delta_response=("mean_delta_response", "mean"),
            latency_to_peak=("latency_to_peak", "mean"),
        )
        cell_rows.append(
            {
                "cell_type": row.cell_type,
                "direction_offdiag_accuracy": pd_dir["correct"].mean(),
                "feature_family_offdiag_accuracy": pd_fam["correct"].mean(),
                "worst_direction_heldout_scale": pd_dir.groupby("heldout_scale")["correct"].mean().idxmin(),
                "worst_direction_accuracy": pd_dir.groupby("heldout_scale")["correct"].mean().min(),
                "mean_peak_response": curve["peak_delta_response"].mean(),
                "mean_delta_response": curve["mean_delta_response"].mean(),
                "mean_latency_to_peak": curve["latency_to_peak"].mean(),
            }
        )
    celltype_scale = pd.DataFrame(cell_rows)
    celltype_scale.to_csv(tables / "celltype_scale_generalization.csv", index=False)
    cell_type_curves(type_summary).to_csv(tables / "celltype_peak_mean_latency_curves.csv", index=False)

    # 4. Cell-type ablation.
    base_dir = flyvis_preds["direction"]
    base_acc = base_dir["correct"].mean()
    ablation_rows = []
    for i, row in cell_meta.iterrows():
        keep = [j for j in range(len(cell_meta)) if j != i]
        cols = []
        for b in range(5):
            cols.extend([b * len(cell_meta) + j for j in keep])
        p = loso_predictions(flyvis[:, cols], meta, "direction", direction_subset, "ablated", args.seed)
        acc = p["correct"].mean()
        ablation_rows.append({"cell_type": row.cell_type, "ablated_accuracy": acc, "baseline_accuracy": base_acc, "drop_accuracy": base_acc - acc})
    ablation = pd.DataFrame(ablation_rows).sort_values("drop_accuracy", ascending=False)
    ablation.to_csv(tables / "celltype_ablation_importance.csv", index=False)

    # 5. T4/T5-specific analysis.
    groups = {
        "T4a/b/c/d": ["T4a", "T4b", "T4c", "T4d"],
        "T5a/b/c/d": ["T5a", "T5b", "T5c", "T5d"],
        "T4+T5": ["T4a", "T4b", "T4c", "T4d", "T5a", "T5b", "T5c", "T5d"],
        "all_other": [c for c in cell_meta["cell_type"] if c not in {"T4a", "T4b", "T4c", "T4d", "T5a", "T5b", "T5c", "T5d"}],
    }
    group_rows = []
    for group, cells in groups.items():
        idx = [int(cell_meta.index[cell_meta["cell_type"] == c][0]) for c in cells if c in set(cell_meta["cell_type"])]
        cols = []
        for b in range(5):
            cols.extend([b * len(cell_meta) + j for j in idx])
        p = loso_predictions(flyvis[:, cols], meta, "direction", direction_subset, group, args.seed)
        group_rows.append({"group": group, "offdiag_direction_accuracy": p["correct"].mean(), "n_cells": len(idx)})
    t4t5 = pd.DataFrame(group_rows)
    t4t5.to_csv(tables / "t4_t5_vs_other.csv", index=False)

    # 6. Matched controls and baselines.
    model_features = {"flyvis": flyvis}
    model_features.update(baseline_features(np.asarray(stimuli), coords, meta, args.max_features, args.seed))
    shuffled_resp = np.asarray(resp).copy()
    rng.shuffle(shuffled_resp, axis=0)
    model_features["response_shuffled_flyvis"] = zscore(temporal_bin_features(shuffled_resp, t_pre_frames))
    time_shuf = np.asarray(resp).copy()
    for i in range(time_shuf.shape[0]):
        time_shuf[i] = time_shuf[i, rng.permutation(time_shuf.shape[1])]
    model_features["time_shuffled_flyvis"] = zscore(temporal_bin_features(time_shuf, t_pre_frames))
    cell_shuf = np.asarray(resp).copy()
    for i in range(cell_shuf.shape[0]):
        cell_shuf[i] = cell_shuf[i][:, rng.permutation(cell_shuf.shape[2])]
    model_features["cell_type_label_shuffled_flyvis"] = zscore(temporal_bin_features(cell_shuf, t_pre_frames))

    pred_by_model = {}
    for model, x in model_features.items():
        x = zscore(cap_features(x, args.max_features))
        pred_by_model[model] = loso_predictions(x, meta, "direction", direction_subset, model, args.seed)
    # Label permutation controls.
    meta_scale_perm = meta.copy()
    meta_scale_perm["scale_perm"] = rng.permutation(meta_scale_perm["scale"].to_numpy())
    pred_by_model["scale_label_permutation"] = loso_predictions(flyvis, meta_scale_perm, "direction", direction_subset, "scale_label_permutation", args.seed, scale_column="scale_perm")
    meta_dir_perm = meta.copy()
    perm_idx = np.flatnonzero(direction_subset)
    meta_dir_perm.loc[perm_idx, "direction"] = rng.permutation(meta_dir_perm.loc[perm_idx, "direction"].to_numpy())
    pred_by_model["direction_label_permutation"] = loso_predictions(flyvis, meta_dir_perm, "direction", direction_subset, "direction_label_permutation", args.seed)

    controls = pd.DataFrame(
        [{"model": m, "offdiag_direction_accuracy": p["correct"].mean(), "n": len(p)} for m, p in pred_by_model.items()]
    ).sort_values("offdiag_direction_accuracy", ascending=False)
    controls.to_csv(tables / "matched_controls_direction_accuracy.csv", index=False)

    # Full scale matrices for main FlyVis direction.
    direction_matrix = train_scale_matrix(flyvis, meta, "direction", direction_subset, "flyvis", args.seed)
    direction_matrix.to_csv(tables / "direction_scale_generalization_flyvis.csv", index=False)
    pd.concat(pred_by_model.values(), ignore_index=True).to_csv(tables / "direction_loso_predictions_all_models.csv", index=False)

    # 7. Bootstrap confidence.
    comparisons = [
        ("flyvis", "pixel"),
        ("flyvis", "local_rnn"),
        ("flyvis", "cnn"),
        ("flyvis", "response_shuffled_flyvis"),
        ("flyvis", "time_shuffled_flyvis"),
        ("flyvis", "cell_type_label_shuffled_flyvis"),
        ("flyvis", "scale_label_permutation"),
        ("flyvis", "direction_label_permutation"),
        ("flyvis", "nuisance_area_contrast_energy"),
    ]
    ci = bootstrap_ci(pred_by_model, comparisons, args.n_bootstrap, args.seed)
    ci.to_csv(tables / "bootstrap_ci_1000.csv", index=False)

    # Figures.
    make_figures(out, {"direction": direction_matrix}, ci, ablation, t4t5, family_breakdown, confusion_tables)

    main_acc = float(controls.loc[controls["model"] == "flyvis", "offdiag_direction_accuracy"].iloc[0])
    best_family = family_breakdown.sort_values("offdiag_accuracy", ascending=False).iloc[0]
    top_cells = ablation.head(10)["cell_type"].tolist()
    t4t5_acc = float(t4t5.loc[t4t5["group"] == "T4+T5", "offdiag_direction_accuracy"].iloc[0])
    other_acc = float(t4t5.loc[t4t5["group"] == "all_other", "offdiag_direction_accuracy"].iloc[0])
    perm_max = controls[controls["model"].str.contains("permutation|shuffled", regex=True)]["offdiag_direction_accuracy"].max()
    manuscript_ready = bool(main_acc > perm_max + 0.1 and main_acc > float(controls.loc[controls["model"] == "pixel", "offdiag_direction_accuracy"].iloc[0]) + 0.05)
    write_json(
        {
            "main_offdiag_direction_accuracy": main_acc,
            "best_generalizing_feature_family": best_family.to_dict(),
            "top_ablation_cell_types": top_cells,
            "t4_t5_offdiag_direction_accuracy": t4t5_acc,
            "other_cell_types_offdiag_direction_accuracy": other_acc,
            "max_shuffled_or_permutation_control": float(perm_max),
            "strong_enough_for_manuscript_figure": manuscript_ready,
            "n_bootstrap": args.n_bootstrap,
        },
        out / "analysis_manifest.json",
    )

    ci_text = ci.to_csv(index=False)
    family_text = family_breakdown.to_csv(index=False)
    controls_text = controls.to_csv(index=False)
    ablation_text = ablation.head(15).to_csv(index=False)
    report = [
        "# FlyVis Pilot v3 Report",
        "",
        "## Main Numerical Result",
        f"- FlyVis off-diagonal direction accuracy across held-out apparent scales: `{main_acc:.3f}`.",
        f"- Highest shuffled/permutation control: `{perm_max:.3f}`.",
        f"- Strong enough for a manuscript figure by the current heuristic: `{manuscript_ready}`.",
        "",
        "## Confidence Intervals",
        ci_text,
        "",
        "## Which Feature Families Generalize",
        family_text,
        "",
        "## Which Cell Types Matter",
        f"- Top ablation-sensitive cell types: `{', '.join(top_cells)}`.",
        ablation_text,
        "",
        "## Do T4/T5 Explain The Effect?",
        f"- T4+T5 off-diagonal direction accuracy: `{t4t5_acc:.3f}`.",
        f"- All-other cell types off-diagonal direction accuracy: `{other_acc:.3f}`.",
        "- Interpretation: T4/T5 are tested explicitly; dominance requires T4+T5 to match or exceed all-other and ablation to rank T4/T5 highly.",
        "",
        "## Do Nuisance/Permutation Controls Destroy The Effect?",
        controls_text,
        "",
        "## Guardrails",
        "- The analysis concerns retinal projection and apparent scale, not physical distance.",
        "- Direction and dynamic-feature scale-generalization are primary; static appendix shapes are secondary controls.",
    ]
    (out / "REPORT.md").write_text("\n".join(report), encoding="utf-8")
    print(f"Wrote FlyVis Pilot v3 matched-control hardening to {out}")


if __name__ == "__main__":
    main()
