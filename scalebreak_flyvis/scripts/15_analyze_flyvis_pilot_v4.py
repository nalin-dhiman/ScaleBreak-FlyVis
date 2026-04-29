#!/usr/bin/env python
"""FlyVis Pilot v4: connectome-causality and representation hardening.

Uses existing Pilot v2/v3 outputs. It does not regenerate stimuli.
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


def response_feature_matrix(resp: np.ndarray, t_pre_frames: int, bins: int = 5) -> np.ndarray:
    base = resp[:, :t_pre_frames].mean(axis=1, keepdims=True)
    delta = resp - base
    chunks = np.array_split(delta, bins, axis=1)
    return zscore(np.concatenate([c.mean(axis=1) for c in chunks], axis=1))


def mean_response_matrix(resp: np.ndarray, t_pre_frames: int) -> np.ndarray:
    base = resp[:, :t_pre_frames].mean(axis=1, keepdims=True)
    return zscore((resp - base).mean(axis=1))


def classifier(seed: int):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed))


def encode(y: np.ndarray):
    from sklearn.preprocessing import LabelEncoder

    le = LabelEncoder()
    yy = le.fit_transform(pd.Series(y).astype(str).to_numpy())
    return le, yy


def loso_predictions(
    x: np.ndarray,
    meta: pd.DataFrame,
    target: str,
    subset: np.ndarray,
    model: str,
    seed: int,
    scale_col: str = "scale",
) -> pd.DataFrame:
    le, y = encode(meta[target].to_numpy())
    rows = []
    for heldout in sorted(meta.loc[subset, scale_col].dropna().unique()):
        test = subset & (meta[scale_col].to_numpy() == heldout)
        train = subset & ~test
        if train.sum() == 0 or test.sum() == 0 or len(np.unique(y[train])) < 2:
            continue
        clf = classifier(seed)
        clf.fit(x[train], y[train])
        pred = clf.predict(x[test])
        for sample, yt, yp in zip(np.flatnonzero(test), y[test], pred):
            rows.append(
                {
                    "model": model,
                    "target": target,
                    "sample": int(sample),
                    "heldout_scale": float(heldout),
                    "true_label": le.inverse_transform([yt])[0],
                    "pred_label": le.inverse_transform([yp])[0],
                    "correct": bool(yt == yp),
                }
            )
    return pd.DataFrame(rows)


def offdiag_accuracy(x: np.ndarray, meta: pd.DataFrame, target: str, subset: np.ndarray, model: str, seed: int) -> float:
    p = loso_predictions(x, meta, target, subset, model, seed)
    return float(p["correct"].mean()) if len(p) else float("nan")


def train_scale_matrix(x: np.ndarray, meta: pd.DataFrame, target: str, subset: np.ndarray, model: str, seed: int) -> pd.DataFrame:
    le, y = encode(meta[target].to_numpy())
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
                }
            )
    return pd.DataFrame(rows)


def add_motion_type(meta: pd.DataFrame) -> pd.DataFrame:
    meta = meta.copy()
    meta["motion_type"] = meta["feature_family"].map(
        {
            "moving_edge": "moving_edge",
            "moving_bar": "moving_bar",
            "looming_disk": "looming",
            "small_translating_target": "translating_target",
            "static_shape": "static_appendix",
        }
    )
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


def graph_variants(type_edges_path: Path, out_dir: Path, seed: int) -> dict[str, Path]:
    rng = np.random.default_rng(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    type_edges = pd.read_parquet(type_edges_path)
    variants: dict[str, pd.DataFrame] = {"real optic-lobe graph": type_edges.copy()}
    deg = type_edges.copy()
    deg["pre_type"] = rng.permutation(deg["pre_type"].to_numpy())
    deg["post_type"] = rng.permutation(deg["post_type"].to_numpy())
    variants["degree-matched graph"] = deg
    w = type_edges.copy()
    w["total_weight"] = rng.permutation(w["total_weight"].to_numpy())
    variants["weight-shuffled graph"] = w
    labels = np.array(sorted(set(type_edges["pre_type"]) | set(type_edges["post_type"])))
    shuffled = labels.copy()
    rng.shuffle(shuffled)
    mapping = dict(zip(labels, shuffled))
    lab = type_edges.copy()
    lab["pre_type"] = lab["pre_type"].map(mapping)
    lab["post_type"] = lab["post_type"].map(mapping)
    variants["type-shuffled graph"] = lab
    paths = {}
    for name, df in variants.items():
        path = out_dir / (name.replace(" ", "_").replace("-", "_") + ".parquet")
        df.to_parquet(path, index=False)
        paths[name] = path
    return paths


def build_model_features(
    stimuli: np.ndarray,
    coords: pd.DataFrame,
    meta: pd.DataFrame,
    graph_paths: dict[str, Path],
    max_features: int,
    seed: int,
) -> dict[str, np.ndarray]:
    from scalebreak.features import make_feature_matrices
    from scalebreak.models import local_rnn, optic_lobe_type_rate, pixel_baseline, small_cnn_random

    features: dict[str, np.ndarray] = {}
    hex_movie = stimuli[:, :, 0, :].reshape(stimuli.shape[0], stimuli.shape[1], 1, stimuli.shape[3])
    features["pixel"] = cap_features(pixel_baseline(hex_movie), max_features)
    square = make_square_movies(stimuli, coords)
    # Fixed feature extractors with trained linear readouts in the exact same
    # held-out-scale protocol as FlyVis.
    features["CNN"] = make_feature_matrices(small_cnn_random(square, seed=seed))["mean_time"]
    features["local RNN"] = make_feature_matrices(local_rnn(square, seed=seed))["mean_time"]
    for name, path in graph_paths.items():
        act, _ = optic_lobe_type_rate(square, path, seed=seed)
        features[name] = make_feature_matrices(act)["mean_time"]
    energy = np.abs(stimuli - 0.5).mean(axis=(1, 2, 3))
    peak_energy = np.abs(stimuli - 0.5).max(axis=(1, 2, 3))
    features["pixel area/contrast/energy nuisance"] = np.column_stack(
        [meta["mean_area_hex"], meta["max_area_hex"], meta["contrast"], energy, peak_energy]
    )
    return {k: zscore(cap_features(v, max_features)) for k, v in features.items()}


def bootstrap_difference(base: pd.DataFrame, other: pd.DataFrame, n_boot: int, seed: int) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    b = base.drop_duplicates("sample").set_index("sample")
    o = other.drop_duplicates("sample").set_index("sample")
    ids = np.array(sorted(set(b.index) & set(o.index)))
    diff = b.loc[ids, "correct"].astype(float).to_numpy() - o.loc[ids, "correct"].astype(float).to_numpy()
    boots = np.array([diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(n_boot)])
    return float(diff.mean()), float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))


def bootstrap_accuracy(pred: pd.DataFrame, n_boot: int, seed: int) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    vals = pred["correct"].astype(float).to_numpy()
    boots = np.array([vals[rng.integers(0, len(vals), len(vals))].mean() for _ in range(n_boot)])
    return float(vals.mean()), float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))


def class_means(x: np.ndarray, meta: pd.DataFrame, subset: np.ndarray, label: str) -> tuple[np.ndarray, pd.DataFrame]:
    keys = meta.loc[subset, [label, "scale"]].astype(str).agg("|".join, axis=1)
    labels = sorted(keys.unique())
    xs = x[subset]
    means = np.stack([xs[keys.to_numpy() == k].mean(axis=0) for k in labels])
    parts = [k.split("|") for k in labels]
    return means, pd.DataFrame({label: [p[0] for p in parts], "scale": [float(p[1]) for p in parts]})


def cosine_matrix(x: np.ndarray) -> np.ndarray:
    x = zscore(x)
    denom = np.linalg.norm(x, axis=1, keepdims=True)
    x = x / np.where(denom > 1e-8, denom, 1.0)
    return x @ x.T


def linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    x = x - x.mean(axis=0, keepdims=True)
    y = y - y.mean(axis=0, keepdims=True)
    hsic = np.linalg.norm(x @ y.T, "fro") ** 2
    denom = np.linalg.norm(x @ x.T, "fro") * np.linalg.norm(y @ y.T, "fro")
    return float(hsic / denom) if denom > 1e-12 else float("nan")


def representation_metrics(x: np.ndarray, meta: pd.DataFrame, subset: np.ndarray, pred: pd.DataFrame, label: str = "direction") -> pd.DataFrame:
    means, lab = class_means(x, meta, subset, label)
    sim = cosine_matrix(means)
    rows = []
    within = []
    between = []
    for i in range(len(lab)):
        for j in range(i + 1, len(lab)):
            same_label = lab.iloc[i][label] == lab.iloc[j][label]
            same_scale = lab.iloc[i]["scale"] == lab.iloc[j]["scale"]
            if same_label and not same_scale:
                within.append(sim[i, j])
            elif not same_label:
                between.append(sim[i, j])
    rsa_margin = float(np.mean(within) - np.mean(between))
    scales = sorted(meta.loc[subset, "scale"].unique())
    ckas = []
    for a_i, a in enumerate(scales):
        for b in scales[a_i + 1 :]:
            xa = x[subset & (meta["scale"].to_numpy() == a)]
            xb = x[subset & (meta["scale"].to_numpy() == b)]
            n = min(len(xa), len(xb))
            ckas.append(linear_cka(xa[:n], xb[:n]))
    acc = pred["correct"].mean()
    n_classes = meta.loc[subset, label].nunique()
    chance = 1.0 / n_classes
    # Conservative normalized information lower bound proxy.
    bits = max(0.0, np.log2(n_classes) * (acc - chance) / max(1e-8, 1 - chance))
    base_resp = x[subset]
    activity_cost = float(np.mean(np.abs(base_resp)))
    return pd.DataFrame(
        [
            {
                "label": label,
                "rsa_same_direction_cross_scale_margin": rsa_margin,
                "mean_offdiag_cka": float(np.nanmean(ckas)),
                "direction_information_lower_bound_bits": bits,
                "mean_abs_activity_proxy": activity_cost,
                "activity_cost_per_retained_direction_bit": activity_cost / max(bits, 1e-8),
                "retained_direction_bits_per_activity": bits / max(activity_cost, 1e-8),
            }
        ]
    )


def cell_groups(cell_types: list[str]) -> dict[str, list[int]]:
    t4 = {"T4a", "T4b", "T4c", "T4d"}
    t5 = {"T5a", "T5b", "T5c", "T5d"}
    early = {f"R{i}" for i in range(1, 9)} | {f"L{i}" for i in range(1, 6)} | {"Lawf1", "Lawf2"}
    medulla_prefixes = ("Mi", "Tm", "TmY", "C", "T1", "T2", "T2a", "T3")
    groups = {
        "T4 only": [i for i, c in enumerate(cell_types) if c in t4],
        "T5 only": [i for i, c in enumerate(cell_types) if c in t5],
        "T4+T5": [i for i, c in enumerate(cell_types) if c in t4 | t5],
        "non-T4/T5": [i for i, c in enumerate(cell_types) if c not in t4 | t5],
        "early photoreceptor/lamina": [i for i, c in enumerate(cell_types) if c in early],
        "medulla-like": [i for i, c in enumerate(cell_types) if c.startswith(medulla_prefixes)],
        "lobula/lobula-plate tagged": [i for i, c in enumerate(cell_types) if "Lo" in c or c in t4 | t5],
    }
    return groups


def remove_cells_from_temporal_bins(x: np.ndarray, remove: list[int], n_cells: int = 65, n_bins: int = 5) -> np.ndarray:
    remove = set(remove)
    keep_cols = []
    for b in range(n_bins):
        keep_cols.extend([b * n_cells + i for i in range(n_cells) if i not in remove])
    return x[:, keep_cols]


def plot_outputs(out: Path, main: pd.DataFrame, feature: pd.DataFrame, ablation: pd.DataFrame, rep: pd.DataFrame) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    mm = main.sort_values("offdiag_accuracy")
    plt.figure(figsize=(8, 5))
    plt.barh(mm["model"], mm["offdiag_accuracy"])
    plt.xlabel("Off-diagonal direction accuracy")
    plt.tight_layout()
    plt.savefig(fig_dir / "fig_v4_main_controls.png", dpi=180)
    plt.close()

    plt.figure(figsize=(10, 5))
    plot_df = feature.melt(id_vars=["feature_family_task"], var_name="model", value_name="accuracy")
    piv = plot_df.pivot(index="feature_family_task", columns="model", values="accuracy")
    plt.imshow(piv.values, vmin=0, vmax=1, aspect="auto", cmap="viridis")
    plt.xticks(range(len(piv.columns)), piv.columns, rotation=90, fontsize=7)
    plt.yticks(range(len(piv.index)), piv.index, fontsize=7)
    plt.colorbar(label="Off-diagonal accuracy")
    plt.tight_layout()
    plt.savefig(fig_dir / "fig_v4_feature_family_controls.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 5))
    aa = ablation.sort_values("drop_accuracy")
    xerr = np.vstack([aa["drop_accuracy"] - aa["drop_ci_low"], aa["drop_ci_high"] - aa["drop_accuracy"]])
    plt.barh(aa["ablation"], aa["drop_accuracy"], xerr=xerr)
    plt.xlabel("Accuracy drop vs full FlyVis")
    plt.tight_layout()
    plt.savefig(fig_dir / "fig_v4_group_ablation.png", dpi=180)
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].bar(rep["model"], rep["rsa_same_direction_cross_scale_margin"])
    axes[0].set_ylabel("RSA margin")
    axes[0].tick_params(axis="x", rotation=90)
    axes[1].bar(rep["model"], rep["mean_offdiag_cka"])
    axes[1].set_ylabel("Mean off-diagonal CKA")
    axes[1].tick_params(axis="x", rotation=90)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig_v4_rsa_cka.png", dpi=180)
    plt.close(fig)

    plt.figure(figsize=(7, 4))
    plt.bar(rep["model"], rep["retained_direction_bits_per_activity"])
    plt.ylabel("Retained direction bits per activity proxy")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(fig_dir / "fig_v4_bits_per_activity.png", dpi=180)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v2-dir", default="scalebreak_flyvis/outputs/flyvis_pilot_v2")
    parser.add_argument("--v3-dir", default="scalebreak_flyvis/outputs/flyvis_pilot_v3")
    parser.add_argument("--out-dir", default="scalebreak_flyvis/outputs/flyvis_pilot_v4")
    parser.add_argument("--type-edges", default="scalebreak_flyvis/outputs/connectome/type_edges.parquet")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--max-features", type=int, default=2048)
    args = parser.parse_args()

    root = Path.cwd()
    v2 = root / args.v2_dir
    out = root / args.out_dir
    tables = out / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    meta = add_motion_type(pd.read_csv(v2 / "responses" / "metadata.csv"))
    resp = np.load(v2 / "responses" / "flyvis_central_cell_responses.npy", mmap_mode="r")
    stimuli = np.load(v2 / "stimuli" / "stimuli.npy", mmap_mode="r")
    coords = pd.read_csv(v2 / "stimuli" / "hex_coordinates.csv")
    cell_meta = pd.read_csv(v2 / "responses" / "cell_metadata.csv")
    t_pre_frames = int(round(float(meta["t_pre"].iloc[0]) / float(meta["dt"].iloc[0])))
    flyvis = response_feature_matrix(np.asarray(resp), t_pre_frames)

    direction_subset = meta["feature_family"].isin(["moving_edge", "moving_bar", "small_translating_target"]).to_numpy()
    graph_paths = graph_variants(root / args.type_edges, out / "graph_controls", args.seed)
    features = {"FlyVis": flyvis}
    features.update(build_model_features(np.asarray(stimuli), coords, meta, graph_paths, args.max_features, args.seed))

    # Destructive representation controls.
    shuffled_resp = np.asarray(resp).copy()
    rng.shuffle(shuffled_resp, axis=0)
    features["response-shuffled FlyVis"] = response_feature_matrix(shuffled_resp, t_pre_frames)
    time_resp = np.asarray(resp).copy()
    for i in range(time_resp.shape[0]):
        time_resp[i] = time_resp[i, rng.permutation(time_resp.shape[1])]
    features["time-shuffled FlyVis"] = response_feature_matrix(time_resp, t_pre_frames)
    cell_resp = np.asarray(resp).copy()
    for i in range(cell_resp.shape[0]):
        cell_resp[i] = cell_resp[i][:, rng.permutation(cell_resp.shape[2])]
    features["cell-identity-shuffled FlyVis"] = response_feature_matrix(cell_resp, t_pre_frames)
    noise = rng.normal(np.mean(flyvis), np.std(flyvis), size=flyvis.shape).astype(np.float32)
    features["Gaussian response noise"] = zscore(noise)

    # Main controls.
    pred_main = {}
    main_rows = []
    for model, x in features.items():
        if model in {"response-shuffled FlyVis", "time-shuffled FlyVis", "cell-identity-shuffled FlyVis", "Gaussian response noise"}:
            continue
        p = loso_predictions(x, meta, "direction", direction_subset, model, args.seed)
        pred_main[model] = p
        main_rows.append({"model": model, "offdiag_accuracy": p["correct"].mean(), "n": len(p)})
    main = pd.DataFrame(main_rows).sort_values("offdiag_accuracy", ascending=False)
    main.to_csv(tables / "table_v4_main_results.csv", index=False)

    # Feature-family separated main table.
    requested_cols = [
        "FlyVis",
        "pixel",
        "CNN",
        "local RNN",
        "real optic-lobe graph",
        "degree-matched graph",
        "weight-shuffled graph",
        "type-shuffled graph",
    ]
    tasks = [
        ("moving edge direction", "moving_edge", "direction"),
        ("moving bar direction", "moving_bar", "direction"),
        ("small translating target direction", "small_translating_target", "direction"),
        ("looming angle-position", "looming_disk", "angle_position"),
        ("static appendix shape", "static_shape", "shape"),
    ]
    feature_rows = []
    for label, family, target in tasks:
        row = {"feature_family_task": label}
        subset = (meta["feature_family"] == family).to_numpy()
        for model in requested_cols:
            row[model] = offdiag_accuracy(features[model], meta, target, subset, model, args.seed)
        feature_rows.append(row)
    feature_table = pd.DataFrame(feature_rows)
    feature_table.to_csv(tables / "table_v4_feature_family_controls.csv", index=False)

    # Destructive controls table.
    destructive = []
    for model in ["response-shuffled FlyVis", "time-shuffled FlyVis", "cell-identity-shuffled FlyVis", "Gaussian response noise"]:
        p = loso_predictions(features[model], meta, "direction", direction_subset, model, args.seed)
        pred_main[model] = p
        destructive.append({"model": model, "offdiag_accuracy": p["correct"].mean(), "n": len(p)})
    meta_dir_perm = meta.copy()
    idx = np.flatnonzero(direction_subset)
    meta_dir_perm.loc[idx, "direction"] = rng.permutation(meta_dir_perm.loc[idx, "direction"].to_numpy())
    p_dir_perm = loso_predictions(flyvis, meta_dir_perm, "direction", direction_subset, "direction-label permutation", args.seed)
    pred_main["direction-label permutation"] = p_dir_perm
    destructive.append({"model": "direction-label permutation", "offdiag_accuracy": p_dir_perm["correct"].mean(), "n": len(p_dir_perm)})
    # Random train/test cell dropout mismatch.
    dropout_preds = []
    n_cells = len(cell_meta)
    for heldout in sorted(meta.loc[direction_subset, "scale"].unique()):
        train_subset = direction_subset & (meta["scale"].to_numpy() != heldout)
        test_subset = direction_subset & (meta["scale"].to_numpy() == heldout)
        train_drop = set(rng.choice(n_cells, size=n_cells // 3, replace=False))
        test_drop = set(rng.choice(n_cells, size=n_cells // 3, replace=False))
        train_cols = []
        test_cols = []
        for b in range(5):
            train_cols.extend([b * n_cells + i for i in range(n_cells) if i not in train_drop])
            test_cols.extend([b * n_cells + i for i in range(n_cells) if i not in test_drop])
        # Align dimensions by truncating to shared count. This intentionally
        # creates a mismatched feature identity control.
        n = min(len(train_cols), len(test_cols))
        x_mix = np.zeros((len(meta), n), dtype=np.float32)
        x_mix[train_subset] = flyvis[train_subset][:, train_cols[:n]]
        x_mix[test_subset] = flyvis[test_subset][:, test_cols[:n]]
        p = loso_predictions(x_mix, meta, "direction", train_subset | test_subset, "random cell dropout mismatch", args.seed)
        dropout_preds.append(p[p["heldout_scale"] == heldout])
    p_dropout = pd.concat(dropout_preds, ignore_index=True)
    pred_main["random cell dropout mismatch"] = p_dropout
    destructive.append({"model": "random cell dropout mismatch", "offdiag_accuracy": p_dropout["correct"].mean(), "n": len(p_dropout)})
    pd.DataFrame(destructive).to_csv(tables / "destructive_representation_controls.csv", index=False)

    # Scale-label permutation is only a sanity check.
    meta_scale = meta.copy()
    meta_scale["scale_perm"] = rng.permutation(meta_scale["scale"].to_numpy())
    sanity = loso_predictions(flyvis, meta_scale, "direction", direction_subset, "scale-label permutation sanity check", args.seed, scale_col="scale_perm")
    pd.DataFrame([{"model": "scale-label permutation sanity check", "offdiag_accuracy": sanity["correct"].mean(), "n": len(sanity)}]).to_csv(
        tables / "scale_label_permutation_sanity_check.csv", index=False
    )

    # Cell group mechanism and proper group ablation.
    cell_types = cell_meta["cell_type"].tolist()
    groups = cell_groups(cell_types)
    group_rows = []
    ablation_rows = []
    full_pred = pred_main["FlyVis"]
    full_acc, _, _ = bootstrap_accuracy(full_pred, args.n_bootstrap, args.seed)
    top10 = pd.read_csv(root / args.v3_dir / "tables" / "celltype_ablation_importance.csv").head(10)["cell_type"].tolist()
    top5 = top10[:5]
    ablation_groups = {
        "remove all T4": groups["T4 only"],
        "remove all T5": groups["T5 only"],
        "remove T4+T5": groups["T4+T5"],
        "remove top-5 ablation-sensitive types": [cell_types.index(c) for c in top5 if c in cell_types],
        "remove top-10 ablation-sensitive types": [cell_types.index(c) for c in top10 if c in cell_types],
    }
    ablation_groups["remove random matched top-5 count"] = list(rng.choice(n_cells, size=len(ablation_groups["remove top-5 ablation-sensitive types"]), replace=False))
    ablation_groups["remove random matched top-10 count"] = list(rng.choice(n_cells, size=len(ablation_groups["remove top-10 ablation-sensitive types"]), replace=False))
    for group, idxs in groups.items():
        xg = flyvis[:, [b * n_cells + i for b in range(5) for i in idxs]]
        p = loso_predictions(xg, meta, "direction", direction_subset, group, args.seed)
        group_rows.append({"group": group, "n_cell_types": len(idxs), "offdiag_direction_accuracy": p["correct"].mean()})
    for name, idxs in ablation_groups.items():
        xa = remove_cells_from_temporal_bins(flyvis, idxs, n_cells=n_cells, n_bins=5)
        p = loso_predictions(xa, meta, "direction", direction_subset, name, args.seed)
        acc, _, _ = bootstrap_accuracy(p, args.n_bootstrap, args.seed)
        drop, lo, hi = bootstrap_difference(full_pred, p, args.n_bootstrap, args.seed)
        ablation_rows.append(
            {
                "ablation": name,
                "n_removed_cell_types": len(idxs),
                "accuracy": acc,
                "full_accuracy": full_acc,
                "drop_accuracy": drop,
                "drop_ci_low": lo,
                "drop_ci_high": hi,
            }
        )
    pd.DataFrame(group_rows).to_csv(tables / "cell_group_mechanism.csv", index=False)
    ablation_table = pd.DataFrame(ablation_rows)
    ablation_table.to_csv(tables / "table_v4_group_ablation.csv", index=False)

    # Representation metrics.
    rep_rows = []
    for model in requested_cols:
        p = pred_main[model]
        r = representation_metrics(features[model], meta, direction_subset, p, label="direction")
        r["model"] = model
        rep_rows.append(r)
    rep_table = pd.concat(rep_rows, ignore_index=True)
    rep_table.to_csv(tables / "table_v4_representation_metrics.csv", index=False)

    # Bootstrap CIs for main and destructive comparisons.
    ci_rows = []
    for model, p in pred_main.items():
        acc, lo, hi = bootstrap_accuracy(p, args.n_bootstrap, args.seed)
        ci_rows.append({"metric": f"{model} offdiag direction accuracy", "estimate": acc, "ci_low": lo, "ci_high": hi})
        if model != "FlyVis":
            d, dlo, dhi = bootstrap_difference(full_pred, p, args.n_bootstrap, args.seed)
            ci_rows.append({"metric": f"FlyVis minus {model}", "estimate": d, "ci_low": dlo, "ci_high": dhi})
    ci_table = pd.DataFrame(ci_rows)
    ci_table.to_csv(tables / "bootstrap_ci_v4.csv", index=False)

    plot_outputs(out, main, feature_table, ablation_table, rep_table)

    robust_dynamic = bool(
        ci_table.loc[ci_table["metric"] == "FlyVis offdiag direction accuracy", "ci_low"].iloc[0] > 0.75
        and feature_table.loc[feature_table["feature_family_task"].str.contains("moving edge|moving bar|small translating"), "FlyVis"].min() > 0.85
    )
    detailed_needed = bool(
        main.loc[main["model"] == "FlyVis", "offdiag_accuracy"].iloc[0]
        > max(
            main.loc[main["model"].isin(["pixel", "CNN", "local RNN", "real optic-lobe graph", "degree-matched graph", "weight-shuffled graph", "type-shuffled graph"]), "offdiag_accuracy"]
        )
        + 0.05
    )
    write_json(
        {
            "robust_dynamic_scale_generalization_in_flyvis": robust_dynamic,
            "evidence_detailed_connectome_model_structure_necessary": detailed_needed,
            "flyvis_offdiag_direction_accuracy": float(main.loc[main["model"] == "FlyVis", "offdiag_accuracy"].iloc[0]),
            "best_nonflyvis_control_accuracy": float(main.loc[main["model"] != "FlyVis", "offdiag_accuracy"].max()),
            "scale_label_permutation_reclassified_as_sanity_check": True,
        },
        out / "analysis_manifest.json",
    )

    report = [
        "# FlyVis Pilot v4 Report",
        "",
        "## A. Robust Dynamic Scale-Generalization In FlyVis?",
        f"Call: `{robust_dynamic}`.",
        f"FlyVis off-diagonal direction accuracy: `{main.loc[main['model'] == 'FlyVis', 'offdiag_accuracy'].iloc[0]:.3f}`.",
        "",
        "## B. Is Detailed Connectome/Model Structure Necessary?",
        f"Call: `{detailed_needed}`.",
        "This compares FlyVis against pixel, CNN, local RNN, real optic-lobe type-rate, and graph-shuffled type-rate controls with trained linear readouts.",
        "",
        "## Main Results",
        main.to_csv(index=False),
        "",
        "## Feature-Family Controls",
        feature_table.to_csv(index=False),
        "",
        "## Group Ablation",
        ablation_table.to_csv(index=False),
        "",
        "## Representation Metrics",
        rep_table.to_csv(index=False),
        "",
        "## Confidence Intervals",
        ci_table.to_csv(index=False),
        "",
        "## Notes",
        "- Scale-label permutation is reclassified as a sanity check, not a destructive null control.",
        "- All claims are about retinal projection and apparent scale, not physical distance.",
        "- Static appendix shape is secondary; dynamic feature representation is primary.",
    ]
    (out / "REPORT.md").write_text("\n".join(report), encoding="utf-8")
    print(f"Wrote FlyVis Pilot v4 to {out}")


if __name__ == "__main__":
    main()
