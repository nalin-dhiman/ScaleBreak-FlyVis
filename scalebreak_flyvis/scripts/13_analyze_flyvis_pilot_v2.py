#!/usr/bin/env python
"""Analyze FlyVis Pilot v2 dynamic apparent-scale representations."""

from __future__ import annotations

import argparse
import json
import os
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


def cap_features(x: np.ndarray, max_features: int = 2048) -> np.ndarray:
    x = np.nan_to_num(x.astype(np.float32))
    if x.shape[1] <= max_features:
        return x
    v = x.var(axis=0)
    keep = np.argsort(v)[-max_features:]
    keep.sort()
    return x[:, keep]


def response_features(resp: np.ndarray, t_pre_frames: int, bins: int = 5) -> dict[str, np.ndarray]:
    base = resp[:, :t_pre_frames].mean(axis=1, keepdims=True)
    delta = resp - base
    chunks = np.array_split(delta, bins, axis=1)
    return {
        "mean": delta.mean(axis=1),
        "peak": delta.max(axis=1),
        "final": delta[:, -1],
        "temporal_bins": np.concatenate([c.mean(axis=1) for c in chunks], axis=1),
    }


def zscore(x: np.ndarray) -> np.ndarray:
    mu = x.mean(axis=0, keepdims=True)
    sd = x.std(axis=0, keepdims=True)
    return (x - mu) / np.where(sd > 1e-8, sd, 1.0)


def classifier(seed: int = 42):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed))


def train_test_accuracy(x: np.ndarray, y: np.ndarray, train: np.ndarray, test: np.ndarray, seed: int = 42) -> float:
    from sklearn.metrics import accuracy_score
    from sklearn.preprocessing import LabelEncoder

    le = LabelEncoder()
    yy = le.fit_transform(pd.Series(y).astype(str))
    if len(np.unique(yy[train])) < 2 or len(np.unique(yy[test])) < 2:
        return float("nan")
    clf = classifier(seed)
    clf.fit(x[train], yy[train])
    return float(accuracy_score(yy[test], clf.predict(x[test])))


def scale_generalization(
    x: np.ndarray,
    meta: pd.DataFrame,
    target: str,
    subset: np.ndarray | None = None,
    model_name: str = "flyvis",
    feature_name: str = "mean",
    seed: int = 42,
) -> pd.DataFrame:
    if subset is None:
        subset = np.ones(len(meta), dtype=bool)
    scales = sorted(meta.loc[subset, "scale"].unique())
    rows = []
    y = meta[target].to_numpy()
    for train_scale in scales:
        train = subset & (meta["scale"].to_numpy() == train_scale)
        for test_scale in scales:
            test = subset & (meta["scale"].to_numpy() == test_scale)
            rows.append(
                {
                    "model": model_name,
                    "feature": feature_name,
                    "target": target,
                    "train_scale": train_scale,
                    "test_scale": test_scale,
                    "accuracy": train_test_accuracy(x, y, train, test, seed),
                    "n_train": int(train.sum()),
                    "n_test": int(test.sum()),
                }
            )
    return pd.DataFrame(rows)


def leave_one_scale(x: np.ndarray, meta: pd.DataFrame, target: str, subset: np.ndarray, model_name: str, seed: int) -> pd.DataFrame:
    rows = []
    y = meta[target].to_numpy()
    for scale in sorted(meta.loc[subset, "scale"].unique()):
        test = subset & (meta["scale"].to_numpy() == scale)
        train = subset & ~test
        rows.append(
            {
                "model": model_name,
                "target": target,
                "heldout_scale": scale,
                "accuracy": train_test_accuracy(x, y, train, test, seed),
                "n_train": int(train.sum()),
                "n_test": int(test.sum()),
            }
        )
    return pd.DataFrame(rows)


def within_scale_curve(x: np.ndarray, meta: pd.DataFrame, target: str, subset: np.ndarray, model_name: str, seed: int) -> pd.DataFrame:
    rows = []
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import accuracy_score
    from sklearn.preprocessing import LabelEncoder

    for scale in sorted(meta.loc[subset, "scale"].unique()):
        idx = np.flatnonzero(subset & (meta["scale"].to_numpy() == scale))
        y = meta.iloc[idx][target].astype(str).to_numpy()
        le = LabelEncoder()
        yy = le.fit_transform(y)
        min_count = np.bincount(yy).min() if len(np.unique(yy)) > 1 else 0
        if min_count < 2:
            rows.append({"model": model_name, "target": target, "scale": scale, "accuracy": np.nan})
            continue
        cv = StratifiedKFold(n_splits=min(5, min_count), shuffle=True, random_state=seed)
        scores = []
        for tr, te in cv.split(x[idx], yy):
            clf = classifier(seed)
            clf.fit(x[idx][tr], yy[tr])
            scores.append(accuracy_score(yy[te], clf.predict(x[idx][te])))
        rows.append({"model": model_name, "target": target, "scale": scale, "accuracy": float(np.mean(scores))})
    return pd.DataFrame(rows)


def estimate_breakpoints(curves: pd.DataFrame, target: str, n_classes: int, margin: float = 0.05) -> pd.DataFrame:
    rows = []
    chance = 1.0 / max(n_classes, 1)
    for model, sub_model in curves[curves["target"] == target].groupby("model"):
        for family, sub in sub_model.groupby("feature_family") if "feature_family" in sub_model else [("all", sub_model)]:
            ordered = sub.sort_values("scale")
            ok = ordered["accuracy"].to_numpy() >= chance + margin
            bp = float(ordered["scale"].to_numpy()[np.argmax(ok)]) if ok.any() else np.nan
            rows.append({"model": model, "target": target, "feature_family": family, "chance": chance, "breakpoint_scale": bp})
    return pd.DataFrame(rows)


def make_square_movies(stimuli: np.ndarray, coords: pd.DataFrame, size: int = 32) -> np.ndarray:
    x = coords["x"].to_numpy()
    y = coords["y"].to_numpy()
    xi = np.round((x - x.min()) / (x.max() - x.min()) * (size - 1)).astype(int)
    yi = np.round((y - y.min()) / (y.max() - y.min()) * (size - 1)).astype(int)
    out = np.full((stimuli.shape[0], stimuli.shape[1], size, size), 0.5, dtype=np.float32)
    for hp in range(len(xi)):
        out[:, :, yi[hp], xi[hp]] = stimuli[:, :, 0, hp]
    return out


def run_baseline_features(stimuli: np.ndarray, coords: pd.DataFrame, args: argparse.Namespace) -> dict[str, np.ndarray]:
    from scalebreak.models import local_rnn, optic_lobe_type_rate, pixel_baseline, small_cnn_random
    from scalebreak.features import make_feature_matrices

    features = {"pixel": cap_features(pixel_baseline(stimuli[:, :, 0, :].reshape(stimuli.shape[0], stimuli.shape[1], 1, stimuli.shape[3])))}
    square = make_square_movies(stimuli, coords, size=32)
    features["local_rnn"] = make_feature_matrices(local_rnn(square, seed=args.seed))["mean_time"]
    features["cnn"] = make_feature_matrices(small_cnn_random(square, seed=args.seed))["mean_time"]
    try:
        type_edges = Path(args.type_edges)
        type_path = type_edges if type_edges.exists() else None
        act, _ = optic_lobe_type_rate(square, type_path, seed=args.seed)
        features["optic_lobe_type_rate"] = make_feature_matrices(act)["mean_time"]
    except Exception as exc:
        print(f"Skipping optic_lobe_type_rate baseline: {exc}")
    return {k: zscore(cap_features(v, args.max_features)) for k, v in features.items()}


def plot_figures(
    out_dir: Path,
    stimuli: np.ndarray,
    coords: pd.DataFrame,
    meta: pd.DataFrame,
    type_summary: pd.DataFrame,
    direction_matrix: pd.DataFrame,
    breakpoint_df: pd.DataFrame,
    control_summary: pd.DataFrame,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    # Stimulus grid.
    examples = meta.drop_duplicates(["feature_family", "scale", "contrast"]).groupby("feature_family").head(4).index[:24]
    n = len(examples)
    cols = 6
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.1, rows * 1.8), squeeze=False)
    x = coords["x"].to_numpy()
    y = coords["y"].to_numpy()
    for ax, idx in zip(axes.ravel(), examples):
        frame = stimuli[idx, stimuli.shape[1] // 2, 0]
        ax.scatter(x, y, c=frame, s=9, cmap="gray", vmin=0.5, vmax=1.0)
        r = meta.loc[idx]
        ax.set_title(f"{r.feature_family}\ns={r.scale:g}", fontsize=7)
        ax.axis("off")
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(fig_dir / "fig01_stimulus_grid.png", dpi=180)
    plt.close(fig)

    # T4/T5 response curves by scale.
    ttypes = ["T4a", "T4b", "T4c", "T4d", "T5a", "T5b", "T5c", "T5d"]
    sub = type_summary[(type_summary["cell_type"].isin(ttypes)) & (type_summary["dynamic"])]
    curve = sub.groupby(["cell_type", "scale"], as_index=False)["peak_delta_response"].mean()
    plt.figure(figsize=(8, 4.5))
    for cell_type, ss in curve.groupby("cell_type"):
        plt.plot(ss["scale"], ss["peak_delta_response"], marker="o", label=cell_type)
    plt.xlabel("Apparent scale")
    plt.ylabel("Peak activity proxy")
    plt.legend(ncol=4, fontsize=7)
    plt.tight_layout()
    plt.savefig(fig_dir / "fig02_t4_t5_response_curves_by_scale.png", dpi=180)
    plt.close()

    # Direction generalization heatmap for FlyVis.
    fly = direction_matrix[direction_matrix["model"] == "flyvis"]
    if not fly.empty:
        pivot = fly.pivot_table(index="train_scale", columns="test_scale", values="accuracy", aggfunc="mean")
        plt.figure(figsize=(5, 4))
        plt.imshow(pivot.values, vmin=0, vmax=1, cmap="viridis")
        plt.xticks(range(len(pivot.columns)), pivot.columns)
        plt.yticks(range(len(pivot.index)), pivot.index)
        plt.xlabel("Test scale")
        plt.ylabel("Train scale")
        plt.colorbar(label="Direction accuracy")
        plt.tight_layout()
        plt.savefig(fig_dir / "fig03_direction_decoding_scale_generalization.png", dpi=180)
        plt.close()

    # Breakpoints.
    if not breakpoint_df.empty:
        plt.figure(figsize=(7, 4))
        bp = breakpoint_df[breakpoint_df["target"] == "direction"].dropna(subset=["breakpoint_scale"])
        labels = bp["model"] + "/" + bp["feature_family"]
        plt.barh(labels, bp["breakpoint_scale"])
        plt.xlabel("Breakpoint apparent scale")
        plt.tight_layout()
        plt.savefig(fig_dir / "fig04_feature_family_breakpoint_curves.png", dpi=180)
        plt.close()

    # FlyVis vs controls.
    if not control_summary.empty:
        plt.figure(figsize=(7, 4))
        cs = control_summary.sort_values("offdiag_direction_accuracy", ascending=True)
        plt.barh(cs["model"], cs["offdiag_direction_accuracy"])
        plt.xlabel("Off-diagonal direction accuracy")
        plt.tight_layout()
        plt.savefig(fig_dir / "fig05_flyvis_vs_controls_summary.png", dpi=180)
        plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stim-dir", default="scalebreak_flyvis/outputs/flyvis_pilot_v2/stimuli")
    parser.add_argument("--response-dir", default="scalebreak_flyvis/outputs/flyvis_pilot_v2/responses")
    parser.add_argument("--out-dir", default="scalebreak_flyvis/outputs/flyvis_pilot_v2")
    parser.add_argument("--type-edges", default="scalebreak_flyvis/outputs/connectome/type_edges.parquet")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-features", type=int, default=2048)
    parser.add_argument("--skip-controls", action="store_true")
    args = parser.parse_args()

    root = Path.cwd()
    stim_dir = (root / args.stim_dir).resolve()
    response_dir = (root / args.response_dir).resolve()
    out_dir = (root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    tables = out_dir / "tables"
    tables.mkdir(parents=True, exist_ok=True)

    stimuli = np.load(stim_dir / "stimuli.npy", mmap_mode="r")
    meta = pd.read_csv(response_dir / "metadata.csv")
    responses = np.load(response_dir / "flyvis_central_cell_responses.npy", mmap_mode="r")
    cell_meta = pd.read_csv(response_dir / "cell_metadata.csv")
    coords = pd.read_csv(stim_dir / "hex_coordinates.csv")
    type_summary = pd.read_csv(response_dir / "type_response_summary.csv")
    t_pre_frames = int(round(float(meta["t_pre"].iloc[0]) / float(meta["dt"].iloc[0])))

    fly_features = zscore(response_features(np.asarray(responses), t_pre_frames)["mean"])
    dynamic_direction = meta["dynamic"].to_numpy() & meta["feature_family"].isin(
        ["moving_edge", "moving_bar", "small_translating_target"]
    ).to_numpy()
    dynamic_family = meta["dynamic"].to_numpy()

    all_model_features = {"flyvis": fly_features}
    if not args.skip_controls:
        all_model_features.update(run_baseline_features(np.asarray(stimuli), coords, args))

    direction_mats = []
    family_mats = []
    scale_rows = []
    loso_rows = []
    curves = []
    for model, x in all_model_features.items():
        x = zscore(cap_features(x, args.max_features))
        direction_mats.append(scale_generalization(x, meta, "direction", dynamic_direction, model, "mean", args.seed))
        family_mats.append(scale_generalization(x, meta, "feature_family", dynamic_family, model, "mean", args.seed))
        loso_rows.append(leave_one_scale(x, meta, "direction", dynamic_direction, model, args.seed))
        loso_rows.append(leave_one_scale(x, meta, "feature_family", dynamic_family, model, args.seed))
        scale_rows.append(leave_one_scale(x, meta, "scale", dynamic_family, model, args.seed))
        for family in sorted(meta.loc[dynamic_direction, "feature_family"].unique()):
            subset = dynamic_direction & (meta["feature_family"].to_numpy() == family)
            c = within_scale_curve(x, meta, "direction", subset, model, args.seed)
            c["feature_family"] = family
            curves.append(c)

    direction_matrix = pd.concat(direction_mats, ignore_index=True)
    family_matrix = pd.concat(family_mats, ignore_index=True)
    loso = pd.concat(loso_rows, ignore_index=True)
    scale_decode = pd.concat(scale_rows, ignore_index=True)
    curves_df = pd.concat(curves, ignore_index=True)
    breakpoints = estimate_breakpoints(curves_df, "direction", n_classes=int(meta.loc[dynamic_direction, "direction"].nunique()))

    direction_matrix.to_csv(tables / "direction_scale_generalization.csv", index=False)
    family_matrix.to_csv(tables / "feature_family_scale_generalization.csv", index=False)
    loso.to_csv(tables / "leave_one_scale_decoding.csv", index=False)
    scale_decode.to_csv(tables / "scale_nuisance_decoding.csv", index=False)
    curves_df.to_csv(tables / "within_scale_direction_curves.csv", index=False)
    breakpoints.to_csv(tables / "breakpoints_by_feature_family.csv", index=False)
    type_summary.groupby(["cell_type", "feature_family", "scale"], as_index=False).agg(
        peak_delta_response=("peak_delta_response", "mean"),
        mean_delta_response=("mean_delta_response", "mean"),
        latency_to_peak=("latency_to_peak", "mean"),
    ).to_csv(tables / "typewise_peak_mean_latency_curves.csv", index=False)

    control_summary_rows = []
    for model, sub in direction_matrix.groupby("model"):
        off = sub[sub["train_scale"] != sub["test_scale"]]["accuracy"].mean()
        diag = sub[sub["train_scale"] == sub["test_scale"]]["accuracy"].mean()
        fam_off = family_matrix[(family_matrix["model"] == model) & (family_matrix["train_scale"] != family_matrix["test_scale"])][
            "accuracy"
        ].mean()
        control_summary_rows.append(
            {
                "model": model,
                "offdiag_direction_accuracy": off,
                "diag_direction_accuracy": diag,
                "offdiag_feature_family_accuracy": fam_off,
            }
        )
    control_summary = pd.DataFrame(control_summary_rows)
    control_summary.to_csv(tables / "flyvis_vs_controls_summary.csv", index=False)

    best_worst = []
    for model, sub in loso.groupby("model"):
        for target, ss in sub.groupby("target"):
            best = ss.sort_values("accuracy", ascending=False).iloc[0]
            worst = ss.sort_values("accuracy", ascending=True).iloc[0]
            best_worst.append(
                {
                    "model": model,
                    "target": target,
                    "best_heldout_scale": best.get("heldout_scale"),
                    "best_accuracy": best.accuracy,
                    "worst_heldout_scale": worst.get("heldout_scale"),
                    "worst_accuracy": worst.accuracy,
                }
            )
    best_worst_df = pd.DataFrame(best_worst)
    best_worst_df.to_csv(tables / "best_worst_decoded_labels.csv", index=False)

    plot_figures(out_dir, stimuli, coords, meta, type_summary, direction_matrix, breakpoints, control_summary)

    fly_off = control_summary.loc[control_summary["model"] == "flyvis", "offdiag_direction_accuracy"]
    control_off = control_summary.loc[control_summary["model"] != "flyvis", "offdiag_direction_accuracy"]
    fly_differs = bool(len(fly_off) and len(control_off) and abs(float(fly_off.iloc[0]) - float(control_off.max())) > 0.05)
    offdiag_exists = bool(len(fly_off) and float(fly_off.iloc[0]) > (1.0 / max(1, meta.loc[dynamic_direction, "direction"].nunique()) + 0.05))
    strong_enough = bool(offdiag_exists and fly_differs)
    write_json(
        {
            "stimulus_shape": list(stimuli.shape),
            "response_shape": list(responses.shape),
            "n_trials": int(len(meta)),
            "model_used": "flow/0000/000",
            "dynamic_direction_trials": int(dynamic_direction.sum()),
            "offdiag_scale_generalization_exists": offdiag_exists,
            "flyvis_differs_from_pixel_local_controls": fly_differs,
            "strong_enough_for_paper_direction": strong_enough,
        },
        out_dir / "analysis_manifest.json",
    )

    report = [
        "# FlyVis Pilot v2 Report",
        "",
        "## What Ran",
        "- Generated FlyVis-native retinal projection stimuli with apparent-scale sweeps.",
        "- Ran pretrained FlyVis model `flow/0000/000` on native hex-pixel movies.",
        "- Analyzed direction decoding, feature-family decoding, scale nuisance decoding, scale-generalization, type-wise response curves, and breakpoint scale.",
        "- Compared FlyVis against pixel, local RNN, CNN, and optic-lobe type-rate controls when available.",
        "",
        "## Tensor Shapes",
        f"- Stimuli: `{tuple(stimuli.shape)}` as `(sample, frame, channel, hex_pixel)`.",
        f"- FlyVis central cell responses: `{tuple(responses.shape)}` as `(sample, frame, central_cell_type)`.",
        f"- Number of trials: `{len(meta)}`.",
        "",
        "## Decoding Summary",
        f"- Best/worst held-out scale rows saved in `tables/best_worst_decoded_labels.csv`.",
        f"- FlyVis off-diagonal direction scale generalization exists: `{offdiag_exists}`.",
        f"- FlyVis differs from pixel/local controls by >0.05 off-diagonal direction accuracy: `{fly_differs}`.",
        "",
        "## Best And Worst Decoded Labels",
        best_worst_df.to_csv(index=False) if not best_worst_df.empty else "No decoding rows were produced.",
        "",
        "## Interpretation Guardrails",
        "- These results concern retinal apparent scale and retinal projection, not physical distance.",
        "- Static disk/square/triangle identity is an appendix control, not the main claim.",
        "- The main question is whether dynamic feature representations remain stable across apparent scale.",
        "",
        "## Paper Direction",
        f"- Strong enough for a connectome-paper direction at this stage: `{strong_enough}`.",
        "- Continue only if FlyVis has a distinct breakdown profile from matched controls, dynamic features generalize off-diagonal across scale, or response cost/representation stability differs meaningfully from local controls.",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(report), encoding="utf-8")
    print(f"Wrote FlyVis Pilot v2 analysis to {out_dir}")


if __name__ == "__main__":
    main()
