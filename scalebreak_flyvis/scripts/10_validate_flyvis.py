#!/usr/bin/env python
"""Validate FlyVis installation with a canonical moving-edge response.

This script intentionally does not run the ScaleBreak pipeline. It only checks
that a pretrained FlyVis model can be loaded, accepts a canonical moving-edge
stimulus, exposes activations, saves cell/type-level outputs, and plots a known
motion-sensitive response trace.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def parse_angles(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def to_jsonable(value: Any) -> Any:
    try:
        import numpy as np

        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
    except Exception:
        pass
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=to_jsonable)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--flyvis-root", default="scalebreak_flyvis/flyvis_data")
    parser.add_argument("--out-dir", default="scalebreak_flyvis/outputs/flyvis_validation")
    parser.add_argument("--model", default="flow/0000/000")
    parser.add_argument("--angles", default="0,60,180,240")
    parser.add_argument("--speed", type=float, default=19.0)
    parser.add_argument("--height", type=int, default=80)
    parser.add_argument("--t-pre", type=float, default=0.2)
    parser.add_argument("--t-post", type=float, default=0.2)
    parser.add_argument("--dt", type=float, default=1 / 200)
    parser.add_argument("--target-cell-type", default="T4c")
    args = parser.parse_args()

    repo_root = Path.cwd()
    flyvis_root = (repo_root / args.flyvis_root).resolve()
    out_dir = (repo_root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("FLYVIS_ROOT_DIR", str(flyvis_root))
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl_flyvis")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import torch
    import flyvis
    from flyvis import results_dir
    from flyvis.datasets.moving_bar import MovingEdge
    from flyvis.network import NetworkView

    status: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "flyvis_version": getattr(flyvis, "__version__", "unknown"),
        "torch_version": torch.__version__,
        "flyvis_root": str(flyvis_root),
        "results_dir": str(results_dir),
        "model": args.model,
        "checks": {
            "model_loads": False,
            "stimulus_shape_accepted": False,
            "activations_extracted": False,
            "cell_outputs_saved": False,
            "type_outputs_saved": False,
            "known_response_plotted": False,
        },
    }

    angles = parse_angles(args.angles)
    dataset = MovingEdge(
        offsets=[-10, 11],
        intensities=[1],
        speeds=[args.speed],
        height=args.height,
        post_pad_mode="continue",
        t_pre=args.t_pre,
        t_post=args.t_post,
        dt=args.dt,
        angles=angles,
        device="cpu",
    )
    sample0 = dataset[0]
    status["stimulus_sample_shape"] = list(sample0.shape)
    status["stimulus_arg_df_rows"] = int(len(dataset.arg_df))
    status["checks"]["stimulus_shape_accepted"] = tuple(sample0.shape)[-1] > 0
    dataset.arg_df.to_csv(out_dir / "stimulus_metadata.csv", index=False)

    network_view = NetworkView(results_dir / args.model)
    status["checks"]["model_loads"] = True

    stims_and_resps = network_view.moving_edge_responses(dataset)
    responses = stims_and_resps["responses"]
    stimulus = stims_and_resps["stimulus"]
    response_values = np.asarray(responses.values, dtype=np.float32)
    stimulus_values = np.asarray(stimulus.values, dtype=np.float32)
    status["response_dims"] = list(responses.dims)
    status["response_shape"] = list(response_values.shape)
    status["stimulus_dims"] = list(stimulus.dims)
    status["full_stimulus_shape"] = list(stimulus_values.shape)
    status["checks"]["activations_extracted"] = response_values.size > 0 and np.isfinite(response_values).all()

    np.save(out_dir / "flyvis_cell_responses.npy", response_values)
    np.save(out_dir / "flyvis_stimulus.npy", stimulus_values)

    cell_meta = pd.DataFrame(
        {
            "neuron": np.asarray(responses.coords["neuron"].values),
            "cell_type": np.asarray(responses.coords["cell_type"].values).astype(str),
            "u": np.asarray(responses.coords["u"].values),
            "v": np.asarray(responses.coords["v"].values),
        }
    )
    cell_meta.to_csv(out_dir / "cell_metadata.csv", index=False)

    sample_meta = pd.DataFrame(
        {
            "sample": np.asarray(responses.coords["sample"].values),
            "angle": np.asarray(responses.coords["angle"].values),
            "intensity": np.asarray(responses.coords["intensity"].values),
            "speed": np.asarray(responses.coords["speed"].values),
        }
    )
    sample_meta.to_csv(out_dir / "response_sample_metadata.csv", index=False)

    # Save compact cell-level peak/final summaries.
    baseline = response_values[:, :, : max(1, int(args.t_pre / args.dt)), :].mean(axis=2, keepdims=True)
    delta = response_values - baseline
    peak = delta.max(axis=2)[0]  # sample x neuron for the single network.
    final = delta[:, :, -1, :][0]
    cell_rows = []
    for si, sample in sample_meta.iterrows():
        for ni, cell in cell_meta.iterrows():
            cell_rows.append(
                {
                    "sample": int(sample["sample"]),
                    "angle": float(sample["angle"]),
                    "speed": float(sample["speed"]),
                    "neuron": int(cell["neuron"]),
                    "cell_type": cell["cell_type"],
                    "peak_delta_response": float(peak[si, ni]),
                    "final_delta_response": float(final[si, ni]),
                }
            )
    pd.DataFrame(cell_rows).to_csv(out_dir / "cell_response_summary.csv", index=False)
    status["checks"]["cell_outputs_saved"] = True

    # Type-level time courses: mean over neurons with the same central cell type.
    time = np.asarray(responses.coords["time"].values)
    cell_types = cell_meta["cell_type"].to_numpy()
    type_rows = []
    unique_types = sorted(cell_meta["cell_type"].unique())
    for si, sample in sample_meta.iterrows():
        for cell_type in unique_types:
            idx = np.flatnonzero(cell_types == cell_type)
            trace = response_values[0, si, :, :][:, idx].mean(axis=1)
            base = trace[: max(1, int(args.t_pre / args.dt))].mean()
            for ti, tt in enumerate(time):
                type_rows.append(
                    {
                        "sample": int(sample["sample"]),
                        "angle": float(sample["angle"]),
                        "speed": float(sample["speed"]),
                        "time": float(tt),
                        "cell_type": cell_type,
                        "mean_response": float(trace[ti]),
                        "baseline_subtracted_response": float(trace[ti] - base),
                    }
                )
    type_df = pd.DataFrame(type_rows)
    type_df.to_csv(out_dir / "type_level_responses.csv", index=False)
    type_df.groupby(["cell_type", "angle"], as_index=False)["baseline_subtracted_response"].max().to_csv(
        out_dir / "type_level_peak_summary.csv", index=False
    )
    status["checks"]["type_outputs_saved"] = True

    # Known response figure: T4c moving-edge traces. If T4c is unavailable, use
    # the strongest peak type but record that substitution.
    target = args.target_cell_type
    if target not in set(unique_types):
        target = (
            type_df.groupby("cell_type")["baseline_subtracted_response"].max().sort_values(ascending=False).index[0]
        )
        status["target_cell_type_substituted"] = target
    plot_df = type_df[type_df["cell_type"] == target]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), constrained_layout=True)
    first_sample = int(sample_meta.iloc[0]["sample"])
    axes[0].imshow(stimulus_values[first_sample, stimulus_values.shape[1] // 2, 0].reshape(1, -1), aspect="auto", cmap="gray")
    axes[0].set_title("Moving-edge stimulus, mid-frame")
    axes[0].set_xlabel("hex pixel")
    axes[0].set_yticks([])
    for angle, sub in plot_df.groupby("angle"):
        axes[1].plot(sub["time"], sub["baseline_subtracted_response"], label=f"{angle:g} deg")
    axes[1].axvline(0, color="0.6", lw=1, ls="--")
    axes[1].set_title(f"{target} FlyVis response to moving edge")
    axes[1].set_xlabel("time (s)")
    axes[1].set_ylabel("activity proxy, baseline-subtracted")
    axes[1].legend(frameon=False, fontsize=8)
    fig.savefig(out_dir / "fig_flyvis_moving_edge_response.png", dpi=180)
    plt.close(fig)
    status["plotted_cell_type"] = target
    status["checks"]["known_response_plotted"] = (out_dir / "fig_flyvis_moving_edge_response.png").exists()

    write_json(status, out_dir / "validation_status.json")
    print(json.dumps(status, indent=2, default=to_jsonable))
    if not all(status["checks"].values()):
        raise SystemExit("FlyVis validation did not satisfy all checks.")


if __name__ == "__main__":
    main()
