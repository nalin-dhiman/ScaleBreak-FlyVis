#!/usr/bin/env python
"""Generate synthetic retinal videos and metadata."""

from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scalebreak.io import save_array_store
from scalebreak.plotting import save_stimulus_montage
from scalebreak.stimuli import generate_stimulus_set
from scalebreak.utils import copy_config, ensure_subdir, load_yaml, run_info, setup_logging, write_json


def qa_plots(videos, meta, out_dir: Path) -> None:
    save_stimulus_montage(videos, meta, out_dir / "montage.png")
    examples = meta.drop_duplicates(["shape", "scale"]).sort_values(["shape", "scale"]).head(32).index
    save_stimulus_montage(videos[examples], meta.loc[examples].reset_index(drop=True), out_dir / "qa_shape_examples_across_scale.png", n=len(examples))
    meta.groupby("scale")[["area_pixels", "edge_length_pixels"]].mean().plot(marker="o", subplots=True, figsize=(6, 5))
    plt.tight_layout()
    plt.savefig(out_dir / "qa_area_edge_vs_scale.png", dpi=160)
    plt.close()
    motion = meta[meta["motion_type"] == "translate"].head(1)
    if len(motion):
        i = int(motion.index[0])
        frames = videos[i, :: max(1, videos.shape[1] // 6)]
        fig, axes = plt.subplots(1, len(frames), figsize=(len(frames) * 1.6, 1.8))
        for ax, frame in zip(axes, frames):
            ax.imshow(frame, cmap="gray", vmin=0, vmax=1)
            ax.axis("off")
        plt.tight_layout()
        plt.savefig(out_dir / "qa_motion_trajectory_examples.png", dpi=160)
        plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="scalebreak_flyvis/configs/stimulus_grid_pilot.yaml")
    parser.add_argument("--out-dir")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    out_dir = ensure_subdir(args.out_dir or cfg.get("out_dir", "scalebreak_flyvis/outputs/stimuli/pilot"))
    logger = setup_logging(out_dir, "02_generate_stimuli")
    copy_config(args.config, out_dir)
    if args.dry_run:
        logger.info("Dry run config: %s", cfg)
        return
    videos, meta = generate_stimulus_set(cfg)
    save_array_store(out_dir / "stimuli.zarr", {"videos": videos}, {"format": "video_n_t_h_w"})
    meta.to_csv(out_dir / "metadata.csv", index=False)
    qa_plots(videos, meta, out_dir)
    audit = {
        "n_trials": int(len(meta)),
        "video_shape": list(videos.shape),
        "shapes": sorted(meta["shape"].unique()),
        "scales": sorted(meta["scale"].unique().tolist()),
        "motions": sorted(meta["motion_type"].unique()),
        "contrasts": sorted(meta["contrast"].unique().tolist()),
    }
    write_json(audit, out_dir / "stimulus_audit.json")
    write_json(run_info("02_generate_stimuli", seed=int(cfg.get("seed", 42)), extra=audit), out_dir / "run_info.json")
    logger.info("Generated %d stimuli with shape %s", len(meta), videos.shape)


if __name__ == "__main__":
    main()
