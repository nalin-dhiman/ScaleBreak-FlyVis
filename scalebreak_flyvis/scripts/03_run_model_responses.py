#!/usr/bin/env python
"""Run baseline/FlyVis-adapter model responses on generated stimuli."""

from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
import pandas as pd

from scalebreak.io import load_array_store, save_array_store
from scalebreak.models import flyvis_available, local_rnn, optic_lobe_type_rate, pixel_baseline, small_cnn_random
from scalebreak.utils import copy_config, ensure_subdir, load_yaml, parse_list_arg, run_info, setup_logging, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="scalebreak_flyvis/configs/analysis.yaml")
    parser.add_argument("--models", default="pixel,local_rnn,optic_lobe_type_rate")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    seed = int(cfg.get("seed", 42))
    stim_dir = Path(cfg["paths"]["stimuli_dir"])
    out_root = ensure_subdir(cfg["paths"].get("activations_dir", "scalebreak_flyvis/outputs/activations"))
    logger = setup_logging(out_root, "03_run_model_responses")
    videos = load_array_store(stim_dir / "stimuli.zarr", "videos")
    meta = pd.read_csv(stim_dir / "metadata.csv")
    requested = parse_list_arg(args.models, ["pixel", "local_rnn", "optic_lobe_type_rate"])
    if args.dry_run:
        logger.info("Would run models %s on %s", requested, videos.shape)
        return
    for model_name in requested:
        model_out = ensure_subdir(out_root / model_name)
        copy_config(args.config, model_out)
        logger.info("Running %s", model_name)
        extra = {}
        if model_name == "pixel":
            activations = pixel_baseline(videos)[:, None, :]
        elif model_name == "local_rnn":
            activations = local_rnn(videos, seed=seed)
        elif model_name == "small_cnn":
            activations = small_cnn_random(videos, seed=seed)
        elif model_name == "optic_lobe_type_rate":
            activations, extra = optic_lobe_type_rate(videos, cfg["paths"].get("type_edges"), seed=seed)
        elif model_name == "flyvis":
            available, message = flyvis_available()
            write_json({"available": available, "message": message}, model_out / "run_info.json")
            logger.warning(message)
            continue
        else:
            raise ValueError(f"Unknown model: {model_name}")
        save_array_store(model_out / "activations.zarr", {"activations": activations}, {"model_name": model_name})
        meta.to_csv(model_out / "metadata.csv", index=False)
        write_json(run_info("03_run_model_responses", seed=seed, extra={"model_name": model_name, "activation_shape": list(activations.shape), **extra}), model_out / "run_info.json")


if __name__ == "__main__":
    main()
