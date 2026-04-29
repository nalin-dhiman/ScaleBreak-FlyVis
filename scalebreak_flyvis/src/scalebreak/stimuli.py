"""Synthetic retinal video stimulus generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .geometry import apply_blur, bbox, binary_shape, edge_length


@dataclass(frozen=True)
class StimulusSpec:
    shape: str
    scale: float
    contrast: float
    blur_sigma: float
    motion_type: str
    velocity_x: float
    velocity_y: float
    expansion_rate: float
    position_x: float
    position_y: float
    orientation: float
    background_type: str
    height: int
    width: int
    n_frames: int
    seed: int
    trial_id: str


def render_background(kind: str, height: int, width: int, rng: np.random.Generator) -> np.ndarray:
    if kind == "blank":
        return np.zeros((height, width), dtype=np.float32)
    if kind == "noise":
        return rng.normal(0, 0.05, size=(height, width)).astype(np.float32)
    raise ValueError(f"Unknown background_type: {kind}")


def generate_video(spec: StimulusSpec) -> tuple[np.ndarray, dict[str, Any]]:
    rng = np.random.default_rng(spec.seed)
    video = np.zeros((spec.n_frames, spec.height, spec.width), dtype=np.float32)
    first_mask: np.ndarray | None = None
    for t in range(spec.n_frames):
        if spec.motion_type == "static":
            dx = dy = 0.0
            scale_t = spec.scale
        elif spec.motion_type == "translate":
            centered = t - (spec.n_frames - 1) / 2.0
            dx = spec.velocity_x * centered
            dy = spec.velocity_y * centered
            scale_t = spec.scale
        elif spec.motion_type == "loom":
            dx = dy = 0.0
            scale_t = spec.scale * (1.0 + spec.expansion_rate * t)
        else:
            raise ValueError(f"Unknown motion_type: {spec.motion_type}")

        bg = render_background(spec.background_type, spec.height, spec.width, rng)
        mask = binary_shape(
            spec.shape,
            spec.height,
            spec.width,
            scale_t,
            spec.position_x + dx,
            spec.position_y + dy,
            spec.orientation,
        )
        if first_mask is None:
            first_mask = mask.copy()
        frame = bg + spec.contrast * mask
        frame = apply_blur(frame, spec.blur_sigma)
        video[t] = np.clip(frame, -1.0, 1.0)

    assert first_mask is not None
    bw, bh = bbox(first_mask)
    meta = {
        **spec.__dict__,
        "area_pixels": float(first_mask.sum()),
        "edge_length_pixels": edge_length(first_mask),
        "bbox_width": bw,
        "bbox_height": bh,
    }
    return video, meta


def specs_from_config(config: dict[str, Any]) -> list[StimulusSpec]:
    rng = np.random.default_rng(int(config.get("seed", 42)))
    height = int(config.get("height", 64))
    width = int(config.get("width", 64))
    n_frames = int(config.get("n_frames", 30))
    reps = int(config.get("repeats", 20))
    center_x = float(config.get("position_x", width / 2))
    center_y = float(config.get("position_y", height / 2))
    blur_sigmas = config.get("blur_sigmas", [0.0])
    specs: list[StimulusSpec] = []
    idx = 0
    for shape in config.get("shapes", ["disk", "square", "triangle", "bar"]):
        for scale in config.get("scales", [2, 3, 4, 6, 8, 12, 16, 24]):
            for contrast in config.get("contrasts", [1.0, 0.3]):
                for motion in config.get("motions", ["static", "translate"]):
                    for background in config.get("backgrounds", ["blank"]):
                        for blur_sigma in blur_sigmas:
                            for rep in range(reps):
                                seed = int(rng.integers(0, 2**31 - 1))
                                vx = 0.0 if motion == "static" else float(config.get("velocity_x", 0.35))
                                vy = 0.0 if motion == "static" else float(config.get("velocity_y", 0.0))
                                orientation = float(rng.choice(config.get("orientations", [0.0])))
                                specs.append(
                                    StimulusSpec(
                                        shape=str(shape),
                                        scale=float(scale),
                                        contrast=float(contrast),
                                        blur_sigma=float(blur_sigma),
                                        motion_type=str(motion),
                                        velocity_x=vx,
                                        velocity_y=vy,
                                        expansion_rate=float(config.get("expansion_rate", 0.0)),
                                        position_x=center_x,
                                        position_y=center_y,
                                        orientation=orientation,
                                        background_type=str(background),
                                        height=height,
                                        width=width,
                                        n_frames=n_frames,
                                        seed=seed,
                                        trial_id=f"trial_{idx:06d}",
                                    )
                                )
                                idx += 1
    return specs


def generate_stimulus_set(config: dict[str, Any]) -> tuple[np.ndarray, pd.DataFrame]:
    specs = specs_from_config(config)
    videos = []
    metas = []
    for spec in specs:
        video, meta = generate_video(spec)
        videos.append(video)
        metas.append(meta)
    return np.stack(videos).astype(np.float32), pd.DataFrame(metas)
