from __future__ import annotations

from scalebreak.stimuli import generate_stimulus_set


def test_generated_videos_and_metadata_count() -> None:
    cfg = {
        "height": 32,
        "width": 32,
        "n_frames": 5,
        "shapes": ["disk", "square"],
        "scales": [4, 8],
        "contrasts": [1.0],
        "motions": ["static", "translate"],
        "backgrounds": ["blank"],
        "repeats": 2,
        "seed": 1,
    }
    videos, meta = generate_stimulus_set(cfg)
    assert videos.shape == (16, 5, 32, 32)
    assert len(meta) == 16
    assert {"area_pixels", "edge_length_pixels", "bbox_width", "bbox_height"}.issubset(meta.columns)
