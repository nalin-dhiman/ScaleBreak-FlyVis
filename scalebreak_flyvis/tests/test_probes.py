from __future__ import annotations

import numpy as np
import pandas as pd

from scalebreak.probes import protocol_metrics, scale_generalization_matrix


def test_linear_probe_runs_on_toy_features() -> None:
    rng = np.random.default_rng(0)
    y = np.array(["a", "b"] * 20)
    x = rng.normal(size=(40, 4)) + (y == "b")[:, None]
    meta = pd.DataFrame({"shape": y, "scale": [2, 2, 4, 4, 8, 8, 16, 16] * 5, "motion_type": ["static"] * 40, "contrast": [1.0] * 40})
    metrics, reports, cms = protocol_metrics(x, meta, targets=["shape"], seed=0)
    sg = scale_generalization_matrix(x, meta, seed=0)
    assert metrics["accuracy"].notna().all()
    assert "shape" in reports
    assert "shape" in cms
    assert len(sg) > 0
