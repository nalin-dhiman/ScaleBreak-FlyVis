from __future__ import annotations

import numpy as np
import pandas as pd

from scalebreak.rsa_cka import cka_by_scale, linear_cka, rsa_summary


def test_rsa_cka_finite() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(32, 6))
    meta = pd.DataFrame({"shape": ["a", "b"] * 16, "scale": [2, 4, 8, 16] * 8})
    rsa, summary = rsa_summary(x, meta)
    cka = cka_by_scale(x, meta)
    assert np.isfinite(rsa.values).all()
    assert np.isfinite(summary["scale_invariance_margin"]).all()
    assert np.isfinite(cka["cka"]).all()
    assert np.isfinite(linear_cka(x, x))
