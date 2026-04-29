"""Small statistical helpers."""

from __future__ import annotations

import pandas as pd
import statsmodels.formula.api as smf


def activity_scale_effects(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric in ["mean_abs_activity", "peak_abs_activity", "l2_activity_norm", "temporal_variance"]:
        if metric in df:
            model = smf.ols(f"{metric} ~ scale + contrast + C(shape) + C(motion_type)", data=df).fit()
            rows.append({"metric": metric, "scale_coef": model.params.get("scale", float("nan")), "scale_pvalue": model.pvalues.get("scale", float("nan")), "r2": model.rsquared})
    return pd.DataFrame(rows)
