"""Trailing per-opportunity efficiency covariates.

Efficient players earn future volume, so a player's *past* efficiency is a
leading indicator of their future opportunity share. We compute per-week
efficiency ratios (undefined when the denominator is zero -> NaN, deliberately
excluded from the EWMA rather than counted as zero), then their leak-free
trailing EWMAs via `trailing.add_trailing`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ffmodel.features.trailing import add_trailing

# Per-week ratios; NaN where the opportunity denominator is 0.
EFFICIENCY_RATIOS = ["ypt", "ypc", "catch_rate", "yds_per_touch", "td_rate"]

# The trailing versions are what models actually consume.
EFFICIENCY_COLUMNS = [f"ewma_{c}" for c in EFFICIENCY_RATIOS]


def _ratio(num: pd.Series, den: pd.Series) -> pd.Series:
    """num/den with 0-denominator -> NaN (undefined, not zero)."""
    num = pd.to_numeric(num, errors="coerce")
    den = pd.to_numeric(den, errors="coerce")
    den_arr = den.to_numpy(dtype=float)
    out = np.divide(num, den, out=np.full(len(num), np.nan), where=(den_arr != 0))
    return pd.Series(out, index=num.index)


def add_efficiency(pw: pd.DataFrame, span: int = 5) -> pd.DataFrame:
    """Add per-week efficiency ratios and their trailing EWMAs."""
    out = pw.copy()
    touches = out["rush_att"] + out["receptions"]

    out["ypt"] = _ratio(out["rec_yds"], out["targets"])
    out["ypc"] = _ratio(out["rush_yds"], out["rush_att"])
    out["catch_rate"] = _ratio(out["receptions"], out["targets"])
    out["yds_per_touch"] = _ratio(out["rush_yds"] + out["rec_yds"], touches)
    out["td_rate"] = _ratio(out["rush_td"] + out["rec_td"], touches)

    return add_trailing(out, EFFICIENCY_RATIOS, span=span)
