"""Leak-free trailing covariates.

Every model predictor that summarizes a player's past lives here, computed in
exactly one place so the no-look-ahead guarantee is tested once. For week W,
a trailing feature sees only weeks strictly before W: we sort chronologically,
group per player, `shift(1)`, then take an EWMA of that lagged series.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

# Player identity for grouping. player_id is populated on the nflverse path;
# the legacy CSV path has none, so we fall back to name+position.
_PRIMARY_KEY = "player_id"
_FALLBACK_KEY = ["player_name", "position"]


def player_key(df: pd.DataFrame) -> pd.Series:
    """A stable per-player grouping key, robust to a missing player_id."""
    if _PRIMARY_KEY in df.columns and df[_PRIMARY_KEY].notna().any():
        pid = df[_PRIMARY_KEY]
        # Fill rows lacking an id with a name+position surrogate so no two
        # distinct players collapse into one NA bucket.
        surrogate = df[_FALLBACK_KEY[0]].astype(str) + "|" + df[_FALLBACK_KEY[1]].astype(str)
        return pid.where(pid.notna(), surrogate).astype(str)
    return (df[_FALLBACK_KEY[0]].astype(str) + "|" + df[_FALLBACK_KEY[1]].astype(str))


def _sort_chrono(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(["season", "week"], kind="stable")


def add_trailing(
    df: pd.DataFrame,
    cols: Sequence[str],
    span: int = 5,
    min_periods: int = 1,
    prefix: str = "ewma_",
) -> pd.DataFrame:
    """Return df with `prefix+col` trailing EWMA columns for each of `cols`.

    The EWMA at row W uses only that player's weeks < W (via shift(1)), so the
    result is safe to use as a predictor of week W. Rows are returned in the
    original order. NaN inputs (e.g. an undefined efficiency ratio) are skipped
    by the EWMA rather than poisoning it.
    """
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"add_trailing: columns not found: {missing}")

    work = df.copy()
    work["_key"] = player_key(work)
    work["_orig"] = np.arange(len(work))
    work = _sort_chrono(work)

    def _trailing(series: pd.Series) -> pd.Series:
        # shift(1) => week W excludes its own value; EWMA over strictly-past weeks.
        return series.shift(1).ewm(span=span, min_periods=min_periods, ignore_na=True).mean()

    grouped = work.groupby("_key", sort=False, group_keys=False)
    for col in cols:
        work[prefix + col] = grouped[col].apply(_trailing)

    work = work.sort_values("_orig", kind="stable").drop(columns=["_key", "_orig"])
    return work
