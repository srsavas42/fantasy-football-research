"""Snap-share integration and PFR-to-GSIS identity resolution."""

from __future__ import annotations

import pandas as pd

from ffmodel.features.trailing import add_trailing

SNAP_COLUMNS = [
    "player_id",
    "season",
    "week",
    "team",
    "position",
    "offense_snaps",
    "snap_share",
]


def _share(values: pd.Series) -> pd.Series:
    strings = values.astype("string")
    had_percent = strings.str.endswith("%", na=False)
    numeric = pd.to_numeric(strings.str.rstrip("%"), errors="coerce")
    numeric = numeric.where(~had_percent, numeric / 100.0)
    # nflverse normally supplies a 0..1 fraction, but tolerate providers that
    # use 0..100 percentage points.
    numeric = numeric.where(numeric <= 1.0, numeric / 100.0)
    return numeric.clip(lower=0.0, upper=1.0)


def canonicalize_snaps(snaps: pd.DataFrame, player_dim: pd.DataFrame) -> pd.DataFrame:
    """Map snap rows to GSIS IDs and normalize offensive snap percentage."""
    out = snaps.copy()
    if "player_id" not in out.columns or out["player_id"].isna().all():
        if "player_id" in out.columns:
            out = out.drop(columns="player_id")
        pfr_col = next(
            (c for c in ("pfr_player_id", "pfr_id") if c in out.columns), None
        )
        dim_pfr = next(
            (c for c in ("pfr_id", "pfr_player_id") if c in player_dim.columns), None
        )
        if pfr_col and dim_pfr and "player_id" in player_dim.columns:
            lookup = player_dim[[dim_pfr, "player_id"]].dropna().drop_duplicates(dim_pfr)
            out = out.merge(lookup, left_on=pfr_col, right_on=dim_pfr, how="left")
        else:
            out["player_id"] = pd.NA

    offense_col = next(
        (c for c in ("offense_snaps", "off_snaps") if c in out.columns), None
    )
    pct_col = next(
        (c for c in ("offense_pct", "offense_percentage", "snap_pct") if c in out.columns),
        None,
    )
    out["offense_snaps"] = (
        pd.to_numeric(out[offense_col], errors="coerce") if offense_col else 0.0
    )
    out["snap_share"] = _share(out[pct_col]) if pct_col else pd.NA
    if "team" not in out.columns:
        out["team"] = pd.NA
    if "position" not in out.columns:
        out["position"] = pd.NA
    for column in ("season", "week"):
        out[column] = pd.to_numeric(out[column], errors="coerce").astype("Int64")

    keep = [column for column in SNAP_COLUMNS if column in out.columns]
    return out[keep].drop_duplicates(["player_id", "season", "week", "team"])


def merge_snap_usage(
    player_weeks: pd.DataFrame,
    snaps: pd.DataFrame,
    player_dim: pd.DataFrame,
    *,
    span: int = 5,
) -> pd.DataFrame:
    """Left-join observed snaps and add leak-free trailing snap share."""
    canonical = canonicalize_snaps(snaps, player_dim)
    keys = ["player_id", "season", "week", "team"]
    values = canonical[keys + ["offense_snaps", "snap_share"]]
    out = player_weeks.merge(values, on=keys, how="left")
    return add_trailing(out, ["snap_share"], span=span)
