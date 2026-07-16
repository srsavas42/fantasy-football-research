"""Canonical column schema shared by every data source.

All loaders (nflverse via nfl_data_py, legacy CSVs) map into these columns so
downstream features/models never care where a row came from.
"""

from __future__ import annotations

import pandas as pd

# Identity / context
ID_COLUMNS = [
    "player_id",      # nflverse gsis id when available, else None
    "player_name",
    "position",       # QB / RB / WR / TE (others passed through as-is)
    "team",
    "season",
    "week",           # None for season-aggregate rows
    "source",         # "nflverse" | "legacy"
]

# Stat line
STAT_COLUMNS = [
    "pass_att",
    "pass_cmp",
    "pass_yds",
    "pass_td",
    "pass_int",
    "rush_att",
    "rush_yds",
    "rush_td",
    "targets",
    "receptions",
    "rec_yds",
    "rec_td",
    "fumbles_lost",
]

PLAYER_WEEK_COLUMNS = ID_COLUMNS + STAT_COLUMNS

_NUMERIC = set(STAT_COLUMNS) | {"season", "week"}


def conform(df: pd.DataFrame) -> pd.DataFrame:
    """Return df with exactly PLAYER_WEEK_COLUMNS, filling absent ones.

    Missing stat columns become 0.0 (a source that lacks e.g. targets simply
    reports none); missing id columns become pd.NA. Column order is fixed.
    """
    out = df.copy()
    for col in PLAYER_WEEK_COLUMNS:
        if col not in out.columns:
            out[col] = 0.0 if col in STAT_COLUMNS else pd.NA
    for col in STAT_COLUMNS:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
    for col in ("season", "week"):
        out[col] = pd.to_numeric(out[col], errors="coerce").astype("Int64")
    return out[PLAYER_WEEK_COLUMNS]
