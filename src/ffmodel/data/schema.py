"""Canonical column schema shared by every data source.

All loaders (nflverse via nflreadpy, legacy CSVs) map into these columns so
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

# Counting-stat line. Sources without one of these fields retain the historical
# zero-fill behavior because the downstream volume accounting expects complete
# additive columns.
COUNTING_STAT_COLUMNS = [
    "pass_att",
    "pass_sacks",
    "pass_sacks_available",
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

# Rich nflverse efficiency observations. These are deliberately nullable:
# legacy files and some providers do not measure them, and a missing EPA/air-
# yard observation must not be presented to a model as a measured zero.
OPTIONAL_STAT_COLUMNS = [
    "pass_air_yds",
    "pass_yac",
    "pass_first_downs",
    "pass_epa",
    "pass_pacr",
    "rush_first_downs",
    "rush_epa",
    "rec_air_yds",
    "rec_yac",
    "rec_first_downs",
    "rec_epa",
    "rec_racr",
    "source_target_share",
    "source_air_yards_share",
    "source_wopr",
]

STAT_COLUMNS = COUNTING_STAT_COLUMNS + OPTIONAL_STAT_COLUMNS

PLAYER_WEEK_COLUMNS = ID_COLUMNS + STAT_COLUMNS

_NUMERIC = set(STAT_COLUMNS) | {"season", "week"}


def conform(df: pd.DataFrame) -> pd.DataFrame:
    """Return df with exactly PLAYER_WEEK_COLUMNS, filling absent ones.

    Missing counting columns become 0.0 (a source that lacks e.g. targets
    simply reports none), while unmeasured optional efficiency fields remain
    NaN. Missing id columns become pd.NA. Column order is fixed.
    """
    out = df.copy()
    for col in PLAYER_WEEK_COLUMNS:
        if col not in out.columns:
            out[col] = 0.0 if col in COUNTING_STAT_COLUMNS else pd.NA
    for col in COUNTING_STAT_COLUMNS:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
    for col in OPTIONAL_STAT_COLUMNS:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    for col in ("season", "week"):
        out[col] = pd.to_numeric(out[col], errors="coerce").astype("Int64")
    return out[PLAYER_WEEK_COLUMNS]
