"""Loaders for the CSVs committed to this repo (fantasydatapros fork).

Coverage: weekly 1999-2021, yearly 1970-2021, snapcounts 2013-2020,
FantasyPros ADP/ECR snapshots. These serve as offline backfill for seasons
(or environments) where nflverse data isn't reachable.
"""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from ffmodel.config import (
    LEGACY_ADP_DIR,
    LEGACY_SNAPCOUNTS_DIR,
    LEGACY_WEEKLY_DIR,
    LEGACY_YEARLY_DIR,
    regular_season_weeks,
)
from ffmodel.data.schema import conform

_WEEKLY_RENAMES = {
    "Player": "player_name",
    "Pos": "position",
    "Tm": "team",
    "PassingYds": "pass_yds",
    "PassingTD": "pass_td",
    "Int": "pass_int",
    "PassingAtt": "pass_att",
    "Cmp": "pass_cmp",
    "RushingAtt": "rush_att",
    "RushingYds": "rush_yds",
    "RushingTD": "rush_td",
    "Rec": "receptions",
    "Tgt": "targets",
    "ReceivingYds": "rec_yds",
    "ReceivingTD": "rec_td",
    "FL": "fumbles_lost",
}

_YEARLY_RENAMES = {**_WEEKLY_RENAMES, "FumblesLost": "fumbles_lost"}


def _read_csv(path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Some files carry an unnamed index column; drop anything unnamed.
    return df.loc[:, ~df.columns.str.match(r"^Unnamed|^$")]


def load_weekly(seasons: Iterable[int]) -> pd.DataFrame:
    """Player-week stat lines from weekly/{season}/week{n}.csv, canonical schema."""
    frames = []
    for season in seasons:
        season_dir = LEGACY_WEEKLY_DIR / str(season)
        if not season_dir.exists():
            continue
        for week in range(1, regular_season_weeks(season) + 1):
            path = season_dir / f"week{week}.csv"
            if not path.exists():
                continue
            df = _read_csv(path).rename(columns=_WEEKLY_RENAMES)
            df["season"] = season
            df["week"] = week
            frames.append(df)
    if not frames:
        return conform(pd.DataFrame())
    out = pd.concat(frames, ignore_index=True)
    out["source"] = "legacy"
    return conform(out)


def load_yearly(seasons: Iterable[int]) -> pd.DataFrame:
    """Player-season stat lines from yearly/{season}.csv, canonical schema (week is NA)."""
    frames = []
    for season in seasons:
        path = LEGACY_YEARLY_DIR / f"{season}.csv"
        if not path.exists():
            continue
        df = _read_csv(path)
        # Old yearly files repeat raw PFR column names (Att, Yds, ...); keep only
        # the derived columns, which exist in every year.
        df = df.loc[:, ~df.columns.duplicated()]
        df = df.rename(columns=_YEARLY_RENAMES)
        df["season"] = season
        frames.append(df)
    if not frames:
        return conform(pd.DataFrame())
    out = pd.concat(frames, ignore_index=True)
    out["source"] = "legacy"
    out["week"] = pd.NA
    return conform(out)


def load_snapcounts(seasons: Iterable[int]) -> pd.DataFrame:
    """Season-level snap shares from snapcounts/{season}.csv (2013-2020)."""
    frames = []
    for season in seasons:
        path = LEGACY_SNAPCOUNTS_DIR / f"{season}.csv"
        if not path.exists():
            continue
        df = _read_csv(path).rename(
            columns={
                "Name": "player_name",
                "Pos": "position",
                "Team": "team",
                "G": "games",
                "Snaps": "snaps",
                "TeamSnaps": "team_snaps",
                "Snap%": "snap_pct",
            }
        )
        df["season"] = season
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out["source"] = "legacy"
    return out


def load_adp(scoring: str = "ppr") -> pd.DataFrame:
    """FantasyPros ADP snapshot (2021 preseason): PPR / HALF_PPR / STANDARD."""
    name = {"ppr": "PPR_ADP.csv", "half_ppr": "HALF_PPR_ADP.csv", "standard": "STANDARD_ADP.csv"}[scoring]
    return _read_csv(LEGACY_ADP_DIR / name)
