"""nflverse ingestion via nfl_data_py, with a local parquet cache.

Every public function caches per (dataset, season) under config.CACHE_DIR so
repeated pipeline runs don't re-download. Player-week stats are mapped to the
canonical schema in schema.py; roster/context datasets keep nflverse columns.

nflverse distributes most datasets as GitHub release assets. In sandboxed
environments where github.com is unreachable these raise DataUnavailableError;
the cache and the legacy CSV loaders keep offline work possible.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from ffmodel.data import cache
from ffmodel.data.schema import conform


class DataUnavailableError(RuntimeError):
    """A remote nflverse dataset could not be downloaded."""


def _fetch(dataset: str, fetch_fn, season: int | None = None, refresh: bool = False,
           cache_dir: Path | None = None) -> pd.DataFrame:
    try:
        return cache.get_or_fetch(dataset, fetch_fn, season=season, refresh=refresh,
                                  cache_dir=cache_dir)
    except OSError as exc:  # HTTPError/URLError/connection failures
        raise DataUnavailableError(
            f"Could not download nflverse dataset '{dataset}'"
            + (f" for {season}" if season else "")
            + f": {exc}. If this environment blocks github.com, use the parquet "
            "cache or the legacy CSV loaders (ffmodel.data.legacy)."
        ) from exc


_WEEKLY_RENAMES = {
    "player_display_name": "player_name",
    "recent_team": "team",
    "attempts": "pass_att",
    "completions": "pass_cmp",
    "passing_yards": "pass_yds",
    "passing_tds": "pass_td",
    "interceptions": "pass_int",
    "carries": "rush_att",
    "rushing_yards": "rush_yds",
    "rushing_tds": "rush_td",
    "receiving_yards": "rec_yds",
    "receiving_tds": "rec_td",
}


def load_weekly(seasons: Iterable[int], refresh: bool = False,
                cache_dir: Path | None = None) -> pd.DataFrame:
    """Player-week stat lines (regular season) in the canonical schema."""
    import nfl_data_py as nfl

    frames = []
    for season in seasons:
        raw = _fetch("weekly", lambda s=season: nfl.import_weekly_data([s]),
                     season=season, refresh=refresh, cache_dir=cache_dir)
        frames.append(raw)
    raw = pd.concat(frames, ignore_index=True)
    if "season_type" in raw.columns:
        raw = raw[raw["season_type"] == "REG"]
    df = raw.rename(columns=_WEEKLY_RENAMES)
    df["fumbles_lost"] = (
        df.get("sack_fumbles_lost", 0)
        + df.get("rushing_fumbles_lost", 0)
        + df.get("receiving_fumbles_lost", 0)
    )
    df["source"] = "nflverse"
    return conform(df)


def load_snap_counts(seasons: Iterable[int], refresh: bool = False,
                     cache_dir: Path | None = None) -> pd.DataFrame:
    """Per-game snap counts/percentages (2012+), nflverse columns."""
    import nfl_data_py as nfl

    return pd.concat(
        [_fetch("snap_counts", lambda s=season: nfl.import_snap_counts([s]),
                season=season, refresh=refresh, cache_dir=cache_dir)
         for season in seasons],
        ignore_index=True,
    )


def load_depth_charts(seasons: Iterable[int], refresh: bool = False,
                      cache_dir: Path | None = None) -> pd.DataFrame:
    """Listed depth charts (2001+) — cold-start fallback only, not ground truth."""
    import nfl_data_py as nfl

    return pd.concat(
        [_fetch("depth_charts", lambda s=season: nfl.import_depth_charts([s]),
                season=season, refresh=refresh, cache_dir=cache_dir)
         for season in seasons],
        ignore_index=True,
    )


def load_injuries(seasons: Iterable[int], refresh: bool = False,
                  cache_dir: Path | None = None) -> pd.DataFrame:
    """Weekly injury reports (2009+)."""
    import nfl_data_py as nfl

    return pd.concat(
        [_fetch("injuries", lambda s=season: nfl.import_injuries([s]),
                season=season, refresh=refresh, cache_dir=cache_dir)
         for season in seasons],
        ignore_index=True,
    )


def load_schedules(seasons: Iterable[int], refresh: bool = False,
                   cache_dir: Path | None = None) -> pd.DataFrame:
    """Game schedules/results incl. Vegas spread & total (game-script priors)."""
    import nfl_data_py as nfl

    df = _fetch("schedules", lambda: nfl.import_schedules(list(seasons)),
                refresh=True if refresh else False, cache_dir=cache_dir)
    return df[df["season"].isin(list(seasons))]


def load_rosters(seasons: Iterable[int], refresh: bool = False,
                 cache_dir: Path | None = None) -> pd.DataFrame:
    """Seasonal rosters (age, experience, team changes)."""
    import nfl_data_py as nfl

    return pd.concat(
        [_fetch("rosters", lambda s=season: nfl.import_seasonal_rosters([s]),
                season=season, refresh=refresh, cache_dir=cache_dir)
         for season in seasons],
        ignore_index=True,
    )


def load_ids(refresh: bool = False, cache_dir: Path | None = None) -> pd.DataFrame:
    """Cross-source player id map (gsis, pfr, fantasypros, ...)."""
    import nfl_data_py as nfl

    return _fetch("ids", nfl.import_ids, refresh=refresh, cache_dir=cache_dir)
