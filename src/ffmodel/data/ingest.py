"""nflverse ingestion via nflreadpy, with parameter-aware parquet caching.

Public functions return pandas DataFrames so existing feature code remains
stable while the provider adapter handles nflreadpy's Polars return values.
Player-week stats are conformed to :mod:`ffmodel.data.schema`; richer datasets
retain their upstream columns and live at their natural grain.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd

from ffmodel.data import cache
from ffmodel.data.providers import nflverse
from ffmodel.data.schema import conform

_NFLVERSE_URL = "https://github.com/nflverse/nflverse-data"
_NFLVERSE_LICENSE = "Upstream-specific; mostly CC-BY-4.0"


class DataUnavailableError(RuntimeError):
    """A remote nflverse dataset could not be downloaded."""


def _fetch(
    dataset: str,
    fetch_fn: Callable[[], pd.DataFrame],
    *,
    season: int | None = None,
    params: Mapping[str, Any] | None = None,
    as_of: str | date | datetime | None = None,
    refresh: bool = False,
    cache_dir: Path | None = None,
) -> pd.DataFrame:
    try:
        return cache.get_or_fetch(
            dataset,
            fetch_fn,
            season=season,
            refresh=refresh,
            cache_dir=cache_dir,
            provider="nflverse",
            params=params,
            as_of=as_of,
            source_url=_NFLVERSE_URL,
            license_name=_NFLVERSE_LICENSE,
        )
    except (OSError, nflverse.NflverseProviderError) as exc:
        raise DataUnavailableError(
            f"Could not load nflverse dataset {dataset!r}"
            + (f" for {season}" if season is not None else "")
            + f": {exc}. Use a populated cache or a legacy loader when available."
        ) from exc


def _by_season(
    dataset: str,
    seasons: Iterable[int],
    *,
    params: Mapping[str, Any] | None = None,
    refresh: bool = False,
    cache_dir: Path | None = None,
) -> pd.DataFrame:
    frames = []
    for season in map(int, seasons):
        frames.append(
            _fetch(
                dataset,
                lambda s=season: nflverse.load(dataset, [s], **(params or {})),
                season=season,
                params=params,
                refresh=refresh,
                cache_dir=cache_dir,
            )
        )
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


_WEEKLY_RENAMES = {
    "player_display_name": "player_name",
    "recent_team": "team",
    "attempts": "pass_att",
    "sacks_suffered": "pass_sacks",
    "completions": "pass_cmp",
    "passing_yards": "pass_yds",
    "passing_tds": "pass_td",
    "interceptions": "pass_int",
    "passing_interceptions": "pass_int",
    "passing_air_yards": "pass_air_yds",
    "passing_yards_after_catch": "pass_yac",
    "passing_first_downs": "pass_first_downs",
    "passing_epa": "pass_epa",
    "pacr": "pass_pacr",
    "carries": "rush_att",
    "rushing_yards": "rush_yds",
    "rushing_tds": "rush_td",
    "rushing_first_downs": "rush_first_downs",
    "rushing_epa": "rush_epa",
    "receiving_yards": "rec_yds",
    "receiving_tds": "rec_td",
    "receiving_air_yards": "rec_air_yds",
    "receiving_yards_after_catch": "rec_yac",
    "receiving_first_downs": "rec_first_downs",
    "receiving_epa": "rec_epa",
    "racr": "rec_racr",
    "target_share": "source_target_share",
    "air_yards_share": "source_air_yards_share",
    "wopr": "source_wopr",
}


def _map_weekly_aliases(raw: pd.DataFrame) -> pd.DataFrame:
    """Map nflverse aliases without creating duplicate canonical labels.

    Recent player-stat releases include both abbreviated ``player_name`` and
    full ``player_display_name``. The richer explicitly mapped source wins,
    with an existing canonical value used only as a null fallback.
    """
    out = raw.copy()
    for source, target in _WEEKLY_RENAMES.items():
        if source not in out.columns or source == target:
            continue
        if target in out.columns:
            out[target] = out[source].combine_first(out[target])
            out = out.drop(columns=source)
        else:
            out = out.rename(columns={source: target})
    return out


def load_weekly(
    seasons: Iterable[int],
    refresh: bool = False,
    cache_dir: Path | None = None,
) -> pd.DataFrame:
    """Regular-season player-week stat lines in the canonical schema."""
    raw = _by_season(
        "player_stats",
        seasons,
        params={"summary_level": "week"},
        refresh=refresh,
        cache_dir=cache_dir,
    )
    if raw.empty:
        return conform(raw)
    if "season_type" in raw.columns:
        raw = raw[raw["season_type"] == "REG"]
    has_pass_sacks = "sacks_suffered" in raw.columns
    df = _map_weekly_aliases(raw)
    # Keep an explicit observation flag. The committed legacy files do not
    # contain sacks, and treating their schema-filled zero as observed would
    # train an impossible zero-sack offense.
    df["pass_sacks_available"] = float(has_pass_sacks)
    fumble_columns = [
        column
        for column in (
            "sack_fumbles_lost",
            "rushing_fumbles_lost",
            "receiving_fumbles_lost",
        )
        if column in df.columns
    ]
    df["fumbles_lost"] = df[fumble_columns].sum(axis=1) if fumble_columns else 0.0
    df["source"] = "nflverse"
    return conform(df)


def load_pbp(seasons: Iterable[int], refresh: bool = False, cache_dir=None) -> pd.DataFrame:
    """Play-by-play data (large; cached one parquet per season)."""
    return _by_season("pbp", seasons, refresh=refresh, cache_dir=cache_dir)


def load_team_stats(
    seasons: Iterable[int],
    summary_level: str = "week",
    refresh: bool = False,
    cache_dir=None,
) -> pd.DataFrame:
    return _by_season(
        "team_stats",
        seasons,
        params={"summary_level": summary_level},
        refresh=refresh,
        cache_dir=cache_dir,
    )


def load_snap_counts(seasons: Iterable[int], refresh: bool = False, cache_dir=None):
    return _by_season("snap_counts", seasons, refresh=refresh, cache_dir=cache_dir)


def load_depth_charts(seasons: Iterable[int], refresh: bool = False, cache_dir=None):
    """Listed depth charts; use as a cold-start fallback, not ground truth.

    nflverse replaced this feed's schema in 2025: weekly rows keyed by
    ``season``/``week`` became timestamped snapshots keyed by ``dt``, with
    renamed team, name, position, and depth columns. Seasons of either shape are
    returned in the historical schema so downstream feature code sees one grain.
    """
    frames = []
    for season in map(int, seasons):
        frame = _by_season(
            "depth_charts", [season], refresh=refresh, cache_dir=cache_dir
        )
        frames.append(
            _conform_depth_charts(
                frame, season, refresh=refresh, cache_dir=cache_dir
            )
        )
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ``pos_grp`` names the personnel package a depth chart belongs to. The offensive
# package is the only one this repo models, but all three are mapped so the
# historical ``formation`` filter keeps behaving as it did.
_DEPTH_FORMATIONS = {
    "3WR 1TE": "Offense",
    "Base 3-4 D": "Defense",
    "Base 4-3 D": "Defense",
    "Special Teams": "Special Teams",
}

_DEPTH_RENAMES = {
    "team": "club_code",
    "player_name": "full_name",
    "pos_abb": "position",
    "pos_rank": "depth_team",
}


def _conform_depth_charts(
    frame: pd.DataFrame,
    season: int,
    *,
    refresh: bool = False,
    cache_dir: Path | None = None,
) -> pd.DataFrame:
    """Map the 2025+ snapshot depth-chart feed onto the historical schema."""
    if frame is None or frame.empty or "dt" not in frame.columns:
        return frame
    out = frame.rename(columns=_DEPTH_RENAMES).copy()
    out["season"] = season
    out["formation"] = out["pos_grp"].map(_DEPTH_FORMATIONS)
    out["depth_team"] = pd.to_numeric(out["depth_team"], errors="coerce")
    stamps = pd.to_datetime(out["dt"], errors="coerce", utc=True).dt.tz_convert(None)
    try:
        schedule = load_schedules([season], refresh=refresh, cache_dir=cache_dir)
    except (DataUnavailableError, OSError):
        # Without kickoff dates a snapshot cannot be placed against a week. Leave
        # the column missing rather than guess: a wrong week silently changes
        # which snapshot a point-in-time cutoff selects.
        schedule = pd.DataFrame()
    out["week"], out["game_type"] = _depth_snapshot_week(stamps, schedule)
    return out


def _depth_snapshot_week(
    stamps: pd.Series, schedule: pd.DataFrame
) -> tuple[pd.Series, pd.Series]:
    """Week a snapshot informs, and whether that week is regular season.

    A depth chart describes the next game to be played, so a snapshot is placed
    against the first regular-season week that kicks off at or after it. Anything
    taken before week 1 therefore lands on week 1, which is what the historical
    weekly feed also published preseason, keeping point-in-time cutoffs intact.
    """
    missing = pd.Series(pd.NA, index=stamps.index, dtype="Float64")
    if schedule is None or schedule.empty or "gameday" not in schedule.columns:
        return missing, pd.Series(pd.NA, index=stamps.index, dtype="string")
    regular = schedule[schedule.get("game_type").eq("REG")] if "game_type" in schedule else schedule
    if regular.empty:
        return missing, pd.Series(pd.NA, index=stamps.index, dtype="string")
    kickoff = (
        pd.to_datetime(regular["gameday"], errors="coerce")
        .groupby(pd.to_numeric(regular["week"], errors="coerce"))
        .min()
        .dropna()
        .sort_index()
    )
    if kickoff.empty:
        return missing, pd.Series(pd.NA, index=stamps.index, dtype="string")
    weeks = kickoff.index.to_numpy(dtype=float)
    starts = kickoff.to_numpy(dtype="datetime64[ns]")
    values = stamps.to_numpy(dtype="datetime64[ns]")
    position = np.searchsorted(starts, values, side="left")
    # Snapshots taken after the final regular-season kickoff describe postseason
    # depth. They are clipped onto the last week but flagged so the regular-season
    # filter still excludes them.
    postseason = position >= len(weeks)
    week = pd.Series(weeks[np.clip(position, 0, len(weeks) - 1)], index=stamps.index)
    week[stamps.isna()] = np.nan
    game_type = pd.Series(
        np.where(postseason, "POST", "REG"), index=stamps.index, dtype="object"
    )
    game_type[stamps.isna()] = pd.NA
    return week, game_type


def load_injuries(seasons: Iterable[int], refresh: bool = False, cache_dir=None):
    """Historical weekly injuries. The upstream feed currently ends in 2024."""
    return _by_season("injuries", seasons, refresh=refresh, cache_dir=cache_dir)


def load_schedules(seasons: Iterable[int], refresh: bool = False, cache_dir=None):
    """Schedules/results with lines, totals, rest, roof, weather, and coaches."""
    return _by_season("schedules", seasons, refresh=refresh, cache_dir=cache_dir)


def load_rosters(seasons: Iterable[int], refresh: bool = False, cache_dir=None):
    return _by_season("rosters", seasons, refresh=refresh, cache_dir=cache_dir)


def load_weekly_rosters(seasons: Iterable[int], refresh: bool = False, cache_dir=None):
    return _by_season("rosters_weekly", seasons, refresh=refresh, cache_dir=cache_dir)


def load_nextgen_stats(
    seasons: Iterable[int], stat_type: str, refresh: bool = False, cache_dir=None
):
    return _by_season(
        "nextgen_stats",
        seasons,
        params={"stat_type": stat_type},
        refresh=refresh,
        cache_dir=cache_dir,
    )


def load_pfr_advstats(
    seasons: Iterable[int],
    stat_type: str,
    summary_level: str = "week",
    refresh: bool = False,
    cache_dir=None,
):
    return _by_season(
        "pfr_advstats",
        seasons,
        params={"stat_type": stat_type, "summary_level": summary_level},
        refresh=refresh,
        cache_dir=cache_dir,
    )


def load_participation(seasons: Iterable[int], refresh: bool = False, cache_dir=None):
    return _by_season("participation", seasons, refresh=refresh, cache_dir=cache_dir)


def load_ftn_charting(seasons: Iterable[int], refresh: bool = False, cache_dir=None):
    return _by_season("ftn_charting", seasons, refresh=refresh, cache_dir=cache_dir)


def load_draft_picks(seasons: Iterable[int], refresh: bool = False, cache_dir=None):
    return _by_season("draft_picks", seasons, refresh=refresh, cache_dir=cache_dir)


def load_combine(seasons: Iterable[int], refresh: bool = False, cache_dir=None):
    return _by_season("combine", seasons, refresh=refresh, cache_dir=cache_dir)


def _static(dataset: str, *, refresh: bool = False, cache_dir=None, params=None):
    return _fetch(
        dataset,
        lambda: nflverse.load(dataset, **(params or {})),
        params=params,
        refresh=refresh,
        cache_dir=cache_dir,
    )


def load_ids(refresh: bool = False, cache_dir=None):
    """Comprehensive player dimension; GSIS ID is the canonical NFL key."""
    return _static("players", refresh=refresh, cache_dir=cache_dir)


def load_ff_playerids(refresh: bool = False, cache_dir=None):
    return _static("ff_playerids", refresh=refresh, cache_dir=cache_dir)


def load_contracts(refresh: bool = False, cache_dir=None):
    return _static("contracts", refresh=refresh, cache_dir=cache_dir)


def _daily_snapshot(snapshot_at: str | date | datetime | None) -> str:
    if snapshot_at is not None:
        return snapshot_at.isoformat() if hasattr(snapshot_at, "isoformat") else str(snapshot_at)
    return datetime.now(timezone.utc).date().isoformat()


def load_ff_rankings(
    ranking_type: str = "draft",
    *,
    snapshot_at: str | date | datetime | None = None,
    refresh: bool = False,
    cache_dir=None,
):
    """FantasyPros/DynastyProcess rankings saved as an immutable daily snapshot."""
    params = {"type": ranking_type}
    return _fetch(
        "ff_rankings",
        lambda: nflverse.load("ff_rankings", **params),
        params=params,
        as_of=_daily_snapshot(snapshot_at),
        refresh=refresh,
        cache_dir=cache_dir,
    )


def load_ff_opportunity(
    seasons: Iterable[int],
    stat_type: str = "weekly",
    model_version: str = "latest",
    refresh: bool = False,
    cache_dir=None,
):
    params = {"stat_type": stat_type, "model_version": model_version}
    return _by_season(
        "ff_opportunity",
        seasons,
        params=params,
        refresh=refresh,
        cache_dir=cache_dir,
    )
