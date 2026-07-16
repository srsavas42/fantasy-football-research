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
    df = _map_weekly_aliases(raw)
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
    """Listed depth charts; use as a cold-start fallback, not ground truth."""
    return _by_season("depth_charts", seasons, refresh=refresh, cache_dir=cache_dir)


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
