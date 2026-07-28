"""Thin pandas adapter around the maintained :mod:`nflreadpy` client."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

import pandas as pd


class NflverseProviderError(RuntimeError):
    """The nflverse dependency or a remote nflverse artifact was unavailable."""


@dataclass(frozen=True)
class DatasetSpec:
    function: str
    accepts_seasons: bool = True
    defaults: dict[str, Any] = field(default_factory=dict)


DATASETS: dict[str, DatasetSpec] = {
    "pbp": DatasetSpec("load_pbp"),
    "player_stats": DatasetSpec("load_player_stats", defaults={"summary_level": "week"}),
    "team_stats": DatasetSpec("load_team_stats", defaults={"summary_level": "week"}),
    "schedules": DatasetSpec("load_schedules"),
    "teams": DatasetSpec("load_teams", accepts_seasons=False),
    "players": DatasetSpec("load_players", accepts_seasons=False),
    "rosters": DatasetSpec("load_rosters"),
    "rosters_weekly": DatasetSpec("load_rosters_weekly"),
    "snap_counts": DatasetSpec("load_snap_counts"),
    "nextgen_stats": DatasetSpec("load_nextgen_stats", defaults={"stat_type": "receiving"}),
    "ftn_charting": DatasetSpec("load_ftn_charting"),
    "participation": DatasetSpec("load_participation"),
    "draft_picks": DatasetSpec("load_draft_picks"),
    "injuries": DatasetSpec("load_injuries"),
    "contracts": DatasetSpec("load_contracts", accepts_seasons=False),
    "combine": DatasetSpec("load_combine"),
    "depth_charts": DatasetSpec("load_depth_charts"),
    "pfr_advstats": DatasetSpec(
        "load_pfr_advstats", defaults={"stat_type": "rec", "summary_level": "week"}
    ),
    "ff_playerids": DatasetSpec("load_ff_playerids", accepts_seasons=False),
    "ff_rankings": DatasetSpec(
        "load_ff_rankings", accepts_seasons=False, defaults={"type": "draft"}
    ),
    "ff_opportunity": DatasetSpec(
        "load_ff_opportunity",
        defaults={"stat_type": "weekly", "model_version": "latest"},
    ),
}


def _client():
    try:
        import nflreadpy
    except ImportError as exc:
        raise NflverseProviderError(
            'nflreadpy is not installed; run `pip install -e ".[dev]"`.'
        ) from exc
    return nflreadpy


def _to_pandas(frame: Any) -> pd.DataFrame:
    if isinstance(frame, pd.DataFrame):
        return frame.copy()
    if hasattr(frame, "to_pandas"):
        return frame.to_pandas()
    raise NflverseProviderError(
        f"nflreadpy returned unsupported frame type {type(frame).__name__}"
    )


def load(
    dataset: str,
    seasons: int | Iterable[int] | None = None,
    **params: Any,
) -> pd.DataFrame:
    """Load one registered nflverse dataset and return a pandas frame."""
    if dataset not in DATASETS:
        raise ValueError(
            f"unknown nflverse dataset {dataset!r}; choose from {sorted(DATASETS)}"
        )
    spec = DATASETS[dataset]
    client = _client()
    function = getattr(client, spec.function)
    kwargs = {**spec.defaults, **params}
    if spec.accepts_seasons:
        if seasons is not None and not isinstance(seasons, int):
            seasons = list(seasons)
        kwargs["seasons"] = seasons
    elif seasons is not None:
        raise ValueError(f"nflverse dataset {dataset!r} does not accept seasons")
    try:
        return _to_pandas(function(**kwargs))
    except (TypeError, ValueError):
        # Preserve invalid-argument errors: they are programming/configuration
        # problems rather than remote availability failures.
        raise
    except Exception as exc:
        raise NflverseProviderError(
            f"nflreadpy failed to load {dataset!r}: {exc}"
        ) from exc
