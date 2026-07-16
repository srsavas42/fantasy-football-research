"""Unified entry point: one call, one schema, whatever source is reachable."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from ffmodel.data import ingest, legacy
from ffmodel.data.schema import conform


def load_player_weeks(
    seasons: Iterable[int],
    source: str = "auto",
    refresh: bool = False,
    cache_dir: Path | None = None,
) -> pd.DataFrame:
    """Player-week stat lines for the given seasons in the canonical schema.

    source:
      "auto"     - nflverse first (richer: player ids, real targets), falling
                   back to the committed legacy CSVs per season on failure.
      "nflverse" - nflverse only; raises DataUnavailableError if unreachable.
      "legacy"   - committed CSVs only (1999-2021).
    """
    seasons = list(seasons)
    if source == "nflverse":
        return ingest.load_weekly(seasons, refresh=refresh, cache_dir=cache_dir)
    if source == "legacy":
        return legacy.load_weekly(seasons)
    if source != "auto":
        raise ValueError(f"unknown source {source!r}")

    frames = []
    for season in seasons:
        try:
            frames.append(ingest.load_weekly([season], refresh=refresh, cache_dir=cache_dir))
        except ingest.DataUnavailableError:
            fallback = legacy.load_weekly([season])
            if fallback.empty:
                raise
            frames.append(fallback)
    return conform(pd.concat(frames, ignore_index=True))
