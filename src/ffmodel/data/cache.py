"""Local parquet cache so nflverse downloads happen once per dataset/season."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import pandas as pd

from ffmodel.config import CACHE_DIR


def cache_path(dataset: str, season: int | None = None, cache_dir: Path | None = None) -> Path:
    root = Path(cache_dir) if cache_dir is not None else CACHE_DIR
    name = f"{dataset}_{season}.parquet" if season is not None else f"{dataset}.parquet"
    return root / name


def get_or_fetch(
    dataset: str,
    fetch: Callable[[], pd.DataFrame],
    season: int | None = None,
    refresh: bool = False,
    cache_dir: Path | None = None,
) -> pd.DataFrame:
    """Return the cached frame for (dataset, season), fetching and caching on miss."""
    path = cache_path(dataset, season, cache_dir)
    if path.exists() and not refresh:
        return pd.read_parquet(path)
    df = fetch()
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return df
