"""Point-in-time Sleeper player, injury, and depth-chart snapshots."""

from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

from ffmodel.data import cache
from ffmodel.data.http import get_json, records_frame

BASE_URL = "https://api.sleeper.app/v1"
SOURCE_URL = "https://docs.sleeper.com/"


def capture_players_payload() -> bytes:
    """Capture the current Sleeper roster response as deterministic raw JSON.

    Release code uses this only at the immutable source boundary; normal data
    loaders retain their existing cache-aware dataframe behavior.
    """
    payload = get_json(f"{BASE_URL}/players/nfl")
    if not isinstance(payload, dict):
        raise ValueError("Sleeper players response must be a JSON object")
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")


def _snapshot_day(value: str | date | datetime | None) -> str:
    if value is None:
        return datetime.now(timezone.utc).date().isoformat()
    text = value.isoformat() if hasattr(value, "isoformat") else str(value)
    return text[:10]


def _require_real_snapshot(
    dataset: str,
    as_of: str,
    *,
    params=None,
    refresh: bool,
    cache_dir: Path | None,
) -> None:
    """Never fetch today's Sleeper state into a past/future cache partition."""
    today = datetime.now(timezone.utc).date().isoformat()
    if as_of == today:
        return
    path = cache.cache_path(
        dataset, cache_dir=cache_dir, provider="sleeper", params=params, as_of=as_of
    )
    if refresh or not path.exists():
        raise ValueError(
            "Sleeper cannot retrieve historical snapshots. Omit snapshot_at to "
            "archive today's state, or read a snapshot that was archived earlier."
        )


def _player_frame(payload: dict) -> pd.DataFrame:
    rows = []
    for sleeper_id, player in payload.items():
        if not isinstance(player, dict):
            continue
        rows.append({"sleeper_id": sleeper_id, **player})
    frame = records_frame(rows)
    frame = frame.rename(
        columns={
            "full_name": "player_name",
            "player_id": "sleeper_player_id",
            "depth_chart_order": "depth_order",
        }
    )
    if "sleeper_player_id" not in frame:
        frame["sleeper_player_id"] = frame.get("sleeper_id")
    return frame


def load_players(
    *,
    snapshot_at: str | date | datetime | None = None,
    refresh: bool = False,
    cache_dir: Path | None = None,
) -> pd.DataFrame:
    """Current NFL players, injury status, practice status, and depth order.

    Sleeper has no historical-snapshot endpoint. This function stores one
    immutable cache artifact per UTC day; backtests can use only snapshots that
    were actually archived on or before their prediction cutoff.
    """
    as_of = _snapshot_day(snapshot_at)
    _require_real_snapshot(
        "players", as_of, refresh=refresh, cache_dir=cache_dir
    )

    def fetch() -> pd.DataFrame:
        frame = _player_frame(get_json(f"{BASE_URL}/players/nfl"))
        frame["observed_at"] = datetime.now(timezone.utc).isoformat()
        frame["source"] = "sleeper"
        return frame

    return cache.get_or_fetch(
        "players",
        fetch,
        refresh=refresh,
        cache_dir=cache_dir,
        provider="sleeper",
        as_of=as_of,
        source_url=SOURCE_URL,
        license_name="Sleeper API terms",
    )


def load_trending(
    kind: str = "add",
    *,
    lookback_hours: int = 24,
    limit: int = 100,
    snapshot_at: str | date | datetime | None = None,
    refresh: bool = False,
    cache_dir: Path | None = None,
) -> pd.DataFrame:
    """Most added or dropped players; a weak live signal, not training truth."""
    if kind not in {"add", "drop"}:
        raise ValueError("kind must be 'add' or 'drop'")
    params = {"lookback_hours": int(lookback_hours), "limit": int(limit), "kind": kind}
    as_of = _snapshot_day(snapshot_at)
    _require_real_snapshot(
        "trending", as_of, params=params, refresh=refresh, cache_dir=cache_dir
    )

    def fetch() -> pd.DataFrame:
        payload = get_json(
            f"{BASE_URL}/players/nfl/trending/{kind}",
            params={"lookback_hours": lookback_hours, "limit": limit},
        )
        frame = records_frame(payload)
        frame["observed_at"] = datetime.now(timezone.utc).isoformat()
        frame["source"] = "sleeper"
        return frame

    return cache.get_or_fetch(
        "trending",
        fetch,
        refresh=refresh,
        cache_dir=cache_dir,
        provider="sleeper",
        params=params,
        as_of=as_of,
        source_url=SOURCE_URL,
        license_name="Sleeper API terms",
    )
