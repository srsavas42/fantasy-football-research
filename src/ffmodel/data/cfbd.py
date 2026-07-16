"""CollegeFootballData pulls for prospect production and identity features."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from ffmodel.config import (
    CACHE_DIR,
    CFBD_API_KEY_ENV,
    CFBD_MONTHLY_LIMIT_ENV,
    project_env_value,
)
from ffmodel.data import cache
from ffmodel.data.http import get_json, records_frame

BASE_URL = "https://api.collegefootballdata.com"
SOURCE_URL = "https://collegefootballdata.com/"


class CfbdConfigurationError(RuntimeError):
    """The required CollegeFootballData credential was not configured."""


class CfbdQuotaError(RuntimeError):
    """A request was blocked by the local monthly safety limit."""


def _api_key(api_key: str | None) -> str:
    value = api_key or project_env_value(CFBD_API_KEY_ENV)
    if not value:
        raise CfbdConfigurationError(
            f"Set {CFBD_API_KEY_ENV} to a key from https://collegefootballdata.com/key"
        )
    return value


def _monthly_limit() -> int:
    raw = project_env_value(CFBD_MONTHLY_LIMIT_ENV) or "1000"
    limit = int(raw)
    if limit <= 0:
        raise CfbdConfigurationError(f"{CFBD_MONTHLY_LIMIT_ENV} must be positive")
    return limit


def _request_log_path(cache_dir: Path | None = None) -> Path:
    root = Path(cache_dir) if cache_dir is not None else CACHE_DIR
    return root / "raw" / "cfbd" / "request_log.jsonl"


def _request_records(cache_dir: Path | None = None) -> list[dict[str, Any]]:
    path = _request_log_path(cache_dir)
    if not path.exists():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


def local_request_budget(
    cache_dir: Path | None = None,
    *,
    now: datetime | None = None,
) -> dict[str, int | str]:
    """Report calls reserved locally in the current UTC calendar month."""
    now = now or datetime.now(timezone.utc)
    month = now.strftime("%Y-%m")
    used = sum(
        str(record.get("requested_at", "")).startswith(month)
        for record in _request_records(cache_dir)
    )
    limit = _monthly_limit()
    return {
        "month": month,
        "local_used": used,
        "local_limit": limit,
        "local_remaining": max(0, limit - used),
    }


def _reserve_request(
    dataset: str,
    endpoint: str,
    year: int | None,
    params: dict[str, Any],
    cache_dir: Path | None,
) -> None:
    budget = local_request_budget(cache_dir)
    if int(budget["local_remaining"]) <= 0:
        raise CfbdQuotaError(
            f"local CFBD safety limit reached for {budget['month']}: "
            f"{budget['local_used']}/{budget['local_limit']} calls"
        )
    path = _request_log_path(cache_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "requested_at": datetime.now(timezone.utc).isoformat(),
        "dataset": dataset,
        "endpoint": endpoint,
        "year": year,
        "params": params,
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def load_account_info(*, api_key: str | None = None) -> dict[str, Any]:
    """Return authoritative account/quota information from the no-cost info endpoint."""
    payload = get_json(
        f"{BASE_URL}/info",
        headers={"Authorization": f"Bearer {_api_key(api_key)}"},
    )
    if not isinstance(payload, dict):
        raise RuntimeError("CFBD /info returned an unexpected response")
    return payload


def account_quota(*, api_key: str | None = None) -> dict[str, Any]:
    """Return only non-sensitive quota fields from the authoritative account info."""
    payload = load_account_info(api_key=api_key)
    fields = (
        "tierName",
        "monthlyLimit",
        "usedCalls",
        "remainingCalls",
        "resetAt",
        "sharedPool",
    )
    return {field: payload.get(field) for field in fields if field in payload}


def _load(
    dataset: str,
    endpoint: str,
    *,
    year: int | None,
    params: dict[str, Any],
    api_key: str | None,
    refresh: bool,
    cache_dir: Path | None,
) -> pd.DataFrame:
    request_params = {key: value for key, value in params.items() if value is not None}

    def fetch() -> pd.DataFrame:
        _reserve_request(dataset, endpoint, year, request_params, cache_dir)
        payload = get_json(
            f"{BASE_URL}{endpoint}",
            params=request_params,
            headers={"Authorization": f"Bearer {_api_key(api_key)}"},
        )
        frame = records_frame(payload)
        frame["source"] = "cfbd"
        return frame

    # Validate before returning a pre-existing cache only when the caller
    # explicitly supplied a key. Cached artifacts remain usable offline.
    if api_key is not None:
        _api_key(api_key)
    return cache.get_or_fetch(
        dataset,
        fetch,
        season=year,
        refresh=refresh,
        cache_dir=cache_dir,
        provider="cfbd",
        params=request_params,
        source_url=f"{BASE_URL}{endpoint}",
        license_name="CollegeFootballData API terms",
    )


_DEFAULT_DATASETS = {
    "stats": ("player_season_stats", lambda year: {"year": year}),
    "usage": (
        "player_usage",
        lambda year: {"year": year, "excludeGarbageTime": "false"},
    ),
    "roster": ("roster", lambda year: {"year": year}),
    "recruits": ("recruits", lambda year: {"year": year}),
    "draft": ("draft_picks", lambda year: {"year": year}),
}


def default_request_is_cached(
    dataset: str,
    year: int,
    *,
    cache_dir: Path | None = None,
) -> bool:
    """Return whether the CLI's unfiltered request already exists locally."""
    cache_dataset, params_factory = _DEFAULT_DATASETS[dataset]
    path = cache.cache_path(
        cache_dataset,
        year,
        cache_dir,
        provider="cfbd",
        params=params_factory(year),
    )
    return path.exists()


def load_player_season_stats(
    year: int,
    *,
    team: str | None = None,
    conference: str | None = None,
    start_week: int | None = None,
    end_week: int | None = None,
    api_key: str | None = None,
    refresh: bool = False,
    cache_dir: Path | None = None,
) -> pd.DataFrame:
    return _load(
        "player_season_stats",
        "/stats/player/season",
        year=year,
        params={
            "year": year,
            "team": team,
            "conference": conference,
            "startWeek": start_week,
            "endWeek": end_week,
        },
        api_key=api_key,
        refresh=refresh,
        cache_dir=cache_dir,
    )


def load_player_usage(
    year: int,
    *,
    team: str | None = None,
    conference: str | None = None,
    position: str | None = None,
    player_id: int | None = None,
    exclude_garbage_time: bool = False,
    api_key: str | None = None,
    refresh: bool = False,
    cache_dir: Path | None = None,
) -> pd.DataFrame:
    return _load(
        "player_usage",
        "/player/usage",
        year=year,
        params={
            "year": year,
            "team": team,
            "conference": conference,
            "position": position,
            "playerId": player_id,
            "excludeGarbageTime": str(exclude_garbage_time).lower(),
        },
        api_key=api_key,
        refresh=refresh,
        cache_dir=cache_dir,
    )


def load_roster(
    year: int,
    *,
    team: str | None = None,
    api_key: str | None = None,
    refresh: bool = False,
    cache_dir: Path | None = None,
) -> pd.DataFrame:
    return _load(
        "roster",
        "/roster",
        year=year,
        params={"year": year, "team": team},
        api_key=api_key,
        refresh=refresh,
        cache_dir=cache_dir,
    )


def load_recruits(
    year: int,
    *,
    team: str | None = None,
    position: str | None = None,
    api_key: str | None = None,
    refresh: bool = False,
    cache_dir: Path | None = None,
) -> pd.DataFrame:
    return _load(
        "recruits",
        "/recruiting/players",
        year=year,
        params={"year": year, "team": team, "position": position},
        api_key=api_key,
        refresh=refresh,
        cache_dir=cache_dir,
    )


def load_draft_picks(
    year: int,
    *,
    api_key: str | None = None,
    refresh: bool = False,
    cache_dir: Path | None = None,
) -> pd.DataFrame:
    return _load(
        "draft_picks",
        "/draft/picks",
        year=year,
        params={"year": year},
        api_key=api_key,
        refresh=refresh,
        cache_dir=cache_dir,
    )
