"""Current and historical point-in-time NFL odds from The Odds API."""

from __future__ import annotations

import os
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from ffmodel.config import ODDS_API_KEY_ENV
from ffmodel.data import cache
from ffmodel.data.http import get_json

BASE_URL = "https://api.the-odds-api.com/v4"
SOURCE_URL = "https://the-odds-api.com/liveapi/guides/v4/"


class OddsConfigurationError(RuntimeError):
    """The Odds API credential was not configured."""


def _api_key(api_key: str | None) -> str:
    value = api_key or os.environ.get(ODDS_API_KEY_ENV)
    if not value:
        raise OddsConfigurationError(
            f"Set {ODDS_API_KEY_ENV} to a key from https://the-odds-api.com/"
        )
    return value


def _as_of(value: str | date | datetime | None) -> str:
    if value is None:
        return datetime.now(timezone.utc).isoformat(timespec="minutes")
    return value.isoformat() if hasattr(value, "isoformat") else str(value)


def _events_frame(events: list[dict[str, Any]], observed_at: str) -> pd.DataFrame:
    rows = []
    for event in events:
        for bookmaker in event.get("bookmakers", []):
            for market in bookmaker.get("markets", []):
                for outcome in market.get("outcomes", []):
                    rows.append(
                        {
                            "event_id": event.get("id"),
                            "sport_key": event.get("sport_key"),
                            "commence_time": event.get("commence_time"),
                            "home_team": event.get("home_team"),
                            "away_team": event.get("away_team"),
                            "bookmaker": bookmaker.get("key"),
                            "bookmaker_title": bookmaker.get("title"),
                            "market": market.get("key"),
                            "market_last_update": market.get("last_update"),
                            "outcome": outcome.get("name"),
                            "price": outcome.get("price"),
                            "point": outcome.get("point"),
                            "observed_at": observed_at,
                            "source": "the_odds_api",
                        }
                    )
    return pd.DataFrame(rows)


def load_nfl_odds(
    *,
    regions: str = "us",
    markets: str = "spreads,totals",
    odds_format: str = "american",
    historical_at: str | datetime | None = None,
    snapshot_at: str | date | datetime | None = None,
    api_key: str | None = None,
    refresh: bool = False,
    cache_dir: Path | None = None,
) -> pd.DataFrame:
    """NFL odds in long form, one row per book/market/outcome.

    ``historical_at`` requires The Odds API's historical-data plan and should
    equal the model's prediction cutoff. The API key is never written to cache
    paths or manifests.
    """
    observed_at = _as_of(historical_at or snapshot_at)
    public_params = {
        "regions": regions,
        "markets": markets,
        "oddsFormat": odds_format,
        "historical_at": _as_of(historical_at) if historical_at is not None else None,
    }
    endpoint = "/historical/sports/americanfootball_nfl/odds" if historical_at else "/sports/americanfootball_nfl/odds"

    if historical_at is None and snapshot_at is not None:
        today = datetime.now(timezone.utc).date().isoformat()
        requested_day = observed_at[:10]
        path = cache.cache_path(
            "nfl_odds",
            cache_dir=cache_dir,
            provider="the_odds_api",
            params=public_params,
            as_of=observed_at,
        )
        if requested_day != today and (refresh or not path.exists()):
            raise ValueError(
                "Current odds cannot be backdated. Omit snapshot_at when fetching "
                "live odds, or use historical_at with a historical-data plan."
            )

    def fetch() -> pd.DataFrame:
        request_params = {
            "apiKey": _api_key(api_key),
            "regions": regions,
            "markets": markets,
            "oddsFormat": odds_format,
        }
        if historical_at is not None:
            request_params["date"] = _as_of(historical_at)
        payload = get_json(f"{BASE_URL}{endpoint}", params=request_params)
        events = payload.get("data", []) if isinstance(payload, dict) else payload
        return _events_frame(events or [], observed_at)

    if api_key is not None:
        _api_key(api_key)
    return cache.get_or_fetch(
        "nfl_odds",
        fetch,
        refresh=refresh,
        cache_dir=cache_dir,
        provider="the_odds_api",
        params=public_params,
        as_of=observed_at,
        source_url=SOURCE_URL,
        license_name="The Odds API terms",
    )
