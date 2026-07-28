"""Hourly current or historical forecasts from Open-Meteo."""

from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

from ffmodel.data import cache
from ffmodel.data.http import get_json

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
HISTORICAL_FORECAST_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"
PREVIOUS_RUNS_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
SOURCE_URL = "https://open-meteo.com/en/docs/historical-forecast-api"

DEFAULT_VARIABLES = (
    "temperature_2m",
    "precipitation_probability",
    "precipitation",
    "wind_speed_10m",
    "wind_gusts_10m",
)


def _hourly_frame(
    payload: dict,
    latitude: float,
    longitude: float,
    observed_at: str,
    *,
    suffix: str = "",
):
    hourly = payload.get("hourly", {})
    times = hourly.get("time", [])
    rows = []
    for index, timestamp in enumerate(times):
        row = {
            "forecast_time": timestamp,
            "latitude": latitude,
            "longitude": longitude,
            "observed_at": observed_at,
            "source": "open_meteo",
        }
        for name, values in hourly.items():
            if name != "time" and isinstance(values, list):
                clean_name = name.removesuffix(suffix) if suffix else name
                row[clean_name] = values[index] if index < len(values) else None
        rows.append(row)
    return pd.DataFrame(rows)


def load_hourly_forecast(
    latitude: float,
    longitude: float,
    start_date: str | date,
    end_date: str | date,
    *,
    variables: tuple[str, ...] = DEFAULT_VARIABLES,
    historical: bool = False,
    snapshot_at: str | datetime | None = None,
    timezone_name: str = "UTC",
    refresh: bool = False,
    cache_dir: Path | None = None,
) -> pd.DataFrame:
    """Pull hourly weather and retain the forecast acquisition timestamp."""
    start = start_date.isoformat() if hasattr(start_date, "isoformat") else str(start_date)
    end = end_date.isoformat() if hasattr(end_date, "isoformat") else str(end_date)
    now = datetime.now(timezone.utc)
    observed_at = now.isoformat(timespec="minutes")
    if snapshot_at is not None:
        requested = snapshot_at.isoformat() if hasattr(snapshot_at, "isoformat") else str(snapshot_at)
        if requested[:10] != now.date().isoformat():
            raise ValueError(
                "Live/historical weather retrieval cannot be labeled as a past "
                "acquisition. Use load_previous_run_forecast for backtests."
            )
        observed_at = requested
    params = {
        "latitude": round(float(latitude), 5),
        "longitude": round(float(longitude), 5),
        "start_date": start,
        "end_date": end,
        "hourly": ",".join(variables),
        "timezone": timezone_name,
        "historical": historical,
    }
    endpoint = HISTORICAL_FORECAST_URL if historical else FORECAST_URL

    def fetch() -> pd.DataFrame:
        request_params = {key: value for key, value in params.items() if key != "historical"}
        payload = get_json(endpoint, params=request_params)
        return _hourly_frame(payload, float(latitude), float(longitude), observed_at)

    return cache.get_or_fetch(
        "hourly_forecast",
        fetch,
        refresh=refresh,
        cache_dir=cache_dir,
        provider="open_meteo",
        params=params,
        as_of=observed_at,
        source_url=SOURCE_URL,
        license_name="Open-Meteo CC-BY-4.0; commercial terms may apply",
    )


def load_previous_run_forecast(
    latitude: float,
    longitude: float,
    start_date: str | date,
    end_date: str | date,
    *,
    lead_days: int,
    variables: tuple[str, ...] = (
        "temperature_2m",
        "precipitation",
        "wind_speed_10m",
    ),
    timezone_name: str = "UTC",
    refresh: bool = False,
    cache_dir: Path | None = None,
) -> pd.DataFrame:
    """Load forecasts made exactly ``lead_days`` before their valid time.

    Open-Meteo archives fixed offsets from 1 through 7 days. These rows expose
    ``available_at`` so a backtest can enforce its prediction cutoff.
    """
    if lead_days not in range(1, 8):
        raise ValueError("lead_days must be between 1 and 7")
    start = start_date.isoformat() if hasattr(start_date, "isoformat") else str(start_date)
    end = end_date.isoformat() if hasattr(end_date, "isoformat") else str(end_date)
    suffix = f"_previous_day{lead_days}"
    archived_variables = tuple(f"{name}{suffix}" for name in variables)
    params = {
        "latitude": round(float(latitude), 5),
        "longitude": round(float(longitude), 5),
        "start_date": start,
        "end_date": end,
        "hourly": ",".join(archived_variables),
        "timezone": timezone_name,
        "lead_days": lead_days,
    }

    def fetch() -> pd.DataFrame:
        request_params = {key: value for key, value in params.items() if key != "lead_days"}
        payload = get_json(PREVIOUS_RUNS_URL, params=request_params)
        frame = _hourly_frame(
            payload,
            float(latitude),
            float(longitude),
            datetime.now(timezone.utc).isoformat(timespec="minutes"),
            suffix=suffix,
        )
        valid_time = pd.to_datetime(frame["forecast_time"], utc=True)
        frame["available_at"] = valid_time - pd.Timedelta(days=lead_days)
        frame["lead_days"] = lead_days
        frame["weather_data_kind"] = "previous_run"
        return frame

    return cache.get_or_fetch(
        "previous_run_forecast",
        fetch,
        refresh=refresh,
        cache_dir=cache_dir,
        provider="open_meteo",
        params=params,
        source_url="https://open-meteo.com/en/docs/previous-runs-api",
        license_name="Open-Meteo CC-BY-4.0; commercial terms may apply",
    )
