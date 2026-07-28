"""Small standard-library HTTP client shared by non-nflverse providers."""

from __future__ import annotations

import json
from typing import Any, Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from ffmodel.config import HTTP_TIMEOUT

import pandas as pd


class RemoteDataError(RuntimeError):
    """A remote provider request failed or returned an invalid response."""


def records_frame(payload: Any) -> pd.DataFrame:
    """Normalize a JSON list/object and serialize remaining nested cells."""
    if payload is None:
        return pd.DataFrame()
    records = payload if isinstance(payload, list) else [payload]
    frame = pd.json_normalize(records, sep="__")
    for column in frame.columns:
        if frame[column].map(lambda value: isinstance(value, (dict, list))).any():
            frame[column] = frame[column].map(
                lambda value: json.dumps(value, sort_keys=True)
                if isinstance(value, (dict, list))
                else value
            )
    return frame


def get_json(
    url: str,
    *,
    params: Mapping[str, Any] | None = None,
    headers: Mapping[str, str] | None = None,
    timeout: float = HTTP_TIMEOUT,
) -> Any:
    """GET JSON with consistent errors and a project-specific user agent."""
    clean_params = {
        key: value
        for key, value in (params or {}).items()
        if value is not None
    }
    if clean_params:
        url = f"{url}?{urlencode(clean_params, doseq=True)}"
    request_headers = {
        "Accept": "application/json",
        "User-Agent": "ffmodel/0.1 (+https://github.com/srsavas42/fantasy-football-models)",
        **(headers or {}),
    }
    request = Request(url, headers=request_headers)
    try:
        with urlopen(request, timeout=timeout) as response:
            charset = response.headers.get_content_charset() or "utf-8"
            return json.loads(response.read().decode(charset))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        raise RemoteDataError(
            f"GET {exc.url} returned HTTP {exc.code}: {detail}"
        ) from exc
    except (URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise RemoteDataError(f"GET {url} failed: {exc}") from exc
