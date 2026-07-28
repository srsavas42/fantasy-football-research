"""Registry and configuration checks for every supported data source."""

from __future__ import annotations

import importlib.util
from dataclasses import asdict, dataclass

import pandas as pd

from ffmodel.config import CFBD_API_KEY_ENV, ODDS_API_KEY_ENV, project_env_value


@dataclass(frozen=True)
class SourceSpec:
    name: str
    purpose: str
    credential_env: str | None
    dependency: str | None
    historical: bool
    point_in_time: bool
    license_note: str


SOURCES = {
    "nflverse": SourceSpec(
        "nflverse",
        "NFL stats, PBP, snaps, rosters, schedules, IDs, rankings",
        None,
        "nflreadpy",
        True,
        False,
        "Mostly CC-BY; check each upstream dataset, especially FTN charting.",
    ),
    "sleeper": SourceSpec(
        "sleeper",
        "Current NFL injuries, roster status, and depth order",
        None,
        None,
        False,
        True,
        "Free read-only API; cache the players endpoint at most daily.",
    ),
    "cfbd": SourceSpec(
        "cfbd",
        "College production, usage, rosters, recruiting, and draft links",
        CFBD_API_KEY_ENV,
        None,
        True,
        False,
        "Subject to CollegeFootballData API terms and plan limits.",
    ),
    "odds": SourceSpec(
        "odds",
        "Point-in-time NFL spreads and totals",
        ODDS_API_KEY_ENV,
        None,
        True,
        True,
        "The Odds API key and historical-snapshot plan are user supplied.",
    ),
    "open_meteo": SourceSpec(
        "open_meteo",
        "Current and archived hourly weather forecasts",
        None,
        None,
        True,
        True,
        "CC-BY for non-commercial use; review commercial-use terms.",
    ),
}


def source_status() -> pd.DataFrame:
    """Return configuration status without making network requests."""
    rows = []
    for spec in SOURCES.values():
        dependency_ok = spec.dependency is None or importlib.util.find_spec(spec.dependency) is not None
        credential_ok = spec.credential_env is None or bool(
            project_env_value(spec.credential_env)
        )
        rows.append(
            {
                **asdict(spec),
                "dependency_ok": dependency_ok,
                "credential_ok": credential_ok,
                "configured": dependency_ok and credential_ok,
            }
        )
    return pd.DataFrame(rows)
