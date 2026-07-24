"""Project-wide configuration: paths, seasons, and scoring rules."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# Repo root = two levels up from src/ffmodel/config.py
REPO_ROOT = Path(__file__).resolve().parents[2]

# Legacy CSV locations (committed to the repo, forked from fantasydatapros/data)
LEGACY_WEEKLY_DIR = REPO_ROOT / "weekly"
LEGACY_YEARLY_DIR = REPO_ROOT / "yearly"
LEGACY_SNAPCOUNTS_DIR = REPO_ROOT / "snapcounts"
LEGACY_ADP_DIR = REPO_ROOT / "fantasypros" / "adp"
LEGACY_ECR_DIR = REPO_ROOT / "fantasypros" / "ecr"
LEGACY_SOS_DIR = REPO_ROOT / "sos"
MANUAL_DATA_DIR = REPO_ROOT / "data" / "manual"
COACHING_PERIODS_PATH = MANUAL_DATA_DIR / "coach_team_period.csv"
IDENTITY_OVERRIDES_PATH = MANUAL_DATA_DIR / "player_identity_overrides.csv"
WIKIPEDIA_COACHING_DIR = REPO_ROOT / "data" / "coaching" / "wikipedia"

# Parquet cache for downloaded nflverse data. Override with FFMODEL_CACHE_DIR.
CACHE_DIR = Path(os.environ.get("FFMODEL_CACHE_DIR", REPO_ROOT / ".cache" / "ffmodel"))

# Remote-source settings. API keys are intentionally read inside each request
# function, not here, so tests and notebooks can set them after importing ffmodel.
HTTP_TIMEOUT = float(os.environ.get("FFMODEL_HTTP_TIMEOUT", "30"))
CFBD_API_KEY_ENV = "FFMODEL_CFBD_API_KEY"
ODDS_API_KEY_ENV = "FFMODEL_ODDS_API_KEY"
CFBD_MONTHLY_LIMIT_ENV = "FFMODEL_CFBD_MONTHLY_LIMIT"


def project_env_value(name: str) -> str | None:
    """Read a process variable, falling back to the Git-ignored project .env."""
    value = os.environ.get(name)
    if value:
        return value
    path = REPO_ROOT / ".env"
    if not path.exists():
        return None
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, candidate = line.removeprefix("export ").split("=", 1)
        if key.strip() == name:
            candidate = candidate.strip()
            if len(candidate) >= 2 and candidate[0] == candidate[-1] and candidate[0] in "\"'":
                candidate = candidate[1:-1]
            return candidate or None
    return None

# Season coverage
LEGACY_WEEKLY_SEASONS = range(1999, 2022)   # weekly/{year}/week{n}.csv
LEGACY_YEARLY_SEASONS = range(1970, 2022)   # yearly/{year}.csv
NFLVERSE_FIRST_SEASON = 1999                # pbp / weekly player stats
NFLVERSE_SNAPS_FIRST_SEASON = 2012          # snap counts
NFLVERSE_DEPTH_FIRST_SEASON = 2001          # depth charts
NFLVERSE_INJURY_FIRST_SEASON = 2009         # injury reports
# The historical nflverse injury-report feed is currently unavailable after
# 2024. Live projections should supply an archived current snapshot instead.
NFLVERSE_INJURY_LAST_SEASON = 2024

# Weeks: 17 games through 2020, 18 from 2021 on
def regular_season_weeks(season: int) -> int:
    return 17 if season <= 2020 else 18


@dataclass(frozen=True)
class ScoringRules:
    """Point weights for converting a stat line to fantasy points.

    Defaults verified to reproduce the legacy CSVs' StandardFantasyPoints
    exactly; reception is 0 / 0.5 / 1.0 for standard / half-PPR / PPR.
    """

    pass_yd: float = 0.04
    pass_td: float = 4.0
    interception: float = -2.0
    rush_yd: float = 0.1
    rush_td: float = 6.0
    rec_yd: float = 0.1
    rec_td: float = 6.0
    reception: float = 0.0
    fumble_lost: float = -2.0


STANDARD = ScoringRules()
HALF_PPR = ScoringRules(reception=0.5)
PPR = ScoringRules(reception=1.0)

SCORING_FORMATS = {"standard": STANDARD, "half_ppr": HALF_PPR, "ppr": PPR}
