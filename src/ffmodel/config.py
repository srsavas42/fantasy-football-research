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

# Parquet cache for downloaded nflverse data. Override with FFMODEL_CACHE_DIR.
CACHE_DIR = Path(os.environ.get("FFMODEL_CACHE_DIR", REPO_ROOT / ".cache" / "ffmodel"))

# Season coverage
LEGACY_WEEKLY_SEASONS = range(1999, 2022)   # weekly/{year}/week{n}.csv
LEGACY_YEARLY_SEASONS = range(1970, 2022)   # yearly/{year}.csv
NFLVERSE_FIRST_SEASON = 1999                # pbp / weekly player stats
NFLVERSE_SNAPS_FIRST_SEASON = 2012          # snap counts
NFLVERSE_DEPTH_FIRST_SEASON = 2001          # depth charts
NFLVERSE_INJURY_FIRST_SEASON = 2009         # injury reports

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
