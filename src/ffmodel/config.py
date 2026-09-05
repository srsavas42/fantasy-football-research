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
# Relative Athletic Score, if supplied. Not fetched: RAS is published by a third
# party and is not part of nflverse, so it is dropped in by hand and overrides
# the combine-derived composite wherever it is present.
RAS_SCORES_PATH = MANUAL_DATA_DIR / "ras_scores.csv"
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
# Informational only: the last season the injury feed was observed to publish.
# It no longer gates which seasons are requested, because a hardcoded ceiling
# silently drops a season the moment the feed extends past it — as it did for
# 2025. The loader skips seasons the feed declines instead, so coverage follows
# the data. Live projections still supply an archived current snapshot, since
# the report for an unplayed season does not exist at any ceiling.
NFLVERSE_INJURY_LAST_SEASON = 2025

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


@dataclass(frozen=True)
class KickerRules:
    """Point weights for a kicker's week.

    Kicker scoring is distance-tiered in every mainstream league, and the tiers
    are the whole reason kickers are not interchangeable: a leg that converts
    from 50 is worth two points more per make than one that is only trusted
    inside 40. nflverse publishes makes and misses already bucketed by distance
    (``fg_made_0_19`` through ``fg_made_60_``), so the tiers are read directly
    rather than inferred from a total and an average.

    Defaults are the common ESPN configuration. A league that scores differently
    changes these numbers and nothing else -- the model is fitted on whatever
    ``kicker_points`` returns, so the tiers are a scoring convention rather than
    a modelling assumption.
    """

    fg_0_39: float = 3.0
    fg_40_49: float = 4.0
    fg_50_plus: float = 5.0
    fg_miss: float = -1.0
    pat_made: float = 1.0
    pat_miss: float = -1.0


@dataclass(frozen=True)
class DefenseRules:
    """Point weights for a team defense/special-teams week.

    Two halves that behave very differently. The event half -- sacks, takeaways,
    touchdowns -- is close to a count of independent good things. The
    points-allowed half is a **step function of the opponent's final score**,
    which means most of a DST's fantasy week is decided by how good the other
    offence is and how the game script ran, not by anything the defence's own
    box score records. Any model of this response has to project the opponent's
    scoring, which is why the schedule's implied totals matter more here than
    anywhere else in the package.

    ``points_allowed_tiers`` is read as ordered ``(upper_bound, points)`` pairs,
    inclusive on the bound and evaluated in order, with
    ``points_allowed_worst`` for anything above the last bound. Defaults are the
    common ESPN configuration.
    """

    sack: float = 1.0
    interception: float = 2.0
    fumble_recovery: float = 2.0
    touchdown: float = 6.0
    safety: float = 2.0
    block: float = 2.0
    points_allowed_tiers: tuple[tuple[int, float], ...] = (
        (0, 10.0),
        (6, 7.0),
        (13, 4.0),
        (20, 1.0),
        (27, 0.0),
        (34, -1.0),
    )
    points_allowed_worst: float = -4.0


KICKER_STANDARD = KickerRules()
DEFENSE_STANDARD = DefenseRules()

KICKER_FORMATS = {"standard": KICKER_STANDARD}
DEFENSE_FORMATS = {"standard": DEFENSE_STANDARD}
