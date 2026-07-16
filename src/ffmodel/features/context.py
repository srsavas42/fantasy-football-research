"""Team-week game-script context and the team-season model key.

Game script (how pass-happy a game is likely to be) is driven by the Vegas
total and spread, available per game from nflverse schedules. That feed is
github-hosted and may be unreachable offline, so `game_context` degrades
gracefully: real Vegas numbers when reachable, neutral defaults otherwise.

`team_season` is always available and is the key the Phase 3 team model uses
for its per-team-season "scheme" random effect — scheme is *learned* there,
not engineered here.
"""

from __future__ import annotations

import pandas as pd

from ffmodel.config import LEGACY_SOS_DIR
from ffmodel.data import ingest

CONTEXT_COLUMNS = ["team_season", "implied_team_total", "spread", "is_home", "rest_days"]

# Neutral-script defaults when no Vegas data is available.
_NEUTRAL = {"implied_team_total": 22.5, "spread": 0.0, "is_home": 0, "rest_days": 7}


def team_season_key(df: pd.DataFrame) -> pd.Series:
    return df["team"].astype(str) + "_" + df["season"].astype(str)


def game_context(seasons, refresh: bool = False, cache_dir=None) -> pd.DataFrame:
    """Per (season, week, team) game-script covariates.

    Returns one row per team per game with implied team total, spread (from that
    team's perspective), home flag, and rest days. Falls back to neutral values
    for every scheduled team-week if schedules can't be downloaded.
    """
    seasons = list(seasons)
    try:
        sched = ingest.load_schedules(seasons, refresh=refresh, cache_dir=cache_dir)
    except ingest.DataUnavailableError:
        return _neutral_frame(seasons)

    rows = []
    for _, g in sched.iterrows():
        total = g.get("total_line")
        spread = g.get("spread_line")  # home-team perspective, negative = favored
        for side, is_home in (("home", 1), ("away", 0)):
            team = g.get(f"{side}_team")
            if pd.isna(team):
                continue
            team_spread = spread if is_home else (-spread if pd.notna(spread) else None)
            implied = _implied_total(total, team_spread)
            rows.append(
                {
                    "season": g["season"],
                    "week": g["week"],
                    "team": team,
                    "implied_team_total": implied,
                    "spread": team_spread if pd.notna(team_spread) else _NEUTRAL["spread"],
                    "is_home": is_home,
                    "rest_days": g.get(f"{side}_rest", _NEUTRAL["rest_days"]),
                }
            )
    ctx = pd.DataFrame(rows)
    ctx["team_season"] = team_season_key(ctx)
    return ctx


def _implied_total(total, team_spread):
    """Vegas implied points for a team = total/2 - spread/2 (spread<0 favored)."""
    if pd.isna(total) or team_spread is None or pd.isna(team_spread):
        return _NEUTRAL["implied_team_total"]
    return total / 2.0 - team_spread / 2.0


def _neutral_frame(seasons) -> pd.DataFrame:
    """Empty-schedule fallback: callers left-join and get neutral defaults."""
    cols = ["season", "week", "team", "team_season", *CONTEXT_COLUMNS[1:]]
    return pd.DataFrame(columns=cols)


def load_sos(seasons):
    """Season-level strength-of-schedule ranks from the committed sos/ CSVs.

    Keyed by full team name (e.g. 'Tampa Bay Buccaneers'); available 1999-2019.
    Kept as a helper for later context enrichment; not wired into the default
    build so the offline path stays lean.
    """
    frames = []
    for season in seasons:
        path = LEGACY_SOS_DIR / f"{season}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path, index_col=0)
        df.index = df.index.rename("team_name")
        df = df.reset_index()
        df["season"] = season
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
