"""Usage per opportunity: how often he is targeted per route, and how much he plays.

The history the models read already carries shares of the team's work (target share, rush
share, snap share). A share is a product of two things a coach decides separately:

    target share  =  how often he is on the field for a pass  x  how often he is targeted
                                                                    when he is there

Participation says whether he is in the game; the per-opportunity rate says whether the
offense uses him once he is. They age differently and a share cannot tell them apart: a
receiver whose share fell because his snaps fell (a role change) and one whose share fell
because he is targeted less per snap (an offense that stopped looking for him) are the
same number.

Routes run are not published, so the denominator is the proxy the participation file
supports: dropback plays with the player on the field (``pass_snaps``). It counts a back
who stayed in to block, so it understates targets per route for backs and tight ends; for
receivers it is close. The same on designed runs (``run_snaps``).

Every rate is a ratio of two exponentially weighted sums, not an average of weekly ratios,
so a game with three snaps does not count like a game with sixty. Everything is lagged by
:func:`ffmodel.weekly.features._prior`.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from ffmodel.data import ingest
from ffmodel.weekly.features import HISTORY_ALPHA, _prior

LEVEL_ALPHA = 1.0 - 0.5 ** (1.0 / 6.0)     # the slower view of a rate, six games

#: Per-opportunity counts built from play-by-play and participation.
PLAYER_COUNTS = ("op_pass_snaps", "op_run_snaps")
TEAM_COUNTS = ("op_team_dropbacks", "op_team_runs")

#: Feature groups, so each idea can be ruled in or out on its own.
SIMPLE_FEATURES = (            # panel snaps only; no play-by-play needed
    "prior_targets_per_snap_recent", "prior_targets_per_snap_level",
    "prior_rushes_per_snap_recent", "prior_rushes_per_snap_level",
)
PARTICIPATION_FEATURES = (     # is he on the field
    "prior_pass_snap_share_recent", "prior_pass_snap_share_level",
    "prior_run_snap_share_recent", "prior_run_snap_share_level",
)
EARN_FEATURES = (              # does the offense use him when he is
    "prior_targets_per_pass_snap_recent", "prior_targets_per_pass_snap_level",
    "prior_rushes_per_run_snap_recent", "prior_rushes_per_run_snap_level",
)
YIELD_FEATURES = (             # what a snap is worth
    "prior_rec_yards_per_pass_snap_recent",
    "prior_points_per_snap_recent", "prior_points_per_snap_level",
    "prior_touches_per_snap_recent", "prior_touches_per_snap_level",
)
RATE_FEATURES = SIMPLE_FEATURES + PARTICIPATION_FEATURES + EARN_FEATURES + YIELD_FEATURES


def _season_counts(season: int) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    try:
        plays = ingest.load_pbp([season])
        part = ingest.load_participation([season])
    except Exception:
        return None
    part = part if isinstance(part, pd.DataFrame) else part.to_pandas()
    need = ["game_id", "play_id", "week", "posteam", "qb_dropback", "rush_attempt"]
    if any(c not in plays.columns for c in need) or "offense_players" not in part.columns:
        return None
    cols = need + [c for c in ("season_type", "qb_scramble", "qb_kneel", "two_point_attempt") if c in plays.columns]
    plays = plays[cols]
    if "season_type" in plays:
        plays = plays[plays["season_type"] == "REG"]
    plays = plays[plays["posteam"].notna()]
    for flag in ("two_point_attempt", "qb_kneel"):
        if flag in plays:
            plays = plays[plays[flag].fillna(0) == 0]
    part = part[["nflverse_game_id", "play_id", "offense_players"]].dropna()
    plays = plays.merge(part, left_on=["game_id", "play_id"], right_on=["nflverse_game_id", "play_id"], how="inner")

    scramble = plays["qb_scramble"].fillna(0) if "qb_scramble" in plays else 0
    plays["is_pass"] = plays["qb_dropback"].fillna(0) == 1
    plays["is_run"] = (plays["rush_attempt"].fillna(0) == 1) & (scramble == 0)
    plays = plays[plays["is_pass"] | plays["is_run"]]

    key = ["week", "posteam"]
    team = plays.groupby(key).agg(op_team_dropbacks=("is_pass", "sum"), op_team_runs=("is_run", "sum")).reset_index()
    team["season"] = season

    long = plays[key + ["is_pass", "is_run", "offense_players"]].copy()
    long["player_key"] = long["offense_players"].str.split(";")
    long = long.explode("player_key").drop(columns="offense_players")
    player = long.groupby(key + ["player_key"]).agg(op_pass_snaps=("is_pass", "sum"), op_run_snaps=("is_run", "sum")).reset_index()
    player["season"] = season
    return player, team


def load_opportunity(seasons: Iterable[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(player-week, team-week) pass and run snaps from play-by-play and participation."""
    players, teams = [], []
    for season in sorted({int(s) for s in seasons}):
        made = _season_counts(season)
        if made is None:
            continue
        players.append(made[0])
        teams.append(made[1])
    pcols = ["season", "week", "team", "player_key", *PLAYER_COUNTS]
    tcols = ["season", "week", "team", *TEAM_COUNTS]
    if not players:
        return pd.DataFrame(columns=pcols), pd.DataFrame(columns=tcols)
    p = pd.concat(players, ignore_index=True).rename(columns={"posteam": "team"})
    t = pd.concat(teams, ignore_index=True).rename(columns={"posteam": "team"})
    for frame in (p, t):
        frame["season"] = frame["season"].astype(int)
        frame["week"] = frame["week"].astype(int)
    return p[pcols], t[tcols]


def _ratio(
    frame: pd.DataFrame, keys: list[str], num: pd.Series, den: pd.Series, alpha: float, cap: float = 1.0
) -> pd.Series:
    """Ratio of two decayed sums: a rate weighted by its own opportunities."""
    # A game with no opportunities says nothing about the rate (and a numerator over a zero
    # denominator is a gap in the snap counts, not a rate), so it leaves both sums alone.
    observed = num.notna() & den.notna() & (den > 0)
    top = _prior(frame, keys, num.where(observed), how="ewm", alpha=alpha)
    bottom = _prior(frame, keys, den.where(observed), how="ewm", alpha=alpha)
    return (top / bottom.where(bottom > 0)).clip(lower=0.0, upper=cap)


def attach_opportunity_rates(panel: pd.DataFrame) -> pd.DataFrame:
    """Attach the lagged per-opportunity features to a weekly panel (order and index kept).

    The simple group needs only the panel's own snap counts. The others need
    participation data; where it is missing they are left empty.
    """
    out = panel.copy()
    seasons = sorted(panel["season"].unique().tolist())
    players, teams = load_opportunity(seasons)

    frame = panel.reset_index(drop=True)
    frame["_pos"] = np.arange(len(frame))
    if not players.empty:
        players = players.groupby(["season", "week", "player_key"], as_index=False)[list(PLAYER_COUNTS)].sum()
        frame = frame.merge(players, on=["season", "week", "player_key"], how="left")
        frame = frame.merge(teams, on=["season", "week", "team"], how="left")
        for column in PLAYER_COUNTS:
            frame[column] = frame[column].fillna(0.0).where(frame["op_team_dropbacks"].notna())
    else:
        for column in (*PLAYER_COUNTS, *TEAM_COUNTS):
            frame[column] = np.nan

    frame = frame.sort_values(["player_key", "season", "week"], kind="mergesort")
    played = frame["played"].eq(1)
    keys = ["player_key"]

    def num(column: str) -> pd.Series:
        return pd.to_numeric(frame[column], errors="coerce").where(played)

    snaps = num("offense_snaps")
    targets, rushes = num("targets"), num("rush_att")
    pass_snaps, run_snaps = num("op_pass_snaps"), num("op_run_snaps")
    team_pass, team_run = num("op_team_dropbacks"), num("op_team_runs")
    rec_yards, points = num("rec_yds"), num("points")

    # name -> (numerator, denominator, also a six-game view, upper bound). The bounds are
    # physical for the counts (a rate of work per snap cannot sensibly pass one) and
    # generous for yards and points.
    spec = {
        "targets_per_snap": (targets, snaps, True, 1.0),
        "rushes_per_snap": (rushes, snaps, True, 1.0),
        "pass_snap_share": (pass_snaps, team_pass, True, 1.0),
        "run_snap_share": (run_snaps, team_run, True, 1.0),
        "targets_per_pass_snap": (targets, pass_snaps, True, 1.0),
        "rushes_per_run_snap": (rushes, run_snaps, True, 1.0),
        "rec_yards_per_pass_snap": (rec_yards, pass_snaps, False, 15.0),
        "points_per_snap": (points, snaps, True, 3.0),
        "touches_per_snap": (targets + rushes, snaps, True, 1.5),
    }
    for name, (top, bottom, level, cap) in spec.items():
        frame[f"prior_{name}_recent"] = _ratio(frame, keys, top, bottom, HISTORY_ALPHA, cap)
        if level:
            frame[f"prior_{name}_level"] = _ratio(frame, keys, top, bottom, LEVEL_ALPHA, cap)

    frame = frame.sort_values("_pos")
    for column in RATE_FEATURES:
        out[column] = frame[column].to_numpy()
    return out
