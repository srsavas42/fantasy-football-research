"""Usage with the scoreboard taken out.

A box score counts a target thrown in the fourth quarter of a 31-point game exactly as it
counts one thrown in a tied first half, and the history features the models read are
built from box scores. A receiver whose team trailed all day looks like a role that grew;
a back whose team led looks like one that shrank. Only the pass-rate-over-expected column
already separates what a team did from the state it was in.

This module rebuilds player usage from play-by-play, one game state at a time:

``neutral``
    Win probability between 20% and 80% with more than two minutes left in the half: the
    plays a coach calls because he wants to, not because of the clock or the score.

``clean``
    Everything except the plays whose outcome was already decided (win probability under
    5% or over 95%): garbage time.

``trailing`` / ``leading``
    Behind or ahead by nine or more points, which is where scripts turn a receiver into
    a volume sink and a back into a clock.

From those it builds, for each player-week, shares of his team's plays in the state
(neutral target share, neutral rush share, garbage-filtered versions of both), how much
of his work came while trailing or leading, and for the team the share of its plays that
were garbage time and its pass rate when the game was neutral. Everything is lagged by
:func:`ffmodel.weekly.features._prior`, like every other history column.

"Snaps per team play" is not built: snap share already is that ratio.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from ffmodel.data import ingest
from ffmodel.weekly.features import HISTORY_ALPHA, TEAM_ALPHA, _prior

NEUTRAL_WP = (0.20, 0.80)
NEUTRAL_MIN_HALF_SECONDS = 120
GARBAGE_WP = (0.05, 0.95)
SCRIPT_MARGIN = 9
LEVEL_ALPHA = 1.0 - 0.5 ** (1.0 / 6.0)     # the slower view of a share, six games

#: Columns the loaders produce, per player-week and per team-week.
PLAYER_COUNTS = (
    "st_tgt", "st_tgt_neutral", "st_tgt_clean", "st_tgt_trail",
    "st_rush", "st_rush_neutral", "st_rush_clean", "st_rush_lead",
)
TEAM_COUNTS = (
    "st_team_pass", "st_team_pass_neutral", "st_team_pass_clean",
    "st_team_rush", "st_team_rush_neutral", "st_team_rush_clean",
)

#: Feature groups, so each idea can be ruled in or out on its own.
NEUTRAL_FEATURES = (
    "prior_neutral_target_share_recent", "prior_neutral_target_share_level",
    "prior_neutral_rush_share_recent", "prior_neutral_rush_share_level",
)
CLEAN_FEATURES = (
    "prior_clean_target_share_recent", "prior_clean_rush_share_recent",
)
SCRIPT_FEATURES = (
    "prior_trailing_target_frac_recent", "prior_leading_rush_frac_recent",
)
TEAM_STATE_FEATURES = ("team_neutral_pass_rate_recent",)
LAST_GAME_FEATURES = ("prior_last_game_garbage_frac",)
STATE_FEATURES = (
    NEUTRAL_FEATURES + CLEAN_FEATURES + SCRIPT_FEATURES + TEAM_STATE_FEATURES + LAST_GAME_FEATURES
)


def _plays(season: int) -> pd.DataFrame | None:
    try:
        plays = ingest.load_pbp([season])
    except Exception:
        return None
    need = ["season", "week", "posteam", "wp", "score_differential", "qtr",
            "half_seconds_remaining", "pass_attempt", "rush_attempt",
            "receiver_player_id", "rusher_player_id"]
    if any(c not in plays.columns for c in need):
        return None
    cols = need + [c for c in ("season_type", "qb_kneel", "two_point_attempt") if c in plays.columns]
    plays = plays[cols]
    if "season_type" in plays:
        plays = plays[plays["season_type"] == "REG"]
    plays = plays[plays["posteam"].notna() & plays["wp"].notna()]
    if "two_point_attempt" in plays:
        plays = plays[plays["two_point_attempt"].fillna(0) == 0]
    if "qb_kneel" in plays:
        plays = plays[plays["qb_kneel"].fillna(0) == 0]
    return plays


def _flag_states(plays: pd.DataFrame) -> pd.DataFrame:
    wp = plays["wp"].to_numpy(float)
    sd = plays["score_differential"].to_numpy(float)
    half = plays["half_seconds_remaining"].to_numpy(float)
    qtr = plays["qtr"].to_numpy(float)
    out = plays.copy()
    out["neutral"] = (
        (wp >= NEUTRAL_WP[0]) & (wp <= NEUTRAL_WP[1]) & (half > NEUTRAL_MIN_HALF_SECONDS) & (qtr <= 4)
    )
    out["clean"] = (wp >= GARBAGE_WP[0]) & (wp <= GARBAGE_WP[1])
    out["trailing"] = sd <= -SCRIPT_MARGIN
    out["leading"] = sd >= SCRIPT_MARGIN
    return out


def load_state_usage(seasons: Iterable[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(player-week, team-week) counts split by game state, from play-by-play."""
    players, teams = [], []
    for season in sorted({int(s) for s in seasons}):
        plays = _plays(season)
        if plays is None:
            continue
        plays = _flag_states(plays)
        key = ["season", "week", "posteam"]

        passes = plays[plays["pass_attempt"] == 1]
        rushes = plays[plays["rush_attempt"] == 1]

        tp = passes.groupby(key).agg(
            st_team_pass=("pass_attempt", "size"),
            st_team_pass_neutral=("neutral", "sum"),
            st_team_pass_clean=("clean", "sum"),
        )
        tr = rushes.groupby(key).agg(
            st_team_rush=("rush_attempt", "size"),
            st_team_rush_neutral=("neutral", "sum"),
            st_team_rush_clean=("clean", "sum"),
        )
        garbage = (
            plays[(plays["pass_attempt"] == 1) | (plays["rush_attempt"] == 1)]
            .assign(garbage=lambda d: ~d["clean"])
            .groupby(key)["garbage"].mean().rename("st_team_garbage_frac")
        )
        teams.append(tp.join(tr, how="outer").join(garbage, how="outer").reset_index())

        tgt = passes[passes["receiver_player_id"].notna()]
        rec = tgt.groupby(key + ["receiver_player_id"]).agg(
            st_tgt=("pass_attempt", "size"),
            st_tgt_neutral=("neutral", "sum"),
            st_tgt_clean=("clean", "sum"),
            st_tgt_trail=("trailing", "sum"),
        ).reset_index().rename(columns={"receiver_player_id": "player_key"})
        car = rushes[rushes["rusher_player_id"].notna()]
        rus = car.groupby(key + ["rusher_player_id"]).agg(
            st_rush=("rush_attempt", "size"),
            st_rush_neutral=("neutral", "sum"),
            st_rush_clean=("clean", "sum"),
            st_rush_lead=("leading", "sum"),
        ).reset_index().rename(columns={"rusher_player_id": "player_key"})
        players.append(rec.merge(rus, on=key + ["player_key"], how="outer"))
        del plays, passes, rushes

    cols = ["season", "week", "team", "player_key", *PLAYER_COUNTS]
    tcols = ["season", "week", "team", *TEAM_COUNTS, "st_team_garbage_frac"]
    if not players:
        return pd.DataFrame(columns=cols), pd.DataFrame(columns=tcols)
    p = pd.concat(players, ignore_index=True).rename(columns={"posteam": "team"})
    t = pd.concat(teams, ignore_index=True).rename(columns={"posteam": "team"})
    for frame in (p, t):
        frame["season"] = frame["season"].astype(int)
        frame["week"] = frame["week"].astype(int)
    p["player_key"] = p["player_key"].astype(str)
    return p.fillna({c: 0 for c in PLAYER_COUNTS}), t


def _share(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator.div(denominator.where(denominator > 0))


def attach_state_usage(panel: pd.DataFrame) -> pd.DataFrame:
    """Attach the lagged state-split usage features to a weekly panel.

    Rows with no play-by-play (the week being projected, a bye) carry only the lagged
    history. A player who played but was not targeted or did not carry has a zero share,
    not a missing one: that is a real observation of no usage in that state. Row order
    and index are those of ``panel``.
    """
    seasons = sorted(panel["season"].unique().tolist())
    players, teams = load_state_usage(seasons)
    out = panel.copy()
    if players.empty:
        for column in STATE_FEATURES:
            out[column] = np.nan
        return out

    # A player traded mid-week would appear under two clubs; his week is one row.
    players = players.groupby(["season", "week", "player_key"], as_index=False)[list(PLAYER_COUNTS)].sum()
    frame = panel.reset_index(drop=True)
    frame["_pos"] = np.arange(len(frame))
    frame = frame.merge(players, on=["season", "week", "player_key"], how="left")
    frame = frame.merge(teams, on=["season", "week", "team"], how="left")
    for column in PLAYER_COUNTS:
        frame[column] = frame[column].fillna(0.0).where(frame["st_team_pass"].notna())

    frame = frame.sort_values(["player_key", "season", "week"], kind="mergesort")
    played = frame["played"].eq(1)
    obs = {
        "neutral_target_share": _share(frame["st_tgt_neutral"], frame["st_team_pass_neutral"]),
        "neutral_rush_share": _share(frame["st_rush_neutral"], frame["st_team_rush_neutral"]),
        "clean_target_share": _share(frame["st_tgt_clean"], frame["st_team_pass_clean"]),
        "clean_rush_share": _share(frame["st_rush_clean"], frame["st_team_rush_clean"]),
        "trailing_target_frac": _share(frame["st_tgt_trail"], frame["st_tgt"]),
        "leading_rush_frac": _share(frame["st_rush_lead"], frame["st_rush"]),
    }
    keys = ["player_key"]
    for name, series in obs.items():
        frame[f"prior_{name}_recent"] = _prior(
            frame, keys, series.where(played), how="ewm", alpha=HISTORY_ALPHA
        )
    for name in ("neutral_target_share", "neutral_rush_share"):
        frame[f"prior_{name}_level"] = _prior(
            frame, keys, obs[name].where(played), how="ewm", alpha=LEVEL_ALPHA
        )
    frame["prior_last_game_garbage_frac"] = _prior(
        frame, keys, frame["st_team_garbage_frac"].where(played), how="ewm", alpha=1.0
    )

    # Team level, across seasons like the other team context, from one row per team-week.
    tw = teams.copy()
    tw["neutral_pass_rate"] = _share(
        tw["st_team_pass_neutral"], tw["st_team_pass_neutral"] + tw["st_team_rush_neutral"]
    )
    tw = tw.sort_values(["team", "season", "week"], kind="mergesort").reset_index(drop=True)
    tw["team_neutral_pass_rate_recent"] = _prior(
        tw, ["team"], tw["neutral_pass_rate"], how="ewm", alpha=TEAM_ALPHA
    )
    frame = frame.merge(
        tw[["season", "week", "team", "team_neutral_pass_rate_recent"]],
        on=["season", "week", "team"], how="left",
    )
    # A week with no play-by-play of its own (the one being projected) inherits the club's
    # latest value. The lag already guarantees the value never includes that week.
    frame = frame.sort_values(["team", "season", "week"], kind="mergesort")
    frame["team_neutral_pass_rate_recent"] = frame.groupby("team")["team_neutral_pass_rate_recent"].ffill()

    frame = frame.sort_values("_pos")
    for column in STATE_FEATURES:
        out[column] = frame[column].to_numpy()
    return out
