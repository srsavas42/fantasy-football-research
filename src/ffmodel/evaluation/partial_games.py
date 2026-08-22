"""Per-game scoring rates, with and without the games a player left early.

``games`` counts a game as one whether the player took three snaps or seventy,
so a season's points-per-game mixes two different quantities: what a player
scores when he plays, and how often he does not finish. A back who tears a
hamstring on the opening drive of week six contributes a near-zero game to his
own average, and nothing in the season row says so.

This separates them. A week is **partial** when the player's offensive snap
share that week falls below half his own median for the season -- his own, not
the position's, because a third-down back at a 35% median is not injured and a
bell-cow at 12% for one week is. Two rates come out of that:

``raw_ppg``
    season points divided by games played, the conventional quantity.
``clean_ppg``
    points in full games divided by the count of full games.

The gap between them is the attrition tax: how much of a player's per-game
average is games he did not finish.

Two things this is not.

It is **not a cleaned training target**. Regressing on the clean quantity was
measured and is worse -- the contamination is partly signal, because attrition
persists, and the clean mean is computed on fewer games (see
docs/partial-games-2026-08.md). This is for evaluation only.

It is **not available before the season**. Every input here is an outcome. A
projection compared against ``clean_ppg`` is being asked a counterfactual
question -- what would he have averaged in the games he finished -- which is
worth asking precisely because it is not the question the model was trained on.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ffmodel.simulation.scoring import fantasy_points

# Half the player's own median. A share of exactly zero is not "partial" but
# "absent": nflverse publishes a snap row at 0% for players who dressed and did
# not take an offensive snap, and those are not games a scoring average should
# be divided by either -- they are handled by MIN_SHARE below.
PARTIAL_FRACTION = 0.5

# Below this the week is not a game at all. Without a floor, a player whose
# median share is 4% (a short-yardage back) has no week that can fall under half
# of it in absolute terms worth caring about.
MIN_SHARE = 0.02

# A median taken over three games is not a median. Player-seasons shorter than
# this get no partial flags -- their clean rate equals their raw rate rather
# than being computed from a threshold built on nothing.
MIN_GAMES_FOR_MEDIAN = 4

# A player has to have had a role for "half his usual role" to mean anything.
# Without this the rule inverts on the players it least applies to: a returner
# who takes no offensive snaps has a median of zero, every week falls under the
# MIN_SHARE floor, and his entire season reads as partial. On the 2024 weekly
# feed that was 63% of all player-weeks. Ten percent is roughly a rotational
# third receiver; below it a week's snap share is not measuring playing time so
# much as whether the player dressed.
MIN_MEDIAN_SHARE = 0.10

STAT_COLUMNS = (
    "pass_yds",
    "pass_td",
    "pass_int",
    "rush_yds",
    "rush_td",
    "rec_yds",
    "rec_td",
    "receptions",
    "fumbles_lost",
)


def weekly_snap_shares(seasons, load_snaps=None, load_ids=None) -> pd.DataFrame:
    """Per player-week offensive snap share, keyed by GSIS id.

    The snap feed is keyed by Pro-Football-Reference identifiers and the stat
    feed by GSIS ones, so the two only meet through the id bridge. Rows the
    bridge cannot resolve are dropped rather than name-matched: a wrong join
    here would silently attribute one player's early exit to another.
    """
    from ffmodel.data import ingest

    load_snaps = load_snaps or (lambda years: ingest.load_snap_counts(years))
    load_ids = load_ids or ingest.load_ids

    snaps = load_snaps(sorted({int(s) for s in seasons}))
    if "game_type" in snaps:
        snaps = snaps[snaps["game_type"].astype(str).eq("REG")]
    bridge = load_ids()[["pfr_id", "gsis_id"]].dropna().drop_duplicates("pfr_id")
    out = snaps.merge(
        bridge, left_on="pfr_player_id", right_on="pfr_id", how="inner"
    )
    out = out.rename(columns={"gsis_id": "player_id"})
    out["snap_share"] = pd.to_numeric(out["offense_pct"], errors="coerce")
    keep = ["player_id", "season", "week", "snap_share"]
    return (
        out[keep]
        .dropna(subset=["player_id", "snap_share"])
        .drop_duplicates(["player_id", "season", "week"])
        .reset_index(drop=True)
    )


def weekly_points(seasons, scoring: str = "ppr", load_weekly=None) -> pd.DataFrame:
    """Per player-week fantasy points under the package's own scoring rules.

    Computed from the stat components rather than read from a provider's
    ``fantasy_points_ppr`` column, so a week here and a season in the model are
    scored by the same code.
    """
    from ffmodel.data import ingest

    load_weekly = load_weekly or (lambda years: ingest.load_weekly(years))
    weekly = load_weekly(sorted({int(s) for s in seasons})).copy()
    for column in STAT_COLUMNS:
        weekly[column] = pd.to_numeric(
            weekly.get(column, pd.Series(np.nan, index=weekly.index)), errors="coerce"
        ).fillna(0.0)
    weekly["points"] = fantasy_points(weekly[list(STAT_COLUMNS)], scoring)
    return weekly[["player_id", "season", "week", "team", "position", "points"]]


def per_game_rates(seasons, scoring: str = "ppr", **loaders) -> pd.DataFrame:
    """Raw and partial-cleaned points per game, per player-season.

    Returned per player-season rather than per player-team-season: a player
    traded in October has one scoring average, and splitting it would make the
    denominators disagree with the model's ``games``.
    """
    points = weekly_points(seasons, scoring=scoring, load_weekly=loaders.get("load_weekly"))
    shares = weekly_snap_shares(
        seasons,
        load_snaps=loaders.get("load_snaps"),
        load_ids=loaders.get("load_ids"),
    )
    frame = points.merge(shares, on=["player_id", "season", "week"], how="left")

    keys = ["player_id", "season"]
    median = (
        frame.groupby(keys)["snap_share"]
        .transform("median")
        .astype(float)
    )
    observed = frame.groupby(keys)["snap_share"].transform("count")
    threshold = np.maximum(PARTIAL_FRACTION * median, MIN_SHARE)
    # A week with no snap row is not evidence of an early exit. Left as full so
    # the clean average never quietly drops a game for a missing feed row --
    # which is common for quarterbacks in older seasons.
    frame["partial"] = (
        frame["snap_share"].notna()
        & frame["snap_share"].lt(threshold)
        & observed.ge(MIN_GAMES_FOR_MEDIAN)
        & median.ge(MIN_MEDIAN_SHARE)
    )

    grouped = frame.groupby(keys)
    out = grouped.agg(
        weeks=("points", "size"),
        total_points=("points", "sum"),
        partial_weeks=("partial", "sum"),
        median_snap_share=("snap_share", "median"),
        # A season-average row covers one player-team stint, so for a player
        # traded in October its points and its games both stop at the trade
        # while the totals here do not. Comparing the two would divide a whole
        # season's production by part of a season's exposure. The count is
        # reported rather than resolved: a caller wanting a like-for-like
        # denominator restricts to one team, and one wanting the player's real
        # season does not.
        teams=("team", "nunique"),
    ).reset_index()
    full = frame[~frame["partial"]].groupby(keys).agg(
        full_weeks=("points", "size"),
        full_points=("points", "sum"),
    ).reset_index()
    out = out.merge(full, on=keys, how="left")
    out["full_weeks"] = out["full_weeks"].fillna(0).astype(int)
    out["full_points"] = out["full_points"].fillna(0.0)

    out["raw_ppg"] = out["total_points"] / out["weeks"].replace(0, np.nan)
    # The threshold is a fraction of the player's own median, so at most half
    # his weeks can fall below it and ``full_weeks`` is never zero -- over
    # 2022-2025 the largest partial fraction any player-season reaches is
    # exactly 0.5. The guard stays because a caller raising PARTIAL_FRACTION
    # above 1.0 would break that, and a silent division by zero is worse than a
    # missing value.
    out["clean_ppg"] = out["full_points"] / out["full_weeks"].replace(0, np.nan)
    return out
