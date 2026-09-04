"""What is known before kickoff but has not happened yet.

Every other feature in this layer is an exponentially weighted average of what a
player has already done. That is why the model is seven points low in the week a
role opens up and unbiased two weeks later: a promotion enters the features only
as it is produced, so the week it matters most is the week there is no data for
it. No decay rate fixes that. Only a *leading* indicator does.

Two feeds already cached carry one, and both are published before the game they
describe:

**The injury report.** Teams file a game-status report — Out, Doubtful,
Questionable — plus a practice participation level. Measured against kickoff
across 2022-2024, it lands a median 28 hours early, 99% of it at least 2.7 hours
early, and 0.18% after gameday (Thursday and Monday games in the wrong
timezone). It is legitimately available at decision time, and it is the only
direct statement anyone makes about whether a player will be on the field.

**The depth chart.** Clubs publish a positional ordering weekly. It moves: 38% of
skill players change their listed rank at least once within a season. It is the
one artefact that says a backup has been promoted *before* he touches the ball.

The distinction that matters for both is between the two things a role change
does. A starter being ruled out is a fact about *him*; that his backup will
absorb the work is an inference about *someone else*, and the error attribution
says the second is worth far more — a backup whose lead back is inactive is
projected 2.1 points low, while a player whose role actually grows is projected
7.0 low. So the features here are built in pairs: what this player's own report
says, and what the report says about the players listed ahead of him.

**On leakage.** The depth chart for week ``w`` is placed by the loader against
the next regular-season game to be played, so it precedes that game. That is the
right semantics and it is still a scraped artefact that could in principle be
revised. Every depth feature therefore comes in two forms — the contemporaneous
one, and a strictly-lagged one built from week ``w-1`` that no revision can
contaminate. If the two disagree about how much the signal is worth, the lagged
number is the one to believe.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from ffmodel.data import ingest

# Ordered worst to best; a higher number is more likely to miss the game.
STATUS_SEVERITY = {
    "Out": 3.0,
    "Doubtful": 2.0,
    "Questionable": 1.0,
}

PRACTICE_SEVERITY = {
    "Did Not Participate In Practice": 2.0,
    "Limited Participation in Practice": 1.0,
    "Full Participation in Practice": 0.0,
}

NEWS_COLUMNS = (
    "inj_status",
    "inj_practice",
    "inj_out",
    "inj_questionable_or_worse",
    "depth_rank",
    "depth_rank_lagged",
    "depth_promoted",
    "depth_promoted_lagged",
    "ahead_out",
    "ahead_out_lagged",
    "position_group_out",
)


def load_injury_report(seasons: Iterable[int]) -> pd.DataFrame:
    """Weekly game-status and practice reports, keyed like the panel."""
    try:
        raw = ingest.load_injuries(list(seasons))
    except Exception:
        return pd.DataFrame(
            columns=["season", "week", "player_key", "inj_status", "inj_practice"]
        )
    if raw.empty:
        return pd.DataFrame(
            columns=["season", "week", "player_key", "inj_status", "inj_practice"]
        )
    frame = raw
    if "game_type" in frame.columns:
        frame = frame[frame["game_type"] == "REG"]
    out = pd.DataFrame(
        {
            "season": pd.to_numeric(frame["season"], errors="coerce"),
            "week": pd.to_numeric(frame["week"], errors="coerce"),
            "player_key": frame["gsis_id"].astype(str),
            "inj_status": frame["report_status"]
            .map(STATUS_SEVERITY)
            .astype(float)
            .fillna(0.0),
            "inj_practice": frame["practice_status"]
            .map(PRACTICE_SEVERITY)
            .astype(float)
            .fillna(0.0),
        }
    ).dropna(subset=["season", "week"])
    out["season"] = out["season"].astype(int)
    out["week"] = out["week"].astype(int)
    # A player can appear twice in a week as the report is updated; the most
    # severe entry is the one a manager would act on.
    return out.groupby(["season", "week", "player_key"], as_index=False).max()


def load_depth(seasons: Iterable[int]) -> pd.DataFrame:
    """Weekly offensive depth-chart rank, keyed like the panel."""
    try:
        raw = ingest.load_depth_charts(list(seasons))
    except Exception:
        return pd.DataFrame(columns=["season", "week", "player_key", "depth_rank"])
    if raw.empty:
        return pd.DataFrame(columns=["season", "week", "player_key", "depth_rank"])
    frame = raw
    if "game_type" in frame.columns:
        frame = frame[frame["game_type"].isin(["REG", None]) | frame["game_type"].isna()]
    if "formation" in frame.columns:
        frame = frame[frame["formation"].eq("Offense") | frame["formation"].isna()]
    if "gsis_id" not in frame.columns:
        return pd.DataFrame(columns=["season", "week", "player_key", "depth_rank"])
    out = pd.DataFrame(
        {
            "season": pd.to_numeric(frame["season"], errors="coerce"),
            "week": pd.to_numeric(frame["week"], errors="coerce"),
            "player_key": frame["gsis_id"].astype(str),
            "depth_rank": pd.to_numeric(frame["depth_team"], errors="coerce"),
        }
    ).dropna(subset=["season", "week", "depth_rank"])
    out["season"] = out["season"].astype(int)
    out["week"] = out["week"].astype(int)
    # The best (lowest) listing wins when a player appears in two packages.
    return out.groupby(["season", "week", "player_key"], as_index=False).min()


def _ahead_out(frame: pd.DataFrame, rank: str, status: str) -> pd.Series:
    """Is anyone listed ahead of this player at his position ruled out?

    "Ahead" is by depth rank inside the same club, week and position, so this is
    the inference the error attribution says is worth the most: not that a player
    is hurt, but that the person in front of him is.
    """
    work = frame[["season", "week", "team", "position"]].copy()
    work["rank"] = pd.to_numeric(frame[rank], errors="coerce")
    work["out"] = pd.to_numeric(frame[status], errors="coerce").ge(2.0)
    work["row"] = np.arange(len(work))

    hurt = work[work["out"] & work["rank"].notna()]
    if hurt.empty:
        return pd.Series(0.0, index=frame.index)
    # Best (lowest) rank among the ruled-out players in each room.
    best = hurt.groupby(["season", "week", "team", "position"], as_index=False)[
        "rank"
    ].min().rename(columns={"rank": "best_out"})
    merged = work.merge(best, on=["season", "week", "team", "position"], how="left")
    ahead = merged["best_out"].notna() & merged["rank"].notna() & (
        merged["best_out"] < merged["rank"]
    )
    return pd.Series(ahead.to_numpy(dtype=float), index=frame.index)


def add_news_features(panel: pd.DataFrame, *, seasons=None) -> pd.DataFrame:
    """Attach the pre-game injury and depth signals to a weekly panel."""
    seasons = sorted(panel["season"].unique().tolist()) if seasons is None else seasons
    frame = panel.copy()

    injuries = load_injury_report(seasons)
    if injuries.empty:
        frame["inj_status"] = 0.0
        frame["inj_practice"] = 0.0
    else:
        frame = frame.merge(injuries, on=["season", "week", "player_key"], how="left")
        # No entry on the report is the overwhelmingly common case and means
        # healthy, not unknown -- the report lists only players with something
        # to declare.
        frame["inj_status"] = frame["inj_status"].fillna(0.0)
        frame["inj_practice"] = frame["inj_practice"].fillna(0.0)
    frame["inj_out"] = frame["inj_status"].ge(3.0).astype(float)
    frame["inj_questionable_or_worse"] = frame["inj_status"].ge(1.0).astype(float)

    depth = load_depth(seasons)
    if depth.empty:
        frame["depth_rank"] = np.nan
    else:
        frame = frame.merge(depth, on=["season", "week", "player_key"], how="left")

    frame = frame.sort_values(["player_key", "season", "week"], kind="mergesort")
    grouped = frame.groupby(["player_key", "season"], sort=False)
    frame["depth_rank_lagged"] = grouped["depth_rank"].shift(1)
    frame["inj_status_lagged"] = grouped["inj_status"].shift(1).fillna(0.0)
    # A promotion is a rank that improved. Positive means he moved up.
    frame["depth_promoted"] = (
        frame["depth_rank_lagged"] - frame["depth_rank"]
    ).fillna(0.0)
    frame["depth_promoted_lagged"] = (
        grouped["depth_rank"].shift(2) - frame["depth_rank_lagged"]
    ).fillna(0.0)

    frame["ahead_out"] = _ahead_out(frame, "depth_rank", "inj_status")
    frame["ahead_out_lagged"] = _ahead_out(
        frame, "depth_rank_lagged", "inj_status_lagged"
    )
    frame["position_group_out"] = (
        frame.assign(_out=frame["inj_status"].ge(2.0).astype(float))
        .groupby(["season", "week", "team", "position"])["_out"]
        .transform("sum")
        .to_numpy()
    )
    return frame.reset_index(drop=True)
