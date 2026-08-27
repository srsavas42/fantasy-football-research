"""The player-week panel, with the zeros in it.

A stat feed only contains lines for players who recorded something. Building a
weekly model on those rows alone asks "how many points did he score, given he
scored", which is not the question a lineup decision poses. The week a starter is
inactive is not missing data -- it is the outcome, and it is the outcome that
costs a fantasy manager the most.

So the panel is keyed on the *roster*, not the stat feed: one row per
(season, week, player) for every player under contract with a club that week,
carrying the stat line if there is one and an honest zero if there is not. That
is what makes ``played`` a response rather than a filter, and it is what lets the
rest-of-season target sum weeks a player missed instead of skipping them.

Three construction details matter and are easy to get wrong:

**Bye weeks are removed, not zeroed.** A team-week with no game produces no stat
lines for anybody, which is indistinguishable from a full roster of inactives
unless the schedule is consulted. It is consulted here by asking which
(season, week, team) triples appear anywhere in the stat feed -- every game
played produces lines. A bye is not a decision anyone gets wrong, and leaving it
in would hand the model a large, trivially predictable block of zeros.

**Week-``w`` roster status is recorded but is never a feature.** Whether a player
was declared inactive is the thing being predicted. It is kept for diagnostics
and for defining populations after the fact; ``features.py`` never reads it.

**Team context is derived from the same rows.** Team pass attempts, carries and
targets come from summing the stat feed over a team-week rather than from a
separate team endpoint, so a team total and the player shares that divide by it
can never disagree.

Positions are restricted to the four that fantasy rosters draft. Kickers and
defenses score under different rules and would need their own responses.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

from ffmodel.data import ingest
from ffmodel.data.loaders import load_player_weeks
from ffmodel.simulation.scoring import fantasy_points

PANEL_POSITIONS = ("QB", "RB", "WR", "TE")

# Weekly rosters begin in 2011 upstream, but they are not usable until 2016.
# Before that the reserve label does not mean what it says: players marked RES
# record a stat line 65-73% of the time, which is impossible for injured
# reserve, and the panel holds ~6,000 rows a season against ~9,500 afterwards.
# Whatever those rows are, they are not "employed and did not play", so the
# zeros would not be honest and the response would quietly change meaning
# halfway through the window.
#
# From 2016 the panel is stable: RES falls to a ~0% play rate and the overall
# play rate sits between 0.54 and 0.59 for ten straight seasons. The ACT/INA
# split does move -- inactives are only reported separately from 2019, so
# before that they are carried inside ACT -- but that is a relabelling within
# the panel, not a change to which player-weeks it contains, and no feature
# reads the label.
FIRST_PANEL_SEASON = 2016

# Under contract with the club on gameday, in some capacity. Practice-squad
# (DEV) and cut players are excluded unless they actually recorded a line, which
# is how an elevation still reaches the panel.
CONTRACT_STATUSES = ("ACT", "INA", "RES", "EXE", "PUP", "NON", "TRC")

# Columns summed over a team-week to describe the offense a player is attached
# to. Each has a player-level counterpart, so a share is always a ratio of two
# numbers from the same rows.
TEAM_SUM_COLUMNS = (
    "pass_att",
    "pass_yds",
    "pass_td",
    "rush_att",
    "rush_yds",
    "rush_td",
    "targets",
    "receptions",
    "rec_yds",
    "rec_td",
)

STAT_COLUMNS = (
    "pass_att",
    "pass_cmp",
    "pass_yds",
    "pass_td",
    "pass_int",
    "rush_att",
    "rush_yds",
    "rush_td",
    "targets",
    "receptions",
    "rec_yds",
    "rec_td",
    "fumbles_lost",
    # Efficiency, not just volume. A defence can concede few rushing yards
    # because it is hard to run on or because nobody runs on it, and yards per
    # carry separates the two; EPA does it better still by pricing down and
    # distance. Both are needed for "good against the run" to mean anything.
    "rush_epa",
    "rec_epa",
)


def _opponent_map(seasons: Iterable[int]) -> pd.DataFrame:
    """(season, week, team) -> opponent, read off the stat feed itself.

    The canonical weekly schema drops ``opponent_team``, and the schedule cache
    does not cover every season in the panel. The raw feed carries one opponent
    per player row; collapsing it to the team-week is exact because every player
    on a team plays the same opponent.
    """
    frames = []
    for season in seasons:
        try:
            raw = ingest._by_season(
                "player_stats", [season], params={"summary_level": "week"}
            )
        except Exception:
            continue
        if raw.empty or "opponent_team" not in raw.columns:
            continue
        if "season_type" in raw.columns:
            raw = raw[raw["season_type"] == "REG"]
        block = raw[["season", "week", "team", "opponent_team"]].dropna()
        frames.append(block.drop_duplicates(subset=["season", "week", "team"]))
    if not frames:
        return pd.DataFrame(columns=["season", "week", "team", "opponent"])
    out = pd.concat(frames, ignore_index=True).rename(
        columns={"opponent_team": "opponent"}
    )
    for column in ("season", "week"):
        out[column] = pd.to_numeric(out[column], errors="coerce").astype("Int64")
    return out.dropna(subset=["season", "week"]).drop_duplicates(
        subset=["season", "week", "team"]
    )


def _market_lines(seasons: Iterable[int]) -> pd.DataFrame:
    """(season, week, team) -> closing spread and total, from the schedule.

    The only prospective information in the panel. Every other column describes
    what has already happened; the line is the market's forecast of the game
    about to be played, published before it, and it is the natural home for
    game script -- who is expected to lead, and how much scoring is expected.

    ``spread_line`` is quoted from the home team's perspective, positive when
    the home side is favoured. It is re-signed per team here so that a positive
    spread always means *this* team is favoured, which is the only convention
    under which a fitted coefficient can be read.

    The implied totals follow from the identity that a spread and a total
    determine both sides' expected scores: they sum to the total and differ by
    the spread. ``implied_team_total`` is how many points this offence is
    expected to produce, and ``implied_opponent_total`` is how much it is
    expected to have to keep up with.
    """
    try:
        schedule = ingest.load_schedules(list(seasons))
    except Exception:
        return pd.DataFrame(
            columns=["season", "week", "team", "spread", "game_total"]
        )
    if schedule.empty:
        return pd.DataFrame(columns=["season", "week", "team", "spread", "game_total"])
    needed = {"season", "week", "home_team", "away_team", "spread_line", "total_line"}
    if not needed.issubset(schedule.columns):
        return pd.DataFrame(columns=["season", "week", "team", "spread", "game_total"])
    if "game_type" in schedule.columns:
        schedule = schedule[schedule["game_type"] == "REG"]

    frames = []
    for side, sign in (("home_team", 1.0), ("away_team", -1.0)):
        block = schedule[["season", "week", side, "spread_line", "total_line"]].copy()
        block = block.rename(columns={side: "team", "total_line": "game_total"})
        block["spread"] = sign * pd.to_numeric(
            block.pop("spread_line"), errors="coerce"
        )
        frames.append(block)
    out = pd.concat(frames, ignore_index=True)
    out["game_total"] = pd.to_numeric(out["game_total"], errors="coerce")
    out["implied_team_total"] = out["game_total"] / 2.0 + out["spread"] / 2.0
    out["implied_opponent_total"] = out["game_total"] / 2.0 - out["spread"] / 2.0
    for column in ("season", "week"):
        out[column] = pd.to_numeric(out[column], errors="coerce").astype("Int64")
    return out.dropna(subset=["season", "week"]).drop_duplicates(
        subset=["season", "week", "team"]
    )


def _roster_weeks(seasons: Iterable[int]) -> pd.DataFrame:
    """Skill-position players under contract, one row per club-week."""
    rosters = ingest.load_weekly_rosters(list(seasons))
    if rosters.empty:
        return pd.DataFrame(columns=["season", "week", "team", "player_id", "position"])
    keep = rosters
    if "game_type" in keep.columns:
        keep = keep[keep["game_type"] == "REG"]
    position = keep.get("position")
    if position is None:
        raise ValueError("weekly rosters carry no position column")
    keep = keep[position.isin(PANEL_POSITIONS)]
    status = keep.get("status")
    if status is not None:
        keep = keep[status.isin(CONTRACT_STATUSES)]
    out = keep.rename(columns={"gsis_id": "player_id", "full_name": "player_name"})
    out = out[out["player_id"].notna()]
    columns = ["season", "week", "team", "player_id", "position", "player_name"]
    if "status" in out.columns:
        columns.append("status")
    out = out[[c for c in columns if c in out.columns]].copy()
    for column in ("season", "week"):
        out[column] = pd.to_numeric(out[column], errors="coerce").astype("int64")
    # A player can appear twice in a club-week upstream (a status correction).
    # Keep one row: the panel is keyed on the decision, not on the transaction.
    return out.drop_duplicates(subset=["season", "week", "player_id"], keep="last")


def build_panel(
    seasons: Iterable[int],
    *,
    scoring: str = "ppr",
) -> pd.DataFrame:
    """One row per rostered skill player per team-week, zeros included."""
    seasons = sorted({int(s) for s in seasons})
    too_early = [s for s in seasons if s < FIRST_PANEL_SEASON]
    if too_early:
        raise ValueError(
            f"weekly rosters start in {FIRST_PANEL_SEASON}; cannot build honest "
            f"zeros for {too_early}"
        )

    stats = load_player_weeks(seasons)
    stats = stats[stats["position"].isin(PANEL_POSITIONS)].copy()
    for column in STAT_COLUMNS:
        if column not in stats.columns:
            stats[column] = 0.0
        stats[column] = pd.to_numeric(stats[column], errors="coerce").fillna(0.0)
    stats["points"] = fantasy_points(stats, scoring).astype(float)

    # Every game played leaves stat lines, so the team-weeks present in the feed
    # are exactly the team-weeks that had a game. Anything else is a bye.
    played_team_weeks = (
        load_player_weeks(seasons)[["season", "week", "team"]]
        .drop_duplicates()
        .assign(team_played=1)
    )

    team_totals = (
        stats.groupby(["season", "week", "team"], as_index=False)[
            list(TEAM_SUM_COLUMNS)
        ]
        .sum()
        .rename(columns={c: f"team_{c}" for c in TEAM_SUM_COLUMNS})
    )
    team_totals["team_points"] = (
        stats.groupby(["season", "week", "team"])["points"].sum().to_numpy()
    )
    team_totals["team_plays"] = (
        team_totals["team_pass_att"] + team_totals["team_rush_att"]
    )

    roster = _roster_weeks(seasons)
    line = stats[
        ["season", "week", "player_id", "team", "position", "points", *STAT_COLUMNS]
    ].copy()
    line = line.drop_duplicates(subset=["season", "week", "player_id"], keep="last")
    line["played"] = 1

    # Outer join: a rostered player with no line becomes a zero, and a player who
    # recorded a line while off the contract list (a practice-squad elevation)
    # still reaches the panel.
    panel = roster.merge(
        line,
        on=["season", "week", "player_id"],
        how="outer",
        suffixes=("", "_stat"),
    )
    panel["team"] = panel["team"].combine_first(panel["team_stat"])
    panel["position"] = panel["position"].combine_first(panel["position_stat"])
    panel = panel.drop(columns=["team_stat", "position_stat"], errors="ignore")
    panel = panel[panel["position"].isin(PANEL_POSITIONS)]
    panel = panel[panel["team"].notna()]

    panel["played"] = panel["played"].fillna(0).astype(int)
    panel["points"] = pd.to_numeric(panel["points"], errors="coerce").fillna(0.0)
    for column in STAT_COLUMNS:
        panel[column] = pd.to_numeric(panel[column], errors="coerce").fillna(0.0)

    panel = panel.merge(played_team_weeks, on=["season", "week", "team"], how="left")
    # Drop byes and any roster week whose club did not play.
    panel = panel[panel["team_played"].eq(1)].drop(columns=["team_played"])

    panel = panel.merge(team_totals, on=["season", "week", "team"], how="left")
    opponents = _opponent_map(seasons)
    if not opponents.empty:
        opponents = opponents.astype({"season": "int64", "week": "int64"})
        panel = panel.merge(opponents, on=["season", "week", "team"], how="left")
    else:
        panel["opponent"] = pd.NA

    lines = _market_lines(seasons)
    if not lines.empty:
        lines = lines.astype({"season": "int64", "week": "int64"})
        panel = panel.merge(lines, on=["season", "week", "team"], how="left")
    else:
        for column in (
            "spread",
            "game_total",
            "implied_team_total",
            "implied_opponent_total",
        ):
            panel[column] = np.nan

    panel["player_key"] = panel["player_id"].astype(str)
    panel["scoring"] = scoring
    panel = panel.sort_values(["player_key", "season", "week"], kind="mergesort")
    return panel.reset_index(drop=True)


def team_games_remaining(panel: pd.DataFrame) -> pd.Series:
    """Games this player's club has left in the season, counting the current one.

    The rest-of-season response is a sum over a varying number of games, so the
    count is an offset the model is entitled to know: a schedule is public. It is
    derived from the panel's own team-weeks so a bye already removed cannot be
    counted as a game that will be played.
    """
    schedule = panel[["season", "week", "team"]].drop_duplicates()
    total = schedule.groupby(["season", "team"])["week"].transform("size")
    played_before = schedule.groupby(["season", "team"])["week"].rank(method="first") - 1
    schedule = schedule.assign(games_remaining=(total - played_before).astype(int))
    merged = panel[["season", "week", "team"]].merge(
        schedule, on=["season", "week", "team"], how="left"
    )
    return pd.Series(
        merged["games_remaining"].to_numpy(), index=panel.index, name="games_remaining"
    )


def load_panel(
    seasons: Iterable[int],
    *,
    cache: Path | None = None,
    scoring: str = "ppr",
    refresh: bool = False,
) -> pd.DataFrame:
    """Build the panel, or reuse a cached pickle of exactly these seasons."""
    seasons = sorted({int(s) for s in seasons})
    if cache is not None and cache.exists() and not refresh:
        stored = pd.read_pickle(cache)
        have = sorted(stored["season"].unique().tolist())
        if have == seasons and stored.attrs.get("scoring", scoring) == scoring:
            return stored
    panel = build_panel(seasons, scoring=scoring)
    panel.attrs["scoring"] = scoring
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        panel.to_pickle(cache)
    return panel
