"""The partial-game flag, on constructed weeks where the answer is known."""

from __future__ import annotations

import pandas as pd
import pytest

from ffmodel.evaluation.partial_games import (
    MIN_GAMES_FOR_MEDIAN,
    MIN_MEDIAN_SHARE,
    per_game_rates,
)


def _weekly(rows):
    frame = pd.DataFrame(rows)
    for column in (
        "pass_yds",
        "pass_td",
        "pass_int",
        "rush_yds",
        "rush_td",
        "rec_yds",
        "rec_td",
        "receptions",
        "fumbles_lost",
    ):
        frame[column] = frame.get(column, 0.0)
    if "team" not in frame:
        frame["team"] = "AAA"
    frame["team"] = frame["team"].fillna("AAA")
    frame["position"] = frame.get("position", "WR")
    return frame


def _rates(weeks, shares):
    return per_game_rates(
        [2024],
        load_weekly=lambda years: _weekly(weeks),
        load_snaps=lambda years: pd.DataFrame(shares),
        load_ids=lambda: pd.DataFrame(
            {"pfr_id": ["P1", "P2"], "gsis_id": ["G1", "G2"]}
        ),
    )


def _week(player, week, points, share, pfr="P1"):
    return (
        {
            "player_id": player,
            "season": 2024,
            "week": week,
            "rec_yds": points * 10.0,
        },
        {
            "pfr_player_id": pfr,
            "season": 2024,
            "week": week,
            "offense_pct": share,
            "game_type": "REG",
        },
    )


def test_a_week_under_half_the_player_s_own_median_is_partial():
    pairs = [_week("G1", w, 10.0, 0.80) for w in range(1, 9)]
    pairs.append(_week("G1", 9, 1.0, 0.15))
    weeks, shares = zip(*pairs)
    out = _rates(list(weeks), list(shares)).set_index("player_id").loc["G1"]

    assert out["weeks"] == 9
    assert out["partial_weeks"] == 1
    assert out["full_weeks"] == 8
    # The one bad week drags the raw rate down and is absent from the clean one.
    # 100 receiving yards is 10 PPR points, so the eight full weeks average 10.
    assert out["clean_ppg"] > out["raw_ppg"]
    assert out["clean_ppg"] == pytest.approx(10.0)


def test_the_median_is_the_player_s_own_not_the_population_s():
    """A rotational player at his normal share is not partial."""
    pairs = [_week("G1", w, 10.0, 0.80) for w in range(1, 7)]
    pairs += [_week("G2", w, 3.0, 0.30, pfr="P2") for w in range(1, 7)]
    weeks, shares = zip(*pairs)
    out = _rates(list(weeks), list(shares)).set_index("player_id")

    # 0.30 is well under the starter's median and under half of it, but it is
    # this player's own normal week.
    assert out.loc["G2", "partial_weeks"] == 0
    assert out.loc["G2", "clean_ppg"] == pytest.approx(out.loc["G2", "raw_ppg"])


def test_no_flags_below_the_median_share_floor():
    """A player with no offensive role has no 'half his usual role'.

    Without the floor the rule inverts: a median of zero puts every week under
    the absolute minimum and the whole season reads as partial.
    """
    share = MIN_MEDIAN_SHARE / 2
    pairs = [_week("G1", w, 0.0, share) for w in range(1, 9)]
    pairs.append(_week("G1", 9, 0.0, 0.0))
    weeks, shares = zip(*pairs)
    out = _rates(list(weeks), list(shares)).set_index("player_id").loc["G1"]

    assert out["partial_weeks"] == 0
    assert out["full_weeks"] == 9


def test_no_flags_when_the_median_rests_on_too_few_games():
    pairs = [_week("G1", w, 10.0, 0.80) for w in range(1, MIN_GAMES_FOR_MEDIAN - 1)]
    pairs.append(_week("G1", MIN_GAMES_FOR_MEDIAN - 1, 1.0, 0.05))
    weeks, shares = zip(*pairs)
    out = _rates(list(weeks), list(shares)).set_index("player_id").loc["G1"]

    assert out["weeks"] == MIN_GAMES_FOR_MEDIAN - 1
    assert out["partial_weeks"] == 0


def test_a_missing_snap_row_is_not_evidence_of_an_early_exit():
    pairs = [_week("G1", w, 10.0, 0.80) for w in range(1, 9)]
    weeks, shares = zip(*pairs)
    weeks = list(weeks) + [_week("G1", 9, 10.0, 0.80)[0]]
    out = _rates(weeks, list(shares)).set_index("player_id").loc["G1"]

    assert out["weeks"] == 9
    assert out["partial_weeks"] == 0
    assert out["full_weeks"] == 9


def test_at_most_half_a_season_can_be_partial():
    """The threshold is a fraction of the median, so it cannot swallow a season.

    This is what makes ``clean_ppg`` always defined. A season split evenly
    between a full role and a token one is the worst case: the median sits at
    the boundary and only the strictly-lower half is flagged.
    """
    pairs = [_week("G1", w, 10.0, 0.80) for w in range(1, 7)]
    pairs += [_week("G1", w, 1.0, 0.10) for w in range(7, 13)]
    weeks, shares = zip(*pairs)
    out = _rates(list(weeks), list(shares)).set_index("player_id").loc["G1"]

    assert out["weeks"] == 12
    assert out["partial_weeks"] <= out["weeks"] / 2
    assert out["full_weeks"] > 0
    assert pd.notna(out["clean_ppg"])


def test_traded_players_are_counted_so_a_caller_can_exclude_them():
    pairs = [_week("G1", w, 10.0, 0.80) for w in range(1, 9)]
    weeks, shares = zip(*pairs)
    weeks = list(weeks)
    for row in weeks[4:]:
        row["team"] = "BBB"
    out = _rates(weeks, list(shares)).set_index("player_id").loc["G1"]
    assert out["teams"] == 2
