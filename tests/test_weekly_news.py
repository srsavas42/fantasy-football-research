"""The pre-game signals, whose conventions are all easy to invert silently.

Three things here would produce a plausible-looking model if they were wrong.

An absent injury row means *healthy*, not unknown -- the report lists only players
with something to declare, so filling those rows with a median or a NaN would
turn "nothing to report" into "average injury severity" on 92% of the panel.

``depth_promoted`` is positive when a player moves *up*, which means it is the
lagged rank minus the current one, not the other way round. Backwards, the model
learns that being demoted increases workload.

``ahead_out`` must fire for the players *behind* the injured one and not for the
injured player himself, must respect club and position boundaries, and must
compare against the best-ranked absentee rather than any absentee.
"""

import numpy as np
import pandas as pd
import pytest

from ffmodel.weekly import news as news_module
from ffmodel.weekly.news import (
    PRACTICE_SEVERITY,
    STATUS_SEVERITY,
    _ahead_out,
    add_news_features,
)


def _panel() -> pd.DataFrame:
    rows = []
    for week in (1, 2):
        for key, position, team in (
            ("starter", "RB", "ATL"),
            ("backup", "RB", "ATL"),
            ("third", "RB", "ATL"),
            ("receiver", "WR", "ATL"),
            ("elsewhere", "RB", "BUF"),
        ):
            rows.append(
                {
                    "season": 2023,
                    "week": week,
                    "team": team,
                    "player_key": key,
                    "position": position,
                    "played": 1,
                    "points": 10.0,
                }
            )
    return pd.DataFrame(rows)


def _stub(monkeypatch, injuries: pd.DataFrame, depth: pd.DataFrame) -> None:
    monkeypatch.setattr(news_module, "load_injury_report", lambda seasons: injuries)
    monkeypatch.setattr(news_module, "load_depth", lambda seasons: depth)


def test_status_severity_is_ordered_worst_first() -> None:
    assert STATUS_SEVERITY["Out"] > STATUS_SEVERITY["Doubtful"]
    assert STATUS_SEVERITY["Doubtful"] > STATUS_SEVERITY["Questionable"]
    assert PRACTICE_SEVERITY["Did Not Participate In Practice"] > PRACTICE_SEVERITY[
        "Full Participation in Practice"
    ]


def test_not_being_on_the_report_means_healthy(monkeypatch) -> None:
    injuries = pd.DataFrame(
        [{"season": 2023, "week": 1, "player_key": "starter",
          "inj_status": 3.0, "inj_practice": 2.0}]
    )
    _stub(monkeypatch, injuries, pd.DataFrame(columns=["season", "week", "player_key", "depth_rank"]))
    out = add_news_features(_panel()).set_index(["player_key", "week"])

    assert out.loc[("starter", 1), "inj_status"] == 3.0
    assert out.loc[("starter", 1), "inj_out"] == 1.0
    # Everyone else is healthy, and healthy is zero rather than missing.
    assert out.loc[("backup", 1), "inj_status"] == 0.0
    assert out["inj_status"].notna().all()
    assert out.loc[("starter", 2), "inj_status"] == 0.0


def test_promotion_is_positive_when_the_rank_improves(monkeypatch) -> None:
    depth = pd.DataFrame(
        [
            {"season": 2023, "week": 1, "player_key": "backup", "depth_rank": 2.0},
            {"season": 2023, "week": 2, "player_key": "backup", "depth_rank": 1.0},
            {"season": 2023, "week": 1, "player_key": "third", "depth_rank": 2.0},
            {"season": 2023, "week": 2, "player_key": "third", "depth_rank": 3.0},
        ]
    )
    _stub(monkeypatch, pd.DataFrame(columns=["season", "week", "player_key", "inj_status", "inj_practice"]), depth)
    out = add_news_features(_panel()).set_index(["player_key", "week"])

    # 2 -> 1 is a promotion and must read positive.
    assert out.loc[("backup", 2), "depth_promoted"] == pytest.approx(1.0)
    # 2 -> 3 is a demotion and must read negative.
    assert out.loc[("third", 2), "depth_promoted"] == pytest.approx(-1.0)
    # Week 1 has nothing to compare against.
    assert out.loc[("backup", 1), "depth_promoted"] == 0.0


def test_ahead_out_fires_only_for_players_behind_the_absentee() -> None:
    frame = pd.DataFrame(
        {
            "season": [2023] * 5,
            "week": [1] * 5,
            "team": ["ATL", "ATL", "ATL", "ATL", "BUF"],
            "position": ["RB", "RB", "RB", "WR", "RB"],
            "rank": [1.0, 2.0, 3.0, 2.0, 3.0],
            "status": [3.0, 0.0, 0.0, 0.0, 0.0],
        }
    )
    got = _ahead_out(frame, "rank", "status").to_numpy()
    # The out player himself: no. The two backs behind him: yes.
    assert list(got) == [0.0, 1.0, 1.0, 0.0, 0.0]


def test_ahead_out_respects_position_and_club() -> None:
    """A hurt receiver does not promote a running back, or a rival's back."""
    frame = pd.DataFrame(
        {
            "season": [2023] * 3,
            "week": [1] * 3,
            "team": ["ATL", "ATL", "BUF"],
            "position": ["WR", "RB", "RB"],
            "rank": [1.0, 2.0, 2.0],
            "status": [3.0, 0.0, 0.0],
        }
    )
    assert not _ahead_out(frame, "rank", "status").any()


def test_a_questionable_teammate_does_not_count_as_out() -> None:
    """Only Doubtful or worse clears the bar; Questionable plays two thirds of the time."""
    frame = pd.DataFrame(
        {
            "season": [2023, 2023],
            "week": [1, 1],
            "team": ["ATL", "ATL"],
            "position": ["RB", "RB"],
            "rank": [1.0, 2.0],
            "status": [STATUS_SEVERITY["Questionable"], 0.0],
        }
    )
    assert not _ahead_out(frame, "rank", "status").any()


def test_the_most_severe_report_in_a_week_wins(monkeypatch) -> None:
    """A status updated mid-week must not average into something milder."""
    raw = pd.DataFrame(
        [
            {"season": 2023, "game_type": "REG", "week": 1, "gsis_id": "starter",
             "report_status": "Questionable",
             "practice_status": "Limited Participation in Practice"},
            {"season": 2023, "game_type": "REG", "week": 1, "gsis_id": "starter",
             "report_status": "Out",
             "practice_status": "Did Not Participate In Practice"},
        ]
    )
    monkeypatch.setattr(news_module.ingest, "load_injuries", lambda seasons: raw)
    got = news_module.load_injury_report([2023]).set_index("player_key")
    assert got.loc["starter", "inj_status"] == STATUS_SEVERITY["Out"]
    assert len(got) == 1


def test_missing_feeds_degrade_to_neutral(monkeypatch) -> None:
    """No feed must mean no signal, not a crash and not a fabricated one."""
    empty_i = pd.DataFrame(columns=["season", "week", "player_key", "inj_status", "inj_practice"])
    empty_d = pd.DataFrame(columns=["season", "week", "player_key", "depth_rank"])
    _stub(monkeypatch, empty_i, empty_d)
    out = add_news_features(_panel())
    assert (out["inj_status"] == 0.0).all()
    assert (out["ahead_out"] == 0.0).all()
    assert out["depth_rank"].isna().all()
