"""The panel's zeros have to be real, and its byes have to be gone.

Both failures are silent. If a rostered player who did not play is dropped
instead of zeroed, the response quietly becomes "points given he played" and
every availability number downstream is measured on a population that cannot
contain an absence. If a bye week is zeroed instead of dropped, the model is
handed a large block of zeros that no one needed a forecast for, and both the
play rate and the rest-of-season sum are wrong.

These use a constructed feed rather than the cache so they run anywhere, and
they exercise :func:`build_panel`'s real join.
"""

import numpy as np
import pandas as pd
import pytest

from ffmodel.weekly import frame as frame_module
from ffmodel.weekly.frame import PANEL_POSITIONS, FIRST_PANEL_SEASON, build_panel


def _stub(monkeypatch, stats: pd.DataFrame, rosters: pd.DataFrame) -> None:
    monkeypatch.setattr(frame_module, "load_player_weeks", lambda seasons: stats.copy())
    monkeypatch.setattr(
        frame_module.ingest, "load_weekly_rosters", lambda seasons: rosters.copy()
    )
    monkeypatch.setattr(frame_module, "_opponent_map", lambda seasons: pd.DataFrame(
        columns=["season", "week", "team", "opponent"]
    ))


def _stat_row(week: int, player: str, team: str, **kwargs) -> dict:
    row = {
        "season": 2020,
        "week": week,
        "player_id": player,
        "player_name": player,
        "position": kwargs.pop("position", "WR"),
        "team": team,
        "pass_att": 0.0,
        "pass_cmp": 0.0,
        "pass_yds": 0.0,
        "pass_td": 0.0,
        "pass_int": 0.0,
        "rush_att": 0.0,
        "rush_yds": 0.0,
        "rush_td": 0.0,
        "targets": 0.0,
        "receptions": 0.0,
        "rec_yds": 0.0,
        "rec_td": 0.0,
        "fumbles_lost": 0.0,
    }
    row.update(kwargs)
    return row


def test_a_rostered_player_who_did_not_play_becomes_a_zero(monkeypatch) -> None:
    # Two players on ATL. Only one records a line in week 1.
    stats = pd.DataFrame(
        [_stat_row(1, "A", "ATL", receptions=5.0, rec_yds=60.0)]
    )
    rosters = pd.DataFrame(
        [
            {"season": 2020, "week": 1, "team": "ATL", "position": "WR",
             "gsis_id": name, "full_name": name, "status": "ACT", "game_type": "REG"}
            for name in ("A", "B")
        ]
    )
    _stub(monkeypatch, stats, rosters)
    panel = build_panel([2020])

    assert set(panel["player_id"]) == {"A", "B"}
    absent = panel[panel["player_id"] == "B"].iloc[0]
    assert absent["played"] == 0
    assert absent["points"] == 0.0
    present = panel[panel["player_id"] == "A"].iloc[0]
    assert present["played"] == 1
    # 5 receptions + 60 yards in PPR.
    assert present["points"] == pytest.approx(5.0 + 6.0)


def test_a_bye_week_is_dropped_rather_than_zeroed(monkeypatch) -> None:
    """ATL plays weeks 1 and 3; week 2 is its bye and must not appear."""
    stats = pd.DataFrame(
        [
            _stat_row(1, "A", "ATL", receptions=3.0),
            _stat_row(2, "C", "BUF", receptions=4.0),
            _stat_row(3, "A", "ATL", receptions=2.0),
        ]
    )
    rosters = pd.DataFrame(
        [
            {"season": 2020, "week": week, "team": "ATL", "position": "WR",
             "gsis_id": "A", "full_name": "A", "status": "ACT", "game_type": "REG"}
            for week in (1, 2, 3)
        ]
    )
    _stub(monkeypatch, stats, rosters)
    panel = build_panel([2020])

    weeks = sorted(panel[panel["player_id"] == "A"]["week"].tolist())
    assert weeks == [1, 3], "the bye week survived as a zero"


def test_a_practice_squad_elevation_still_reaches_the_panel(monkeypatch) -> None:
    """A player who recorded a line but is not on the contract list is real."""
    stats = pd.DataFrame([_stat_row(1, "E", "ATL", receptions=7.0)])
    rosters = pd.DataFrame(
        [
            {"season": 2020, "week": 1, "team": "ATL", "position": "WR",
             "gsis_id": "A", "full_name": "A", "status": "ACT", "game_type": "REG"},
            {"season": 2020, "week": 1, "team": "ATL", "position": "WR",
             "gsis_id": "E", "full_name": "E", "status": "DEV", "game_type": "REG"},
        ]
    )
    _stub(monkeypatch, stats, rosters)
    panel = build_panel([2020])

    elevated = panel[panel["player_id"] == "E"]
    assert len(elevated) == 1
    assert elevated.iloc[0]["played"] == 1
    assert elevated.iloc[0]["points"] == pytest.approx(7.0)


def test_team_totals_are_summed_from_the_same_rows(monkeypatch) -> None:
    stats = pd.DataFrame(
        [
            _stat_row(1, "A", "ATL", targets=6.0, receptions=4.0),
            _stat_row(1, "B", "ATL", targets=4.0, receptions=3.0),
        ]
    )
    rosters = pd.DataFrame(
        [
            {"season": 2020, "week": 1, "team": "ATL", "position": "WR",
             "gsis_id": name, "full_name": name, "status": "ACT", "game_type": "REG"}
            for name in ("A", "B")
        ]
    )
    _stub(monkeypatch, stats, rosters)
    panel = build_panel([2020])
    assert (panel["team_targets"] == 10.0).all()


def test_seasons_before_the_usable_window_are_refused() -> None:
    with pytest.raises(ValueError, match="2016"):
        build_panel([2013])


def test_only_fantasy_positions_are_kept(monkeypatch) -> None:
    stats = pd.DataFrame(
        [
            _stat_row(1, "K", "ATL", position="K"),
            _stat_row(1, "A", "ATL", position="WR", receptions=2.0),
        ]
    )
    rosters = pd.DataFrame(
        [
            {"season": 2020, "week": 1, "team": "ATL", "position": position,
             "gsis_id": name, "full_name": name, "status": "ACT", "game_type": "REG"}
            for name, position in (("K", "K"), ("A", "WR"))
        ]
    )
    _stub(monkeypatch, stats, rosters)
    panel = build_panel([2020])
    assert set(panel["position"]) <= set(PANEL_POSITIONS)
