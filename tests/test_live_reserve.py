"""The reserve-list floor: counted in club games, and only ever a floor."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import project_live  # noqa: E402


def _roster(rows):
    return pd.DataFrame(rows, columns=["player_id", "week", "team", "position", "status"])


def _schedule(team, weeks):
    return pd.DataFrame({"season": 2026, "week": weeks, "team": team, "opponent": "XXX"})


def test_a_fresh_placement_is_out_for_four_games(monkeypatch):
    roster = _roster([("a", w, "NYG", "QB", "ACT" if w < 3 else "RES") for w in (1, 2, 3, 4)])
    monkeypatch.setattr(project_live, "_roster_weeks", lambda seasons: roster)
    got = project_live.reserve_out_through(2026, 4, _schedule("NYG", list(range(1, 19))))
    # Placed in week 3, so his four games are weeks 3, 4, 5, 6.
    assert got["a"] == 6


def test_a_bye_inside_the_window_pushes_the_return_out_a_week(monkeypatch):
    """Weeks are not games. Counting them as such lets a player back early."""
    roster = _roster([("a", w, "NYG", "QB", "ACT" if w < 3 else "RES") for w in (1, 2, 3, 4)])
    monkeypatch.setattr(project_live, "_roster_weeks", lambda seasons: roster)
    weeks = [w for w in range(1, 19) if w != 5]  # bye in week 5
    got = project_live.reserve_out_through(2026, 4, _schedule("NYG", weeks))
    assert got["a"] == 7


def test_a_long_stint_is_only_bounded_through_the_current_week(monkeypatch):
    """On reserve since the opener, the minimum is long served: the floor is
    this week and no further, because nothing here says when he returns."""
    roster = _roster([("a", w, "NYG", "QB", "RES") for w in (1, 2, 3, 4, 5, 6)])
    monkeypatch.setattr(project_live, "_roster_weeks", lambda seasons: roster)
    got = project_live.reserve_out_through(2026, 6, _schedule("NYG", list(range(1, 19))))
    # Placed week 1, so his four games ended in week 4; the floor is the target week.
    assert got["a"] == 6


def test_a_player_not_on_reserve_gets_no_floor(monkeypatch):
    roster = _roster([("a", w, "NYG", "QB", "ACT") for w in (1, 2, 3, 4)])
    monkeypatch.setattr(project_live, "_roster_weeks", lambda seasons: roster)
    got = project_live.reserve_out_through(2026, 4, _schedule("NYG", list(range(1, 19))))
    assert "a" not in got.index
