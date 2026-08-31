"""A suspension is arithmetic, not a hazard, and the two encodings must agree.

The failures these guard against are all silent. A reader that knows only the
post-2020 reason codes loses every pre-2020 ban and reports a clean history; a
ban that shrinks the share denominator along with the exposure holds season
totals flat while looking like it worked; a misspelled override name reads as
"no suspension" and projects a banned player at full health. None of that
raises on its own.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_spec = importlib.util.spec_from_file_location(
    "project_season",
    Path(__file__).resolve().parents[1] / "scripts" / "project_season.py",
)
_project_season = importlib.util.module_from_spec(_spec)
sys.modules["project_season"] = _project_season
_spec.loader.exec_module(_project_season)
apply_suspension_overrides = _project_season.apply_suspension_overrides

from ffmodel.features.suspensions import (
    classify_suspension,
    exempt_duration_table,
    preseason_suspension_games,
    suspension_spells,
)
from ffmodel.features.suspensions import classify_reserve, mandatory_missed_games
from ffmodel.models.season_availability import _eligible_games, _playable_games


def _rosters(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "season": r.get("season", 2023),
                "week": r["week"],
                "team": r.get("team", "AAA"),
                "position": r.get("position", "RB"),
                "full_name": r["full_name"],
                "status": r.get("status", "ACT"),
                "status_description_abbr": r.get("code", "A01"),
                "game_type": "REG",
            }
            for r in rows
        ]
    )


def test_both_encodings_resolve_to_the_same_label():
    """``SUS`` before 2020 and ``RES``/``R40`` after are one event."""
    frame = _rosters(
        [
            {"season": 2018, "week": 1, "full_name": "Old", "status": "SUS"},
            {"season": 2023, "week": 1, "full_name": "New", "status": "RES", "code": "R40"},
        ]
    )
    assert list(classify_suspension(frame)) == ["definite", "definite"]


def test_indefinite_and_exempt_are_not_pooled_with_definite():
    frame = _rosters(
        [
            {"week": 1, "full_name": "Banned", "status": "RES", "code": "R40"},
            {"week": 1, "full_name": "Open", "status": "RES", "code": "R30"},
            {"week": 1, "full_name": "Exempt", "status": "EXE", "code": "E02"},
        ]
    )
    assert list(classify_suspension(frame)) == ["definite", "indefinite", "exempt"]


def test_international_pathway_is_not_a_suspension():
    """``E14`` reads like an exempt code and is not a disciplinary action."""
    frame = _rosters(
        [{"week": 1, "full_name": "Pathway", "status": "E14", "code": "E14"}]
    )
    assert classify_suspension(frame).isna().all()


def test_injured_reserve_is_not_a_suspension():
    frame = _rosters(
        [{"week": 1, "full_name": "Hurt", "status": "RES", "code": "R01"}]
    )
    assert classify_suspension(frame).isna().all()


def test_preseason_ban_returns_its_announced_length():
    frame = _rosters(
        [
            {"week": w, "full_name": "Banned", "status": "RES", "code": "R40"}
            for w in range(1, 5)
        ]
        + [
            {"week": w, "full_name": "Banned", "status": "ACT"}
            for w in range(5, 19)
        ]
    )
    out = preseason_suspension_games(frame)
    assert out.loc[0, "suspended_games"] == 4


def test_a_midseason_ban_is_not_reported_as_preseason_known():
    """Nobody knew in August, so it must not enter an August projection."""
    frame = _rosters(
        [{"week": w, "full_name": "Banned", "status": "ACT"} for w in range(1, 8)]
        + [
            {"week": w, "full_name": "Banned", "status": "RES", "code": "R40"}
            for w in range(8, 10)
        ]
    )
    assert preseason_suspension_games(frame).empty
    spells = suspension_spells(frame)
    assert not bool(spells.loc[0, "preseason_known"])


def test_an_indefinite_ban_reports_its_censoring_rather_than_hiding_it():
    """The Ridley case: flagged weeks alone understate the ban.

    A player banned indefinitely comes off the roster entirely rather than
    sitting on a reserve list, so the weeks he is absent are part of the same
    absence. Reporting only ``flagged_weeks`` would call a lost season a
    nine-game ban.
    """
    frame = _rosters(
        [
            {"season": 2022, "week": w, "full_name": "Gone", "status": "RES", "code": "R30"}
            for w in range(10, 19)
        ]
    )
    spell = suspension_spells(frame).iloc[0]
    assert spell["flagged_weeks"] == 9
    assert spell["roster_absent_weeks"] == 9
    assert bool(spell["censored"])


def test_exempt_table_drops_the_covid_season_by_default():
    frame = _rosters(
        [
            {"season": 2020, "week": 9, "full_name": "Covid", "status": "EXE", "code": "E02"},
            {"season": 2023, "week": 8, "full_name": "Real", "status": "EXE", "code": "E02"},
        ]
    )
    assert list(exempt_duration_table(frame)["player_name"]) == ["Real"]
    both = exempt_duration_table(frame, drop_covid_season=False)
    assert set(both["player_name"]) == {"Covid", "Real"}


def test_a_ban_shortens_the_exposure_and_leaves_the_denominator_alone():
    """The share layers divide by ``team_games``; only eligibility moves."""
    rows = pd.DataFrame({"suspended_games": [0.0, 4.0, 17.0]})
    team_games = np.array([17, 17, 17])
    assert list(_eligible_games(rows, team_games)) == [17, 13, 0]


def test_a_frame_without_the_column_is_unchanged():
    """Every artifact fitted before this existed must reproduce exactly."""
    rows = pd.DataFrame({"season": [2023, 2023]})
    assert list(_eligible_games(rows, np.array([17, 17]))) == [17, 17]


def test_a_ban_longer_than_the_season_cannot_go_negative():
    rows = pd.DataFrame({"suspended_games": [25.0]})
    assert list(_eligible_games(rows, np.array([17]))) == [0]


def test_a_negative_ban_is_rejected():
    rows = pd.DataFrame({"suspended_games": [-1.0]})
    with pytest.raises(ValueError, match="nonnegative"):
        _eligible_games(rows, np.array([17]))


def _projection_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "player_name": ["Josh Jacobs", "Bijan Robinson", "Josh Jacobs"],
            "team": ["GB", "ATL", "LV"],
            "suspended_games": [0.0, 0.0, 0.0],
        }
    )


def test_an_override_beats_the_feed(tmp_path):
    path = tmp_path / "susp.csv"
    path.write_text("player_name,team,suspended_games\nJosh Jacobs,GB,4\n")
    out = apply_suspension_overrides(_projection_rows(), path)
    assert list(out["suspended_games"]) == [4.0, 0.0, 0.0]


def test_a_misspelled_name_is_an_error_not_a_silent_pass(tmp_path):
    """Otherwise the ban vanishes and a banned player projects at full health."""
    path = tmp_path / "susp.csv"
    path.write_text("player_name,suspended_games\nJosh Jacobsen,4\n")
    with pytest.raises(SystemExit, match="matched 0 rows"):
        apply_suspension_overrides(_projection_rows(), path)


def test_an_ambiguous_name_must_be_disambiguated(tmp_path):
    path = tmp_path / "susp.csv"
    path.write_text("player_name,suspended_games\nJosh Jacobs,4\n")
    with pytest.raises(SystemExit, match="matched 2 rows"):
        apply_suspension_overrides(_projection_rows(), path)


def test_pup_and_nfi_are_not_suspensions():
    """A floor is not a sentence and must not be read as one."""
    frame = _rosters(
        [
            {"week": 1, "full_name": "Pup", "status": "RES", "code": "R04"},
            {"week": 1, "full_name": "Nfi", "status": "RES", "code": "R05"},
            {"week": 1, "full_name": "Hurt", "status": "RES", "code": "R01"},
        ]
    )
    assert classify_suspension(frame).isna().all()
    assert list(classify_reserve(frame)) == ["pup", "nfi", "injured_reserve"]


def test_the_mandatory_minimum_follows_the_rule_of_its_season():
    """Four games from 2022, six before; the feed reproduces both."""
    seasons = pd.Series([2019, 2021, 2022, 2025])
    assert list(mandatory_missed_games(seasons)) == [6.0, 6.0, 4.0, 4.0]


def test_a_pup_placement_caps_games_rather_than_subtracting_them():
    """Subtracting would charge the player twice for one injury.

    The reserve coefficient is already fitted on players in exactly this
    position, so the fitted mean is close to right. Only the games the rule
    forbids should be removed.
    """
    rows = pd.DataFrame({"mandatory_missed_games": [0.0, 4.0, 6.0]})
    assert list(_playable_games(rows, np.array([17, 17, 17]))) == [17, 13, 11]


def test_a_frame_without_the_mandatory_column_is_uncapped():
    rows = pd.DataFrame({"season": [2025, 2025]})
    assert list(_playable_games(rows, np.array([17, 17]))) == [17, 17]


def test_a_negative_mandatory_minimum_is_rejected():
    with pytest.raises(ValueError, match="nonnegative"):
        _playable_games(pd.DataFrame({"mandatory_missed_games": [-1.0]}), np.array([17]))


def test_a_ban_is_not_served_while_on_pup():
    """A player must be active to serve a suspension, so the two are additive.

    Mike Woods in 2023 is the pattern: eleven weeks on the non-football-injury
    list, then a six-game ban in weeks 12-18. Reading the absences as
    overlapping would give him eleven games lost against an actual seventeen.
    """
    rows = pd.DataFrame(
        {"suspended_games": [6.0, 6.0, 0.0], "mandatory_missed_games": [4.0, 0.0, 4.0]}
    )
    assert list(_playable_games(rows, np.array([17, 17, 17]))) == [7, 11, 13]


def test_a_combined_absence_longer_than_the_season_floors_at_zero():
    rows = pd.DataFrame({"suspended_games": [12.0], "mandatory_missed_games": [6.0]})
    assert list(_playable_games(rows, np.array([17]))) == [0]
