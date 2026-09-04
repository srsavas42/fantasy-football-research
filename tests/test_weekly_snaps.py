"""The snap-count join, and the trap in tuning a feature the population reads.

The join crosses an id boundary: snap counts are keyed on Pro-Football-Reference
ids and everything else here on gsis ids. A bridge that silently drops most rows
would show up only as a feature that mysteriously fails to help.

The second test pins a methodological bug rather than a code path, because it
reversed a real conclusion. ``relevant_population`` reads a feature built at the
half-life under test, so evaluating candidates on "their own" relevant rows
compares different populations -- and on this panel that turned the answer from
"one game" into "eight games".
"""

import numpy as np
import pandas as pd
import pytest

from ffmodel.weekly import frame as frame_module
from ffmodel.weekly.features import add_features, relevant_population


def _stub_snaps(monkeypatch, rosters, snaps) -> None:
    monkeypatch.setattr(frame_module.ingest, "load_weekly_rosters", lambda s: rosters)
    monkeypatch.setattr(frame_module.ingest, "load_snap_counts", lambda s: snaps)


def test_the_bridge_translates_pfr_ids_to_gsis_ids(monkeypatch) -> None:
    rosters = pd.DataFrame(
        [{"gsis_id": "00-001", "pfr_id": "AaaB00", "season": 2020, "week": 1}]
    )
    snaps = pd.DataFrame(
        [
            {
                "season": 2020, "week": 1, "game_type": "REG", "position": "RB",
                "pfr_player_id": "AaaB00", "offense_pct": 0.72, "offense_snaps": 40.0,
            }
        ]
    )
    _stub_snaps(monkeypatch, rosters, snaps)
    got = frame_module._snap_counts([2020])
    assert len(got) == 1
    assert got.iloc[0]["player_key"] == "00-001"
    assert got.iloc[0]["snap_share"] == pytest.approx(0.72)


def test_the_bridge_is_pooled_across_seasons(monkeypatch) -> None:
    """A player missing from one season's roster is still resolvable."""
    rosters = pd.DataFrame(
        [{"gsis_id": "00-001", "pfr_id": "AaaB00", "season": 2019, "week": 3}]
    )
    snaps = pd.DataFrame(
        [
            {
                "season": 2020, "week": 1, "game_type": "REG", "position": "WR",
                "pfr_player_id": "AaaB00", "offense_pct": 0.5, "offense_snaps": 30.0,
            }
        ]
    )
    _stub_snaps(monkeypatch, rosters, snaps)
    got = frame_module._snap_counts([2019, 2020])
    assert len(got) == 1, "a cross-season identifier match was dropped"


def test_an_unmatched_snap_row_is_dropped_not_mislabelled(monkeypatch) -> None:
    rosters = pd.DataFrame(
        [{"gsis_id": "00-001", "pfr_id": "AaaB00", "season": 2020, "week": 1}]
    )
    snaps = pd.DataFrame(
        [
            {
                "season": 2020, "week": 1, "game_type": "REG", "position": "RB",
                "pfr_player_id": "ZzzZ99", "offense_pct": 0.9, "offense_snaps": 50.0,
            }
        ]
    )
    _stub_snaps(monkeypatch, rosters, snaps)
    assert frame_module._snap_counts([2020]).empty


def _panel(weeks: int = 10) -> pd.DataFrame:
    rng = np.random.default_rng(3)
    rows = []
    for player in range(4):
        for week in range(1, weeks + 1):
            rows.append(
                {
                    "player_key": f"P{player}", "player_id": f"P{player}",
                    "player_name": f"Player {player}", "season": 2021, "week": week,
                    "team": f"T{player % 2}", "opponent": f"T{(player + 1) % 2}",
                    "position": ["QB", "RB", "WR", "TE"][player], "played": 1,
                    "points": float(rng.integers(3, 25)),
                    "targets": float(rng.integers(0, 10)),
                    "rush_att": float(rng.integers(0, 15)), "pass_att": 0.0,
                    "receptions": 0.0, "rush_yds": 40.0, "rec_yds": 40.0,
                    "rush_epa": 0.0, "rec_epa": 0.0,
                    "snap_share": float(rng.uniform(0.3, 0.95)),
                    "team_targets": 30.0, "team_rush_att": 25.0, "team_plays": 60.0,
                    "team_points": 40.0, "team_pass_att": 35.0,
                }
            )
    return pd.DataFrame(rows)


def test_the_last_observation_feature_is_the_previous_played_week() -> None:
    """An alpha of one collapses the average onto the most recent value."""
    frame = add_features(_panel()).sort_values(["player_key", "week"])
    for _, block in frame.groupby("player_key"):
        points = block["points"].to_numpy(float)
        got = block["prior_points_last"].to_numpy(float)
        assert np.isnan(got[0])
        np.testing.assert_allclose(got[1:], points[:-1])


def test_the_population_moves_with_the_half_life() -> None:
    """Why a sweep must fix the population before comparing candidates.

    This is the bug that reversed the decay result: the filter reads a feature
    built at the half-life under test, so each candidate would otherwise be
    scored on a different set of rows.
    """
    panel = _panel(weeks=12)
    # A player who oscillates across the 4-point threshold. Smoothed hard he sits
    # steadily above it; tracking the last game he crosses it every week, so
    # which rows are "relevant" depends on the parameter being swept.
    swing = panel[panel["player_key"].eq("P0")].copy()
    swing["player_key"] = "P4"
    swing["player_id"] = "P4"
    swing["points"] = [1.0, 8.0] * (len(swing) // 2)
    panel = pd.concat([panel, swing], ignore_index=True)

    short = relevant_population(add_features(panel, halflife=0.5))
    long = relevant_population(add_features(panel, halflife=8.0))
    assert int(short.sum()) != int(long.sum()), (
        "the relevant population is expected to depend on the half-life; if this "
        "ever stops being true the sweep's population guard can be simplified"
    )
