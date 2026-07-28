"""Leakage-safe injury-history features for season availability."""

import numpy as np
import pandas as pd

from ffmodel.features.season_injury import (
    add_season_injury_features,
    build_injury_episodes,
    load_live_injury_snapshot,
    normalise_injury_reports,
)


def _injury(season, week, player, status, body, *, game_type="REG"):
    return {
        "season": season,
        "week": week,
        "game_type": game_type,
        "team": "BUF",
        "gsis_id": player,
        "full_name": player,
        "position": "RB",
        "report_status": status,
        "practice_status": "Did Not Participate In Practice",
        "report_primary_injury": body,
        "practice_primary_injury": body,
    }


def _roster(season, week, player, status, *, game_type="REG"):
    return {
        "season": season,
        "week": week,
        "game_type": game_type,
        "team": "BUF",
        "gsis_id": player,
        "full_name": player,
        "position": "RB",
        "status": status,
    }


def _history():
    injuries = pd.DataFrame(
        [
            _injury(2021, 1, "p1", "Out", "Ankle"),
            _injury(2022, 2, "p1", "Out", "Knee"),
        ]
    )
    rosters = pd.DataFrame(
        [
            _roster(2021, 1, "p1", "INA"),
            _roster(2021, 2, "p1", "INA"),
            _roster(2021, 3, "p1", "ACT"),
            _roster(2022, 1, "p1", "ACT"),
            _roster(2022, 2, "p1", "INA"),
            _roster(2022, 3, "p1", "ACT"),
        ]
    )
    return injuries, rosters


def test_injury_episodes_use_regular_season_roster_return_signal():
    injuries, rosters = _history()
    injuries = pd.concat(
        [
            injuries,
            pd.DataFrame([_injury(2021, 19, "p1", "Out", "Ankle", game_type="WC")]),
        ],
        ignore_index=True,
    )
    rosters = pd.concat(
        [
            rosters,
            pd.DataFrame([_roster(2021, 19, "p1", "INA", game_type="WC")]),
        ],
        ignore_index=True,
    )

    episodes = build_injury_episodes(injuries, rosters)
    first = episodes.loc[episodes["season"].eq(2021)].iloc[0]

    assert first["episode_start_week"] == 1
    assert first["episode_end_week"] == 1
    assert first["recovery_weeks"] == 2
    assert first["recovery_censored"] == 0


def test_live_reserve_statuses_are_availability_relevant_injuries():
    live_snapshot = pd.DataFrame(
        {
            "season": [2026, 2026, 2026],
            "week": [1, 1, 1],
            "team": ["BUF", "BUF", "BUF"],
            "gsis_id": ["ir", "pup", "sus"],
            "player_name": ["IR back", "PUP back", "Suspended back"],
            "position": ["RB", "RB", "RB"],
            "injury_status": ["IR", "PUP", "Sus"],
            "injury_body_part": ["Knee", "Foot", "Suspension"],
        }
    )

    normalized = normalise_injury_reports(live_snapshot).set_index("player_key")

    assert set(normalized.index) == {"ir", "pup"}
    assert normalized.loc["ir", "injury_severity"] == 3
    assert normalized.loc["pup", "injury_severity"] == 3


def test_live_snapshot_is_labeled_with_the_projection_cutoff(monkeypatch):
    from ffmodel.data import sleeper

    captured = {}

    def fake_load_players(**kwargs):
        captured.update(kwargs)
        return pd.DataFrame({"gsis_id": ["p1"], "position": ["RB"]})

    monkeypatch.setattr(sleeper, "load_players", fake_load_players)

    snapshot = load_live_injury_snapshot(
        2026,
        cutoff_week=2,
        snapshot_at="2026-08-30",
        refresh=True,
    )

    assert captured == {
        "snapshot_at": "2026-08-30",
        "refresh": True,
        "cache_dir": None,
    }
    assert snapshot["season"].tolist() == [2026]
    assert snapshot["week"].tolist() == [2]


def test_injury_features_are_temporal_and_use_current_snapshot():
    injuries, rosters = _history()
    rows = pd.DataFrame(
        {
            "season": [2022, 2023, 2023],
            "team": ["BUF", "BUF", "BUF"],
            "player_key": ["p1", "p1", "p2"],
            "position": ["RB", "RB", "RB"],
        }
    )
    snapshot = pd.DataFrame(
        {
            "season": [2023],
            "week": [1],
            "team": ["BUF"],
            "gsis_id": ["p1"],
            "player_name": ["p1"],
            "position": ["RB"],
            "injury_status": ["Out"],
            "injury_body_part": ["Ankle"],
            "practice_participation": ["Did Not Participate In Practice"],
        }
    )

    featured = add_season_injury_features(
        rows,
        injuries=injuries,
        weekly_rosters=rosters,
        injury_snapshot=snapshot,
    ).set_index(["season", "player_key"])

    prior = featured.loc[(2022, "p1")]
    current = featured.loc[(2023, "p1")]
    healthy = featured.loc[(2023, "p2")]
    assert prior["injury_history_available"] == 1
    assert prior["prior_injury_report_weeks_3yr"] == 1
    assert prior["prior_injury_mean_recovery_weeks_3yr"] == 2
    assert prior["current_injury_snapshot_available"] == 0
    assert current["prior_injury_report_weeks_3yr"] == 2
    assert current["prior_injury_episode_count_3yr"] == 2
    assert current["current_injury_snapshot_available"] == 1
    assert current["current_injury_severity"] == 3
    assert np.isclose(current["current_injury_expected_recovery_weeks"], 1.5)
    assert healthy["current_injury_snapshot_available"] == 1
    assert healthy["current_injury_reported"] == 0

    future_injuries = pd.concat(
        [injuries, pd.DataFrame([_injury(2024, 1, "p1", "Out", "Ankle")])],
        ignore_index=True,
    )
    future_rosters = pd.concat(
        [
            rosters,
            pd.DataFrame(
                [
                    _roster(2024, 1, "p1", "INA"),
                    _roster(2024, 2, "p1", "INA"),
                    _roster(2024, 3, "p1", "INA"),
                    _roster(2024, 4, "p1", "INA"),
                    _roster(2024, 5, "p1", "ACT"),
                ]
            ),
        ],
        ignore_index=True,
    )
    with_future = add_season_injury_features(
        rows,
        injuries=future_injuries,
        weekly_rosters=future_rosters,
        injury_snapshot=snapshot,
    ).set_index(["season", "player_key"])

    assert np.isclose(
        with_future.loc[(2023, "p1"), "current_injury_expected_recovery_weeks"],
        current["current_injury_expected_recovery_weeks"],
    )
