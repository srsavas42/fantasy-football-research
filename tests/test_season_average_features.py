"""Season-average feature construction and leakage boundaries."""

import numpy as np
import pandas as pd

from ffmodel.features.season_average import (
    PRESEASON_FEATURES,
    _nflverse_season_snap_usage,
    build_season_average_data,
    preseason_roster_snapshot,
    team_season_volume,
    team_transition_rows,
)


def test_nflverse_snap_usage_bridges_player_ids_and_builds_qb_simplex():
    snaps = pd.DataFrame(
        {
            "season": [2024, 2024, 2024],
            "week": [1, 1, 1],
            "game_type": ["REG", "REG", "REG"],
            "team": ["BLT", "BLT", "BLT"],
            "player": ["QB One", "QB Two", "Runner"],
            "pfr_player_id": ["One00", "Two00", "Run00"],
            "position": ["QB", "QB", "RB"],
            "offense_snaps": [48, 12, 30],
            "offense_pct": [0.80, 0.20, 0.50],
        }
    )
    players = pd.DataFrame(
        {
            "pfr_id": ["One00", "Two00", "Run00"],
            "gsis_id": ["qb1", "qb2", "rb1"],
        }
    )

    usage = _nflverse_season_snap_usage(snaps, players).set_index("player_key")

    assert usage["team"].eq("BAL").all()
    assert np.isclose(usage.loc["qb1", "snap_share"], 0.80)
    assert np.isclose(usage.loc[["qb1", "qb2"], "qb_snap_share"].sum(), 1.0)
    assert usage.loc["rb1", "qb_snap_share"] == 0.0


def test_week_one_roster_snapshot_is_point_in_time_and_depth_aware():
    rosters = pd.DataFrame(
        {
            "season": [2024] * 5,
            "team": ["ATL"] * 5,
            "week": [1, 1, 1, 1, 2],
            "game_type": ["REG"] * 5,
            "gsis_id": ["qb1", "qb2", "fb1", "cut", "late"],
            "full_name": ["Starter", "Backup", "Full Back", "Cut QB", "Late Add"],
            "position": ["QB", "QB", "FB", "QB", "WR"],
            "status": ["ACT", "ACT", "RES", "CUT", "ACT"],
            "years_exp": [10, 1, 3, 2, 1],
            "birth_date": ["1990-01-01"] * 5,
        }
    )
    depth = pd.DataFrame(
        {
            "season": [2024, 2024, 2024],
            "club_code": ["ATL"] * 3,
            "week": [1, 1, 1],
            "game_type": ["REG"] * 3,
            "formation": ["Offense"] * 3,
            "gsis_id": ["qb1", "qb2", "fb1"],
            "full_name": ["Starter", "Backup", "Full Back"],
            "position": ["QB", "QB", "FB"],
            "depth_team": [1, 2, 1],
        }
    )

    snapshot = preseason_roster_snapshot(rosters, depth)

    assert set(snapshot["player_key"]) == {"qb1", "qb2", "fb1"}
    assert set(snapshot["position"]) == {"QB", "RB"}
    assert snapshot.set_index("player_key").loc["qb1", "qb_listed_starter"] == 1
    assert snapshot.set_index("player_key").loc["qb2", "qb_depth_rank"] == 2
    assert snapshot.set_index("player_key").loc["qb1", "observed_roster_games"] == 1
    assert snapshot["roster_snapshot_source"].eq("nflverse_week1").all()


def test_team_transitions_use_prior_season_rates_and_normalize_franchise_codes():
    rows = []
    for season, team, pass_att, rush_att, targets in (
        (2019, "ARI", 30, 20, 28),
        (2020, "CRD", 36, 24, 32),
    ):
        for week in (1, 2):
            rows.extend(
                [
                    _row(season, week, team, "QB", pass_att=pass_att),
                    _row(season, week, team, "RB", rush_att=rush_att),
                    _row(season, week, team, "WR", targets=targets),
                    _row(season, week, team, "DEF", targets=99),
                ]
            )
    volume = team_season_volume(pd.DataFrame(rows))
    transitions = team_transition_rows(volume)

    assert len(transitions) == 1
    row = transitions.iloc[0]
    assert row["team"] == "ARI"
    assert row["season"] == 2020
    assert np.isclose(row["prior_plays_per_game"], 50.0)
    assert np.isclose(row["plays_per_game"], 60.0)
    # DEF was excluded from both the position universe and target support.
    assert np.isclose(row["targets_per_game"], 32.0)


def test_real_preseason_rows_are_four_position_and_prior_only():
    data = build_season_average_data([2018, 2019, 2020], source="legacy")

    assert set(data.player_rows["position"]) == {"QB", "RB", "WR", "TE"}
    assert data.team_rows.groupby("season").size().eq(32).all()
    assert set(PRESEASON_FEATURES) <= set(data.player_rows.columns)
    assert data.player_rows["observed_availability"].between(0, 1).all()
    # Every modeled row is a Y-1 -> Y transition; no current outcome appears
    # in the documented feature list.
    assert not {
        "pass_att",
        "targets",
        "rush_att",
        "pass_attempt_share",
        "target_share",
        "carry_share",
        "observed_roster_games",
        "offense_snaps",
        "qb_snap_share",
        "observed_qb_workload_share",
    }.intersection(
        PRESEASON_FEATURES
    )


def _row(
    season,
    week,
    team,
    position,
    *,
    pass_att=0,
    pass_sacks=0,
    rush_att=0,
    targets=0,
):
    return {
        "player_id": f"{team}-{position}",
        "player_name": f"{team}-{position}",
        "position": position,
        "team": team,
        "season": season,
        "week": week,
        "pass_att": pass_att,
        "pass_sacks": pass_sacks,
        "pass_sacks_available": 1,
        "rush_att": rush_att,
        "targets": targets,
    }
