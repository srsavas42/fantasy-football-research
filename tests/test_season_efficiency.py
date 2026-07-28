"""Season efficiency aggregation, shrinkage, and temporal boundaries."""

import numpy as np
import pandas as pd

from ffmodel.data.schema import conform
from ffmodel.features.season_efficiency import (
    add_conditional_volume_efficiency_features,
    add_volume_efficiency_features,
    lagged_efficiency_rows,
    player_season_efficiency,
)


def test_efficiency_uses_ratio_of_totals_and_opportunity_weighted_pooling():
    weeks = pd.DataFrame(
        [
            _week("a", "Alpha", 1, targets=2, receptions=2, rec_yds=40, rec_epa=1.0),
            _week("a", "Alpha", 2, targets=8, receptions=3, rec_yds=60, rec_epa=-1.0),
            _week("b", "Beta", 1, targets=90, receptions=60, rec_yds=720, rec_epa=18.0),
        ]
    )

    out = player_season_efficiency(conform(weeks)).set_index("player_key")

    assert np.isclose(out.loc["a", "rec_yards_per_target"], 10.0)
    # Position pool is (100 + 720) / (10 + 90) = 8.2, with a 40-target prior.
    assert np.isclose(
        out.loc["a", "shrunk_rec_yards_per_target"],
        (100 + 40 * 8.2) / 50,
    )
    assert np.isclose(out.loc["a", "rec_epa_per_target"], 0.0)
    assert out.loc["a", "advanced_efficiency_available"] == 1


def test_optional_efficiency_is_missing_not_zero_and_lags_one_season():
    raw = pd.DataFrame([_week("a", "Alpha", 1, targets=10, receptions=5, rec_yds=70)])
    conformed = conform(raw)
    assert conformed["rec_epa"].isna().all()

    efficiency = player_season_efficiency(conformed)
    assert efficiency["rec_epa_per_target"].isna().all()
    prior = lagged_efficiency_rows(efficiency).iloc[0]
    assert prior["season"] == 2025
    assert np.isclose(
        prior["prior_rec_yards_per_target"],
        efficiency.iloc[0]["shrunk_rec_yards_per_target"],
    )
    assert "rec_yards_per_target" not in prior.index


def test_volume_efficiency_features_encode_reliability_and_relative_quality():
    rows = pd.DataFrame(
        {
            "season": [2024, 2024],
            "team": ["BUF", "BUF"],
            "position": ["WR", "WR"],
            "prior_pass_att": [0, 0],
            "prior_targets": [10, 100],
            "prior_rush_att": [0, 0],
            "prior_rec_yards_per_target": [6.0, 9.0],
            "prior_rec_epa_per_target": [-0.1, 0.3],
            "prior_rec_first_down_rate": [0.2, 0.5],
        }
    )
    out = add_volume_efficiency_features(rows)

    assert np.isclose(out.loc[0, "prior_rec_efficiency_reliability"], 0.2)
    assert np.isclose(out.loc[1, "prior_rec_efficiency_reliability"], 100 / 140)
    assert out.loc[1, "prior_rec_quality_rank"] > out.loc[0, "prior_rec_quality_rank"]
    assert (
        out.loc[1, "prior_rec_team_quality_signal"]
        > out.loc[0, "prior_rec_team_quality_signal"]
    )


def test_conditional_efficiency_features_use_preseason_room_and_continuity():
    rows = pd.DataFrame(
        {
            "season": [2025, 2025, 2025],
            "team": ["BUF", "BUF", "BUF"],
            "position": ["RB", "RB", "WR"],
            "team_change": [0, 1, 0],
            "cold_start": [0, 0, 1],
            "prior_carry_role": [0.60, 0.30, 0.10],
            "draft_carry_prior": [0.0, 0.0, 0.0],
            "prior_target_role": [0.15, 0.10, 0.50],
            "draft_target_prior": [0.0, 0.0, 0.0],
            "prior_pass_role": [0.0, 0.0, 0.0],
            "draft_pass_prior": [0.0, 0.0, 0.0],
            "prior_rush_epa_per_carry": [0.20, -0.10, 0.0],
            "prior_rec_epa_per_target": [0.10, 0.0, 0.20],
            "prior_pass_td_rate": [np.nan, np.nan, np.nan],
            "prior_pass_quality_signal": [np.nan, np.nan, np.nan],
            "prior_rec_quality_signal": [0.10, -0.05, 0.20],
        }
    )

    out = add_conditional_volume_efficiency_features(rows)

    # The RB room's leader owns two thirds of its prior role, so competition
    # is the remaining one third for both players in that room.
    assert np.isclose(out.loc[0, "prior_carry_room_competition"], 1 / 3)
    assert np.isclose(out.loc[1, "prior_carry_role_uncertainty"], 2 / 3)
    assert out.loc[0, "prior_role_continuity"] == 1
    assert out.loc[1, "prior_role_continuity"] == 0
    assert out.loc[2, "prior_role_continuity"] == 0
    assert np.isclose(
        out.loc[0, "prior_rush_epa_per_carry_centered_x_room"],
        0.15 / 3,
    )
    assert out.loc[1, "prior_rush_epa_per_carry_centered_x_returning"] == 0


def _week(
    player_id,
    player_name,
    week,
    *,
    targets=0,
    receptions=0,
    rec_yds=0,
    rec_epa=np.nan,
):
    return {
        "player_id": player_id,
        "player_name": player_name,
        "position": "WR",
        "team": "BUF",
        "season": 2024,
        "week": week,
        "source": "test",
        "targets": targets,
        "receptions": receptions,
        "rec_yds": rec_yds,
        "rec_epa": rec_epa,
    }
