"""Efficiency-v1 model and cross-fitting contracts."""

import numpy as np
import pandas as pd

from ffmodel.features.season_average import SeasonAverageData
from ffmodel.models.efficiency_season_average import (
    EFFICIENCY_MODEL_BY_TARGET,
    ExposureWeightedEfficiencyModel,
)
from ffmodel.evaluation.efficiency_season_average import (
    add_walk_forward_volume_features,
)
from ffmodel.evaluation.conditional_efficiency_volume import (
    conditional_volume_metrics,
)


def test_efficiency_model_uses_optional_volume_and_advanced_features():
    rows = _receiver_rows()
    spec = EFFICIENCY_MODEL_BY_TARGET["rec_yards_per_target"]
    full = ExposureWeightedEfficiencyModel(
        spec, alpha=1.0, use_volume=True, use_advanced=True
    ).fit(rows)
    history = ExposureWeightedEfficiencyModel(
        spec, alpha=1.0, use_volume=False, use_advanced=False
    ).fit(rows)

    assert spec.volume_feature in full.feature_names
    assert "prior_rec_epa_per_target" in full.feature_names
    assert spec.volume_feature not in history.feature_names
    assert "prior_rec_epa_per_target" not in history.feature_names
    prediction = full.predict(rows)
    assert np.isfinite(prediction).all()
    assert ((prediction >= spec.lower) & (prediction <= spec.upper)).all()


def test_ridge_efficiency_mean_can_follow_aligned_volume_draws():
    rows = _receiver_rows()
    spec = EFFICIENCY_MODEL_BY_TARGET["rec_yards_per_target"]
    model = ExposureWeightedEfficiencyModel(
        spec, alpha=1.0, use_volume=True, use_advanced=False
    ).fit(rows)
    volume_index = model.feature_names.index(spec.volume_feature)
    matrix_column = 1 + 2 * volume_index
    model.coefficients = np.zeros_like(model.coefficients)
    model.coefficients[matrix_column] = 1.0
    volume = np.column_stack(
        [
            np.full(len(rows), 1.0),
            np.full(len(rows), 20.0),
        ]
    )

    prediction = model.predict_volume_conditioned_samples(rows, volume)

    assert prediction.shape == volume.shape
    assert np.all(prediction[:, 1] > prediction[:, 0])


def test_volume_features_are_cross_fitted_by_season():
    players = []
    teams = []
    for season in (2021, 2022, 2023):
        teams.append(
            {
                "season": season,
                "team": "A",
                "prior_pass_attempts_per_game": 35.0,
                "prior_targets_per_game": 32.0,
                "prior_rush_attempts_per_game": 27.0,
            }
        )
        players.extend(_volume_roster(season))
    data = SeasonAverageData(pd.DataFrame(teams), pd.DataFrame(players))

    out = add_walk_forward_volume_features(data)

    assert out["oof_targets_per_team_game"].notna().all()
    assert out.loc[out["season"].eq(2021), "oof_volume_training_seasons"].eq(0).all()
    assert out.loc[out["season"].eq(2023), "oof_volume_training_seasons"].eq(2).all()
    assert np.isclose(
        out.loc[out["season"].eq(2023), "oof_targets_per_team_game"].sum(),
        32.0,
    )


def test_conditional_summary_compares_with_none_and_stream_reference():
    rows = []
    for model, errors in {
        "none": (10.0, 10.0),
        "unconditional": (9.0, 8.0),
        "room": (8.0, 9.0),
    }.items():
        for season, absolute_error in zip((2018, 2019), errors):
            rows.append(
                {
                    "stream": "carry",
                    "model": model,
                    "season": season,
                    "n": 10,
                    "absolute_error": absolute_error,
                    "squared_error": absolute_error**2,
                    "mae": absolute_error / 10,
                }
            )

    summary = conditional_volume_metrics(pd.DataFrame(rows)).set_index("model")

    assert summary.loc["room", "wins_vs_none"] == 2
    assert summary.loc["room", "wins_vs_reference"] == 1
    assert summary.loc["room", "recent_wins_vs_reference"] == 0


def _receiver_rows():
    rows = []
    for season in range(2017, 2025):
        for i, position in enumerate(("RB", "WR", "TE")):
            rows.append(
                {
                    "season": season,
                    "team": "A",
                    "player_key": f"{position}-{i}",
                    "position": position,
                    "targets": 40 + 10 * i,
                    "rec_yards_per_target": 6.0 + 0.2 * i + 0.05 * (season - 2017),
                    "prior_rec_yards_per_target": 6.0 + 0.2 * i,
                    "prior_targets": 35 + 10 * i,
                    "prior_availability": 0.9,
                    "prior_snap_share": 0.4 + 0.1 * i,
                    "age": 24 + i,
                    "experience": 2 + i,
                    "team_change": 0,
                    "cold_start": 0,
                    "oof_targets_per_team_game": 4.0 + i,
                    "prior_rec_epa_per_target": 0.1 + 0.02 * i,
                    "prior_rec_air_yards_per_target": 7.0 + i,
                    "prior_rec_yac_per_reception": 4.0 + 0.5 * i,
                    "prior_rec_first_down_rate": 0.3 + 0.02 * i,
                }
            )
    return pd.DataFrame(rows)


def _volume_roster(season):
    rows = []
    for position, targets, carries, passes in (
        ("QB", 0, 50, 570),
        ("RB", 80, 280, 0),
        ("WR", 280, 5, 0),
        ("TE", 160, 0, 0),
    ):
        rows.append(
            {
                "season": season,
                "team": "A",
                "player_key": f"A-{position}",
                "player_name": f"A-{position}",
                "position": position,
                "team_games": 17,
                "pass_att": passes,
                "targets": targets,
                "rush_att": carries,
                "prior_pass_role": 0.95 if position == "QB" else 0.0005,
                "prior_target_role": {"QB": 0.0001, "RB": 0.15, "WR": 0.50, "TE": 0.30}[position],
                "prior_carry_role": {"QB": 0.10, "RB": 0.80, "WR": 0.01, "TE": 0.001}[position],
                "draft_pass_prior": 0.0,
                "draft_target_prior": 0.0,
                "draft_carry_prior": 0.0,
                "prior_availability": 0.9,
                "prior_snap_share": 0.5,
                "age": 26,
                "experience": 3,
                "team_change": 0,
                "cold_start": 0,
                "observed_availability": 0.9,
            }
        )
    return rows
