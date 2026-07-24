"""Sampler-free contracts for availability and QB workload layers."""

import numpy as np
import pandas as pd
import pytest

az = pytest.importorskip("arviz")

from ffmodel.features.season_injury import INJURY_AVAILABILITY_FEATURES
from ffmodel.models.season_availability import (
    QBWorkloadShareModel,
    SeasonAvailabilityModel,
)


def _rows():
    return pd.DataFrame(
        [
            {
                "season": 2024,
                "team": "A",
                "player_key": "qb1",
                "player_name": "Starter",
                "position": "QB",
                "team_games": 17,
                "games": 16,
                "prior_availability": 0.94,
                "prior_pass_role": 0.80,
                "prior_qb_snap_share": 0.90,
                "draft_pass_prior": 0.0,
                "offense_snaps": 900,
                "snap_counts_observed": 1,
                "age": 29,
                "experience": 6,
                "team_change": 0,
                "cold_start": 0,
                "roster_active": 1,
                "roster_reserve": 0,
                "depth_rank": 1,
                "qb_depth_rank": 1,
                "qb_listed_starter": 1,
                "primary_qb": 1,
            },
            {
                "season": 2024,
                "team": "A",
                "player_key": "qb2",
                "player_name": "Backup",
                "position": "QB",
                "team_games": 17,
                "games": 3,
                "prior_availability": 0.50,
                "prior_pass_role": 0.15,
                "prior_qb_snap_share": 0.10,
                "draft_pass_prior": 0.0,
                "offense_snaps": 100,
                "snap_counts_observed": 1,
                "age": 25,
                "experience": 2,
                "team_change": 1,
                "cold_start": 0,
                "roster_active": 1,
                "roster_reserve": 0,
                "depth_rank": 2,
                "qb_depth_rank": 2,
                "qb_listed_starter": 0,
                "primary_qb": 0,
            },
            {
                "season": 2024,
                "team": "A",
                "player_key": "rb1",
                "player_name": "Back",
                "position": "RB",
                "team_games": 17,
                "games": 14,
                "prior_availability": 0.80,
                "prior_pass_role": 0.0,
                "prior_qb_snap_share": 0.0,
                "draft_pass_prior": 0.0,
                "offense_snaps": 500,
                "snap_counts_observed": 1,
                "age": 24,
                "experience": 2,
                "team_change": 0,
                "cold_start": 0,
                "roster_active": 1,
                "roster_reserve": 0,
                "depth_rank": 1,
                "qb_depth_rank": np.nan,
                "qb_listed_starter": 0,
                "primary_qb": 0,
            },
        ]
    )


def test_availability_draws_are_bounded_integer_game_outcomes():
    rows = _rows()
    model = SeasonAvailabilityModel()
    prepared = model._prepare(rows)
    matrix = model._matrix(prepared, fit=True)
    draws = 8
    model.idata = az.from_dict(
        posterior={
            "any_intercept": np.full((1, draws), 1.5),
            "any_position_effect": np.zeros((1, draws, len(model.positions))),
            "any_beta": np.zeros((1, draws, matrix.shape[1])),
            "rate_intercept": np.full((1, draws), 2.0),
            "rate_position_effect": np.zeros((1, draws, len(model.positions))),
            "rate_beta": np.zeros((1, draws, matrix.shape[1])),
            "rate_concentration": np.full((1, draws), 30.0),
        }
    )

    prediction = model.predict_samples(rows, seed=3)

    assert prediction.availability.shape == (3, draws)
    assert ((prediction.games_active >= 0) & (prediction.games_active <= 17)).all()
    assert np.issubdtype(prediction.games_active.dtype, np.integer)
    assert ((prediction.availability >= 0) & (prediction.availability <= 1)).all()
    assert np.allclose(prediction.availability, prediction.games_active / 17)


def test_availability_supports_history_and_position_specific_dispersion():
    rows = _rows().assign(
        prior_availability_3yr=[0.90, 0.55, 0.82],
        prior_availability_trend=[0.02, -0.10, 0.01],
    )
    model = SeasonAvailabilityModel(
        extra_features=("prior_availability_3yr", "prior_availability_trend"),
        position_specific_concentration=True,
    )
    prepared = model._prepare(rows)
    model._matrix(prepared, fit=True)

    assert "prior_availability_3yr" in model.feature_names
    assert "prior_availability_trend" in model.feature_names


def test_availability_uses_injury_features_when_the_contract_supplies_them():
    rows = _rows().assign(
        prior_injury_report_weeks_3yr=[0.0, 3.0, 1.0],
        current_injury_expected_recovery_weeks=[0.0, 2.5, 0.0],
    )
    model = SeasonAvailabilityModel(extra_features=INJURY_AVAILABILITY_FEATURES)
    model._matrix(model._prepare(rows), fit=True)

    assert "prior_injury_report_weeks_3yr" in model.feature_names
    assert "current_injury_expected_recovery_weeks" in model.feature_names


def test_qb_workload_share_is_a_team_simplex_and_excludes_non_qbs():
    rows = _rows()
    model = QBWorkloadShareModel(role_innovation_scale=0.0)
    model._design(rows, fit=True)
    model.role_innovation_scale = 0.0
    draws = 6
    model.idata = az.from_dict(
        posterior={
            "beta": np.zeros((1, draws, len(model.feature_names))),
        }
    )

    availability = np.full((len(rows), draws), 0.9)
    prediction = model.predict_share_samples(
        rows, availability_samples=availability, seed=4
    )
    quarterback = prediction.rows["position"].eq("QB").to_numpy()

    assert np.allclose(prediction.shares[quarterback].sum(axis=0), 1.0)
    assert np.all(prediction.shares[~quarterback] == 0.0)
    assert (
        prediction.shares[prediction.rows["player_key"].eq("qb1")]
        > prediction.shares[prediction.rows["player_key"].eq("qb2")]
    ).all()
