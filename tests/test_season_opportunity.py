"""Unit invariants for snap, propensity, and opportunity hurdles."""

import numpy as np
import pandas as pd
import pytest

az = pytest.importorskip("arviz")

from ffmodel.models.season_opportunity import (
    CARRY_ELIGIBILITY_EFFICIENCY_FEATURES,
    CARRY_ELIGIBILITY_FEATURES,
    QB_PROPENSITY_FEATURES,
    SNAP_FEATURES,
    SNAP_HISTORY_FEATURES,
    TARGET_ROLE_FEATURES,
    QBPassPropensityModel,
    SeasonCarryEligibilityModel,
    SeasonSnapShareModel,
    SeasonTargetRoleModel,
)


def _rows():
    return pd.DataFrame(
        [
            {
                "season": 2024,
                "team": "A",
                "player_key": "qb",
                "position": "QB",
                "prior_snap_share": 0.95,
                "prior_availability": 0.9,
                "prior_qb_attempts_per_snap": 0.58,
                "prior_qb_snap_share": 0.95,
                "prior_carry_per_snap": 0.08,
                "prior_carry_role": 0.1,
                "age": 27,
                "experience": 5,
                "team_change": 0,
                "cold_start": 0,
                "roster_active": 1,
                "roster_reserve": 0,
                "depth_rank": 1,
                "qb_listed_starter": 1,
                "is_replacement_qb": 0,
                "offense_snaps": 900,
                "snap_share": 0.88,
                "snap_counts_observed": 1,
                "observed_availability": 0.94,
                "pass_att": 530,
                "rush_att": 55,
            },
            {
                "season": 2024,
                "team": "A",
                "player_key": "rb",
                "position": "RB",
                "prior_snap_share": 0.55,
                "prior_availability": 0.8,
                "prior_carry_per_snap": 0.42,
                "prior_carry_role": 0.55,
                "age": 24,
                "experience": 2,
                "team_change": 0,
                "cold_start": 0,
                "roster_active": 1,
                "roster_reserve": 0,
                "depth_rank": 1,
                "qb_listed_starter": 0,
                "is_replacement_qb": 0,
                "offense_snaps": 500,
                "snap_share": 0.49,
                "snap_counts_observed": 1,
                "observed_availability": 0.82,
                "pass_att": 0,
                "rush_att": 210,
            },
        ]
    )


def test_snap_share_is_gated_by_active_fraction():
    rows = _rows()
    model = SeasonSnapShareModel()
    prepared = model._prepare(rows)
    matrix = model._matrix(prepared, SNAP_FEATURES, fit=True)
    draws = 5
    model.idata = az.from_dict(
        posterior={
            "intercept": np.zeros((1, draws)),
            "position_effect": np.zeros((1, draws, len(model.positions))),
            "beta": np.zeros((1, draws, matrix.shape[1])),
            "concentration": np.full((1, draws), 50.0),
        }
    )
    active = np.vstack([np.ones(draws), np.zeros(draws)])

    prediction = model.predict_samples(rows, active_fraction_samples=active, seed=2)

    assert prediction.snap_share.shape == (2, draws)
    assert np.all(prediction.snap_share[1] == 0.0)
    assert np.all((prediction.snap_share[0] > 0) & (prediction.snap_share[0] < 1))


def test_opportunity_models_allow_gated_challenger_features():
    rows = _rows().assign(prior_rush_epa_per_carry=[0.1, -0.1])
    prepared = SeasonCarryEligibilityModel(
        extra_features=("prior_rush_epa_per_carry",)
    )._prepare(rows)
    model = SeasonCarryEligibilityModel(
        extra_features=("prior_rush_epa_per_carry",)
    )
    model._matrix(
        prepared,
        model._candidates(CARRY_ELIGIBILITY_FEATURES),
        fit=True,
    )

    assert "prior_rush_epa_per_carry" in model.feature_names


def test_promoted_opportunity_defaults_are_exactly_gated_features():
    assert SeasonSnapShareModel().extra_features == SNAP_HISTORY_FEATURES
    assert (
        SeasonCarryEligibilityModel().extra_features
        == CARRY_ELIGIBILITY_EFFICIENCY_FEATURES
    )


def test_feature_projection_removes_exact_collinearity():
    rows = _rows().assign(left_feature=[0.0, 1.0], right_feature=[1.0, 0.0])
    model = SeasonSnapShareModel(
        extra_features=("left_feature", "right_feature")
    )
    prepared = model._prepare(rows)

    matrix = model._matrix(
        prepared,
        model._candidates(SNAP_FEATURES),
        fit=True,
    )

    assert matrix.shape[1] < len(model.feature_names)
    assert model.feature_projection.shape == (
        len(model.feature_names),
        matrix.shape[1],
    )


def test_qb_propensity_is_zero_for_non_qbs():
    rows = _rows()
    model = QBPassPropensityModel()
    prepared = model._prepare(rows)
    matrix = model._matrix(prepared, QB_PROPENSITY_FEATURES, fit=True)
    draws = 4
    model.idata = az.from_dict(
        posterior={
            "intercept": np.zeros((1, draws)),
            "beta": np.zeros((1, draws, matrix.shape[1])),
            "concentration": np.full((1, draws), 40.0),
        }
    )

    prediction = model.predict_samples(rows, seed=3)

    assert np.all(prediction.propensity[1] == 0.0)
    assert np.all((prediction.propensity[0] > 0) & (prediction.propensity[0] < 1))


def test_carry_hurdle_returns_draw_level_eligibility():
    rows = _rows()
    model = SeasonCarryEligibilityModel()
    prepared = model._prepare(rows)
    matrix = model._matrix(prepared, CARRY_ELIGIBILITY_FEATURES, fit=True)
    draws = 6
    model.idata = az.from_dict(
        posterior={
            "intercept": np.zeros((1, draws)),
            "position_effect": np.zeros((1, draws, len(model.positions))),
            "beta": np.zeros((1, draws, matrix.shape[1])),
        }
    )

    prediction = model.predict_samples(rows, seed=4)

    assert prediction.probability.shape == (2, draws)
    assert set(np.unique(prediction.eligible)) <= {0.0, 1.0}


def test_target_role_hurdle_excludes_quarterbacks():
    rows = _rows().assign(
        prior_target_per_snap=[0.0, 0.12],
        prior_target_role=[0.0, 0.18],
        targets=[0, 65],
        team_games=[17, 17],
    )
    model = SeasonTargetRoleModel()
    prepared = model._prepare(rows)
    matrix = model._matrix(
        prepared,
        model._candidates(TARGET_ROLE_FEATURES),
        fit=True,
    )
    draws = 5
    model.idata = az.from_dict(
        posterior={
            "intercept": np.zeros((1, draws)),
            "position_effect": np.zeros((1, draws, 3)),
            "beta": np.zeros((1, draws, matrix.shape[1])),
        }
    )

    prediction = model.predict_samples(rows, seed=5)

    assert np.all(prediction.probability[0] == 0.0)
    assert set(np.unique(prediction.eligible[1])) <= {0.0, 1.0}
    SNAP_HISTORY_FEATURES,
