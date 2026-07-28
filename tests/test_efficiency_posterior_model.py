"""Posterior efficiency prediction and persistence invariants."""

import numpy as np
import pandas as pd
import pytest

az = pytest.importorskip("arviz")

from ffmodel.models.efficiency_season_average import (
    EFFICIENCY_MODEL_BY_TARGET,
    ExposureWeightedEfficiencyModel,
    PosteriorSeasonEfficiencyModel,
    SeasonAveragePosteriorEfficiencyPipeline,
)


def test_beta_binomial_efficiency_draws_respect_position_and_bounds():
    rows = _rows()
    model = PosteriorSeasonEfficiencyModel(
        EFFICIENCY_MODEL_BY_TARGET["pass_completion_rate"],
        mean_mode="posterior",
    )
    eligible = model._eligible(rows)
    matrix = model._matrix(eligible, fit=True)
    draws = 12
    model.idata = az.from_dict(
        posterior={
            "intercept": np.full((1, draws), np.log(0.65 / 0.35)),
            "prior_persistence": np.ones((1, draws)),
            "beta": np.zeros((1, draws, matrix.shape[1])),
            "concentration": np.full((1, draws), 150.0),
        }
    )

    prediction = model.predict_samples(rows, draws=8, seed=3)

    assert prediction.rate.shape == (len(rows), 8)
    assert np.isfinite(prediction.rate[0]).all()
    assert ((prediction.rate[0] > 0) & (prediction.rate[0] < 1)).all()
    assert np.isnan(prediction.rate[1:]).all()
    observed = model.predict_observed_samples(rows, draws=8, seed=4)
    assert np.isfinite(observed[0]).all()
    assert np.allclose(observed[0] * rows.loc[0, "pass_att"], np.round(observed[0] * rows.loc[0, "pass_att"]))


def test_continuous_efficiency_uses_future_exposure_and_stays_bounded():
    rows = _rows()
    model = PosteriorSeasonEfficiencyModel(
        EFFICIENCY_MODEL_BY_TARGET["rush_yards_per_carry"],
        mean_mode="posterior",
    )
    eligible = model._eligible(rows)
    matrix = model._matrix(eligible, fit=True)
    draws = 10
    model.idata = az.from_dict(
        posterior={
            "intercept": np.full((1, draws), 0.0),
            "prior_persistence": np.ones((1, draws)),
            "position_effect": np.zeros((1, draws, 4)),
            "beta": np.zeros((1, draws, matrix.shape[1])),
            "season_sigma": np.full((1, draws), 0.3),
            "opportunity_sigma": np.full((1, draws), 6.0),
        }
    )
    exposure = np.repeat(
        rows["rush_att"].clip(lower=1).to_numpy()[:, None], 7, axis=1
    )

    prediction = model.predict_samples(
        rows, draws=7, exposure_samples=exposure, seed=8
    )

    assert prediction.rate.shape == (len(rows), 7)
    assert np.isfinite(prediction.rate).all()
    assert (prediction.rate >= model.spec.lower).all()
    assert (prediction.rate <= model.spec.upper).all()


def test_posterior_efficiency_mean_can_follow_aligned_volume_draws():
    rows = _rows()
    model = PosteriorSeasonEfficiencyModel(
        EFFICIENCY_MODEL_BY_TARGET["rush_yards_per_carry"],
        mean_mode="posterior",
    )
    eligible = model._eligible(rows)
    raw = model._raw_matrix(eligible, fit=True)
    model.feature_projection = np.eye(raw.shape[1])
    volume_column = model._volume_feature_column()
    assert volume_column is not None
    draws = 5
    beta = np.zeros((1, draws, raw.shape[1]))
    beta[:, :, volume_column] = 1.0
    model.idata = az.from_dict(
        posterior={
            "intercept": np.zeros((1, draws)),
            "prior_persistence": np.zeros((1, draws)),
            "position_effect": np.zeros((1, draws, 4)),
            "beta": beta,
            "season_sigma": np.zeros((1, draws)),
            "opportunity_sigma": np.zeros((1, draws)),
        }
    )
    volume = np.column_stack(
        [
            np.full(len(rows), 0.25),
            np.full(len(rows), 20.0),
            np.full(len(rows), 0.25),
            np.full(len(rows), 20.0),
            np.full(len(rows), 0.25),
        ]
    )

    prediction = model.predict_samples(
        rows,
        draws=draws,
        volume_feature_samples=volume,
        seed=9,
    )

    assert np.all(prediction.mean[:, 1] > prediction.mean[:, 0])
    assert np.all(prediction.mean[:, 3] > prediction.mean[:, 2])


def test_prior_only_rate_preserves_the_accepted_pooled_mean():
    rows = _rows().assign(
        rush_td_rate=[0.04, 0.05, 0.02, 0.01],
        prior_rush_td_rate=[0.035, 0.045, 0.018, 0.009],
        eff_rush_td=[2, 10, 0, 0],
    )
    model = PosteriorSeasonEfficiencyModel(
        EFFICIENCY_MODEL_BY_TARGET["rush_td_rate"]
    )
    eligible = model._eligible(rows)
    model._matrix(eligible, fit=True)
    model._prior_signal(eligible, fit=True)
    draws = 6
    model.idata = az.from_dict(
        posterior={"concentration": np.full((1, draws), 120.0)}
    )

    prediction = model.predict_samples(rows, draws=draws, seed=5)

    expected = rows["prior_rush_td_rate"].to_numpy(dtype=float)
    assert np.allclose(prediction.mean, expected[:, None])


def test_posterior_efficiency_pipeline_round_trips_metadata(tmp_path):
    rows = _rows()
    target = "pass_completion_rate"
    model = PosteriorSeasonEfficiencyModel(
        EFFICIENCY_MODEL_BY_TARGET[target], mean_mode="posterior"
    )
    matrix = model._matrix(model._eligible(rows), fit=True)
    model.training_rows = 1
    model.idata = az.from_dict(
        posterior={
            "intercept": np.zeros((1, 3)),
            "prior_persistence": np.ones((1, 3)),
            "beta": np.zeros((1, 3, matrix.shape[1])),
            "concentration": np.full((1, 3), 100.0),
        }
    )
    pipeline = SeasonAveragePosteriorEfficiencyPipeline(models={target: model})

    restored = SeasonAveragePosteriorEfficiencyPipeline.load(
        pipeline.save(tmp_path / "efficiency")
    )

    assert set(restored.models) == {target}
    assert restored.models[target].feature_names == model.feature_names
    assert restored.models[target].training_rows == 1
    assert np.allclose(
        restored.models[target].feature_projection, model.feature_projection
    )


def test_fixed_ridge_mean_round_trips_with_dispersion_posterior(tmp_path):
    rows = _rows().assign(
        pass_yards_per_attempt=[7.4, np.nan, np.nan, np.nan],
        prior_pass_yards_per_attempt=[7.2, np.nan, np.nan, np.nan],
    )
    target = "pass_yards_per_attempt"
    spec = EFFICIENCY_MODEL_BY_TARGET[target]
    model = PosteriorSeasonEfficiencyModel(spec, mean_mode="ridge")
    eligible = model._eligible(rows)
    model._matrix(eligible, fit=True)
    model._prior_signal(eligible, fit=True)
    model.ridge_model = ExposureWeightedEfficiencyModel(
        spec, alpha=500.0
    ).fit(rows)
    model.idata = az.from_dict(
        posterior={
            "season_sigma": np.full((1, 4), 0.4),
            "opportunity_sigma": np.full((1, 4), 7.0),
        }
    )
    expected = model.predict_samples(rows, draws=4, seed=2).mean
    pipeline = SeasonAveragePosteriorEfficiencyPipeline(models={target: model})

    restored = SeasonAveragePosteriorEfficiencyPipeline.load(
        pipeline.save(tmp_path / "ridge-efficiency")
    )
    actual = restored.models[target].predict_samples(rows, draws=4, seed=2).mean

    assert np.allclose(actual, expected, equal_nan=True)
    assert restored.models[target].ridge_model.coefficients is not None


def _rows():
    rows = []
    for index, position in enumerate(("QB", "RB", "WR", "TE")):
        pass_att = 500 if position == "QB" else 0
        rush_att = (50, 220, 8, 2)[index]
        rows.append(
            {
                "season": 2024,
                "team": "A",
                "player_key": position.lower(),
                "position": position,
                "pass_att": pass_att,
                "pass_completion_rate": 0.65 if position == "QB" else np.nan,
                "eff_pass_cmp": 325 if position == "QB" else 0,
                "rush_att": rush_att,
                "rush_yards_per_carry": 4.2 + index * 0.1,
                "prior_pass_completion_rate": 0.64 if position == "QB" else np.nan,
                "prior_pass_att": 480 if position == "QB" else 0,
                "prior_rush_yards_per_carry": 4.0 + index * 0.1,
                "prior_rush_att": max(rush_att - 10, 0),
                "prior_availability": 0.9 - index * 0.05,
                "prior_snap_share": 0.8 - index * 0.1,
                "age": 25 + index,
                "experience": 3 + index,
                "team_change": index % 2,
                "cold_start": 0,
                "oof_pass_attempts_per_team_game": pass_att / 17,
                "oof_carries_per_team_game": rush_att / 17,
            }
        )
    return pd.DataFrame(rows)
