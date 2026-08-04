"""Season-average model invariants without sampler-heavy fitting."""

import numpy as np
import pandas as pd
import pytest

az = pytest.importorskip("arviz")

from ffmodel.evaluation.season_average import (
    RidgeRosterBaseline,
    persistence_shares,
)
from ffmodel.features.season_average import SeasonAverageData
from ffmodel.models.season_availability import (
    AvailabilityPrediction,
    QBWorkloadShareModel,
    SeasonAvailabilityModel,
)
from ffmodel.models.season_regime import SeasonRegimeModel
from ffmodel.models.season_regime_coupling import SeasonRegimeRoleCoupling
from ffmodel.models.volume_season_average import (
    RosterSharePrediction,
    SeasonAverageVolumePipeline,
    SeasonRosterShareModel,
    TeamSeasonAverageModel,
    _allocate_season_counts,
)


def test_team_posterior_produces_coherent_per_game_counts():
    rows = _team_rows()
    model = TeamSeasonAverageModel()
    model._design(rows, fit=True)
    model.models_sacks = True
    draws = 12
    model.idata = az.from_dict(
        posterior={
            "play_intercept": np.full((1, draws), np.log(62.0)),
            "play_persistence": np.full((1, draws), 0.75),
            "play_era": np.zeros((1, draws)),
            "play_team": np.zeros((1, draws, len(model.teams))),
            "play_alpha_pg": np.full((1, draws), 20.0),
            "play_transition_sd": np.full((1, draws), 0.08),
            "pass_intercept": np.full((1, draws), 0.30),
            "pass_persistence": np.full((1, draws), 0.75),
            "pass_era": np.zeros((1, draws)),
            "pass_team": np.zeros((1, draws, len(model.teams))),
            "pass_transition_sd": np.full((1, draws), 0.10),
            "sack_intercept": np.full((1, draws), -2.67),
            "sack_persistence": np.full((1, draws), 0.50),
            "sack_era": np.zeros((1, draws)),
            "sack_team": np.zeros((1, draws, len(model.teams))),
            "target_intercept": np.full((1, draws), 2.5),
            "target_persistence": np.full((1, draws), 0.65),
            "target_era": np.zeros((1, draws)),
            "target_team": np.zeros((1, draws, len(model.teams))),
            "target_transition_sd": np.full((1, draws), 0.10),
        }
    )
    prediction = model.predict_average_samples(rows, seed=4)

    assert prediction["plays_per_game"].shape == (2, draws)
    assert (prediction["targets_per_game"] <= prediction["pass_attempts_per_game"]).all()
    assert (prediction["pass_attempts_per_game"] <= prediction["plays_per_game"]).all()
    assert np.array_equal(
        prediction["opportunity_plays"],
        prediction["pass_attempts"] + prediction["rush_attempts"],
    )
    assert np.array_equal(
        prediction["plays"],
        prediction["pass_attempts"]
        + prediction["rush_attempts"]
        + prediction["sacks"],
    )
    assert np.array_equal(
        prediction["pass_attempts"],
        prediction["targets"] + prediction["no_target_attempts"],
    )


@pytest.mark.parametrize("stream", ["pass", "target", "carry"])
def test_roster_share_draws_sum_to_one(stream):
    rows = _player_rows()
    model = SeasonRosterShareModel(stream)
    model._design(rows, fit=True, use_observed_availability=True)
    draws = 10
    model.idata = az.from_dict(
        posterior={
            "beta": np.zeros((1, draws, len(model.feature_names))),
            "player_effect": np.zeros((1, draws, len(model.players))),
            "role_innovation_sd": np.full((1, draws), 0.30),
            "allocation_concentration": np.full((1, draws), 100.0),
        }
    )
    prediction = model.predict_share_samples(rows, seed=7)

    for _, group in prediction.rows.groupby(["season", "team"]):
        assert np.allclose(prediction.shares[group.index].sum(axis=0), 1.0)
    if stream == "target":
        assert np.all(prediction.shares[prediction.rows["position"].eq("QB")] == 0.0)
    assert set(prediction.rows["position"]) == {"QB", "RB", "WR", "TE"}


def test_point_baselines_are_roster_coherent():
    rows = _player_rows()
    persistence = persistence_shares(rows, "target")
    ridge = RidgeRosterBaseline("target", alpha=1.0).fit(rows)
    predicted = ridge.predict_shares(rows)

    for _, group in rows.groupby(["season", "team"]):
        idx = group.index.to_numpy(dtype=int)
        assert np.isclose(persistence[idx].sum(), 1.0)
        assert np.isclose(predicted[idx].sum(), 1.0)


def test_volume_efficiency_features_are_stream_specific_and_acceptance_gated():
    rows = _player_rows().assign(
        prior_pass_yards_per_attempt=7.1,
        prior_pass_epa_per_attempt=0.08,
        prior_pass_completion_rate=0.64,
        prior_pass_td_rate=0.045,
        prior_pass_quality_signal=0.15,
        prior_rec_yards_per_target=7.8,
        prior_rec_epa_per_target=0.12,
        prior_rec_air_yards_per_target=8.5,
        prior_rush_yards_per_carry=4.3,
        prior_rush_epa_per_carry=0.02,
        prior_rush_first_down_rate=0.22,
    )

    passer = RidgeRosterBaseline("pass").fit(rows)
    accepted_target = RidgeRosterBaseline("target").fit(rows)
    challenger_target = RidgeRosterBaseline(
        "target", include_experimental_efficiency=True
    ).fit(rows)
    carrier = RidgeRosterBaseline("carry").fit(rows)
    production_carry = SeasonRosterShareModel("carry")
    production_carry._design(rows, fit=True, use_observed_availability=True)
    production_target = SeasonRosterShareModel("target")
    production_target._design(rows, fit=True, use_observed_availability=True)

    assert "prior_pass_quality_signal" in passer.feature_names
    assert "prior_pass_td_rate" in passer.feature_names
    assert "prior_rush_epa_per_carry" in carrier.feature_names
    assert "prior_rush_epa_per_carry" not in production_carry.feature_names
    assert "prior_rec_yards_per_target" not in production_target.feature_names
    assert "prior_rec_yards_per_target" not in accepted_target.feature_names
    assert "prior_rec_yards_per_target" in challenger_target.feature_names


def test_integer_season_allocation_conserves_team_totals():
    rows = pd.DataFrame(
        {
            "season": [2024, 2024],
            "team": ["A", "A"],
            "_group_idx": [0, 0],
        }
    )
    prediction = RosterSharePrediction(
        rows=rows,
        group_keys=pd.DataFrame({"season": [2024], "team": ["A"]}),
        shares=np.array([[0.999, 0.999], [0.001, 0.001]]),
    )
    totals = np.array([[100, 120]])

    counts = _allocate_season_counts(prediction, totals, seed=2)

    assert np.array_equal(counts.sum(axis=0), totals[0])
    assert np.issubdtype(counts.dtype, np.integer)


def test_pipeline_cross_stream_counts_conserve_every_team_total():
    team_rows = pd.DataFrame({"season": [2024], "team": ["A"]})
    player_rows = _player_rows().query("team == 'A'").reset_index(drop=True)
    team_prediction = {
        "rows": team_rows,
        "games": np.array([17]),
        "pass_attempts": np.array([[600, 610, 620]]),
        "targets": np.array([[540, 550, 560]]),
        "rush_attempts": np.array([[430, 420, 410]]),
    }

    class FakeTeamModel:
        def predict_average_samples(self, rows, *, games=None, seed=0):
            return team_prediction

    class FakeShareModel:
        def __init__(self, shares):
            self.shares = np.asarray(shares, dtype=float)

        def predict_share_samples(self, rows, **kwargs):
            prepared = rows.copy().reset_index(drop=True)
            prepared["_group_idx"] = 0
            prepared["_projected_availability"] = 1.0
            shares = np.repeat(self.shares[:, None], 3, axis=1)
            return RosterSharePrediction(
                rows=prepared,
                group_keys=team_rows,
                shares=shares,
            )

    class FakeAvailabilityModel:
        def predict_samples(self, rows, *, seed=0):
            prepared = rows.copy().sort_values(
                ["season", "team", "player_key"]
            ).reset_index(drop=True)
            availability = np.full((len(prepared), 3), 0.9)
            return AvailabilityPrediction(
                rows=prepared,
                probability=availability,
                games_active=np.full((len(prepared), 3), 15),
                availability=availability,
            )

    pipeline = SeasonAverageVolumePipeline(
        team_model=FakeTeamModel(),
        workload_model=FakeShareModel([1.0, 0.0, 0.0, 0.0]),
        target_model=FakeShareModel([0.01, 0.20, 0.50, 0.29]),
        carry_model=FakeShareModel([0.12, 0.84, 0.03, 0.01]),
        availability_model=FakeAvailabilityModel(),
    )
    prediction = pipeline.predict_samples(
        SeasonAverageData(team_rows=team_rows, player_rows=player_rows), seed=11
    )

    assert np.array_equal(
        prediction.pass_attempts.sum(axis=0), team_prediction["pass_attempts"][0]
    )
    assert np.array_equal(
        prediction.targets.sum(axis=0), team_prediction["targets"][0]
    )
    assert np.array_equal(
        prediction.carries.sum(axis=0), team_prediction["rush_attempts"][0]
    )
    assert set(prediction.player_rows["position"]) == {"QB", "RB", "WR", "TE"}


def test_pipeline_save_load_preserves_prediction_metadata(tmp_path):
    team = TeamSeasonAverageModel()
    team._design(_team_rows(), fit=True)
    team.idata = az.from_dict(posterior={"x": np.zeros((1, 2))})
    target = SeasonRosterShareModel("target")
    target._design(_player_rows(), fit=True, use_observed_availability=True)
    target.idata = az.from_dict(posterior={"x": np.zeros((1, 2))})
    carry = SeasonRosterShareModel("carry", cold_role_innovation=True)
    carry._design(_player_rows(), fit=True, use_observed_availability=True)
    carry.cold_role_multiplier = 1.73
    carry.idata = az.from_dict(posterior={"x": np.zeros((1, 2))})
    availability = SeasonAvailabilityModel(
        extra_features=("prior_availability_3yr",),
        position_specific_concentration=True,
    )
    availability._matrix(availability._prepare(_player_rows()), fit=True)
    availability.idata = az.from_dict(posterior={"x": np.zeros((1, 2))})
    workload = QBWorkloadShareModel()
    workload._design(_player_rows(), fit=True)
    workload.idata = az.from_dict(posterior={"x": np.zeros((1, 2))})
    pipeline = SeasonAverageVolumePipeline(
        team_model=team,
        target_model=target,
        carry_model=carry,
        availability_model=availability,
        workload_model=workload,
        role_regime_coupling=True,
    )
    pipeline.regime_model = SeasonRegimeModel(steps=100).fit(_player_rows())
    pipeline.regime_coupler = SeasonRegimeRoleCoupling().fit(
        _player_rows(), thresholds=pipeline.regime_model.thresholds
    )

    restored = SeasonAverageVolumePipeline.load(pipeline.save(tmp_path / "average"))

    assert restored.team_model.teams == team.teams
    assert restored.availability_model.positions == availability.positions
    assert restored.availability_model.extra_features == availability.extra_features
    assert restored.availability_model.position_specific_concentration is True
    assert np.allclose(
        restored.availability_model.feature_projection,
        availability.feature_projection,
    )
    assert restored.workload_model.feature_names == workload.feature_names
    assert np.isclose(
        restored.workload_model.role_innovation_scale,
        workload.role_innovation_scale,
    )
    assert restored.target_model.feature_names == target.feature_names
    assert restored.carry_model.cold_role_prior == carry.cold_role_prior
    assert restored.target_model.availability_prior == target.availability_prior
    assert np.isclose(restored.target_model.per_snap_weight, 0.75)
    # Against the fitted model rather than a literal. What matters here is that
    # the cap survives the round trip; pinning the promoted value in a
    # serialization test makes it fail whenever that value is revalidated, which
    # is a false alarm about the wrong thing.
    assert np.isclose(restored.target_model.innovation_cap, target.innovation_cap)
    assert np.isclose(restored.carry_model.innovation_cap, carry.innovation_cap)
    # The cold-role widening is a fitted quantity, not a setting: a served
    # artifact that restored the flag but not the multiplier would silently
    # revert to one scale for rookies and starters alike.
    assert restored.carry_model.cold_role_innovation == carry.cold_role_innovation
    assert np.isclose(
        restored.carry_model.cold_role_multiplier, carry.cold_role_multiplier
    )
    assert np.isclose(
        restored.target_model.cold_role_multiplier_cap,
        target.cold_role_multiplier_cap,
    )
    assert restored.role_regime_coupling is True
    assert restored.regime_model is not None
    assert restored.regime_coupler is not None
    assert np.allclose(
        restored.regime_model.predict_proba(_player_rows()),
        pipeline.regime_model.predict_proba(_player_rows()),
    )


def _team_rows():
    return pd.DataFrame(
        {
            "season": [2023, 2023],
            "team": ["A", "B"],
            "games": [17, 17],
            "opportunity_plays": [1050, 1000],
            "plays": [1090, 1035],
            "pass_attempts": [620, 560],
            "sacks": [40, 35],
            "sacks_observed": [True, True],
            "targets": [570, 510],
            "valid_target_pass_attempts": [620, 560],
            "valid_targets": [570, 510],
            "prior_opportunity_plays_per_game": [62.0, 60.0],
            "prior_pass_rate": [0.59, 0.56],
            "prior_sack_rate": [0.061, 0.059],
            "prior_target_rate": [0.91, 0.90],
        }
    )


def _player_rows():
    rows = []
    for team in ("A", "B"):
        for i, (position, targets, carries, target_role, carry_role) in enumerate(
            (
                ("QB", 0, 40, np.nan, 0.08),
                ("RB", 70, 260, 0.12, 0.55),
                ("WR", 150, 3, 0.28, 0.004),
                ("TE", 90, 0, 0.16, np.nan),
            )
        ):
            rows.append(
                {
                    "season": 2023,
                    "team": team,
                    "player_key": f"{team}-{position}",
                    "player_name": f"{team}-{position}",
                    "position": position,
                    "pass_att": 580 if position == "QB" else int(position == "WR"),
                    "targets": targets,
                    "rush_att": carries,
                    "prior_pass_role": 0.92 if position == "QB" else 0.0005,
                    "prior_qb_snap_share": 1.0 if position == "QB" else 0.0,
                    "prior_target_role": target_role,
                    "prior_carry_role": carry_role,
                    "draft_pass_prior": 0.0,
                    "draft_target_prior": 0.0,
                    "draft_carry_prior": 0.0,
                    "prior_availability": 0.9,
                    "prior_snap_share": 0.5,
                    "age": 26 + i,
                    "experience": 3,
                    "team_change": 0,
                    "cold_start": int(np.isnan(target_role)),
                    "observed_availability": 0.9,
                    "team_games": 17,
                    "games": 15,
                    "primary_qb": int(position == "QB"),
                    "offense_snaps": 1000 if position == "QB" else 500,
                    "snap_counts_observed": 1,
                }
            )
    return pd.DataFrame(rows)
