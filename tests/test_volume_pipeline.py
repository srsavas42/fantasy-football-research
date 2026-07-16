"""End-to-end opportunity draw conservation and persistence."""

import numpy as np
import pandas as pd
import pytest

az = pytest.importorskip("arviz")

from ffmodel.models.volume_pipeline import VolumePipeline
from ffmodel.models.base import load_idata, save_idata
from ffmodel.models.volume_share import OpportunityShareModel
from ffmodel.models.volume_team import TeamVolumeModel, prepare_team_weeks


def _features():
    rows = []
    for player, position, targets, carries, target_share, carry_share in (
        ("quarterback", "QB", 0, 1, 0.0, 0.25),
        ("back", "RB", 2, 3, 0.25, 0.75),
        ("receiver", "WR", 6, 0, 0.75, 0.0),
    ):
        rows.append(
            {
                "player_id": player,
                "player_name": player,
                "position": position,
                "team": "A",
                "season": 2024,
                "week": 1,
                "pass_att": 10,
                "rush_att": carries,
                "targets": targets,
                "team_pass_att": 10,
                "team_rush_att": 4,
                "team_targets": 8,
                "team_plays": 14,
                "team_opportunity_valid": True,
                "ewma_target_share": target_share,
                "ewma_carry_share": carry_share,
                "ewma_opportunity_share": target_share + carry_share,
                "ewma_ypt": 8.0,
                "ewma_catch_rate": 0.65,
                "ewma_ypc": 4.0,
                "ewma_yds_per_touch": 5.0,
                "role_rank": 1,
                "is_active": 1,
            }
        )
    return pd.DataFrame(rows)


def _team_model(features, draws=12):
    model = TeamVolumeModel()
    model._design(prepare_team_weeks(features), fit=True)
    n_team = len(model.teams)
    model.idata = az.from_dict(
        posterior={
            "play_intercept": np.full((1, draws), np.log(14.0)),
            "play_team": np.zeros((1, draws, n_team)),
            "play_beta": np.zeros((1, draws, 3)),
            "pass_intercept": np.full((1, draws), 0.9),
            "pass_team": np.zeros((1, draws, n_team)),
            "pass_beta": np.zeros((1, draws, 3)),
            "target_intercept": np.full((1, draws), 2.0),
            "target_team": np.zeros((1, draws, n_team)),
            "target_beta": np.zeros((1, draws, 3)),
            "play_alpha": np.full((1, draws), 30.0),
        }
    )
    return model


def _share_model(features, stream, draws=12):
    model = OpportunityShareModel(stream)
    model._design(features, fit=True)
    model.idata = az.from_dict(
        posterior={
            "position_alpha": np.zeros((1, draws, len(model.positions))),
            "player_effect": np.zeros((1, draws, len(model.players))),
            "beta": np.zeros((1, draws, len(model.feature_names))),
            "allocation_concentration": np.full((1, draws), 10.0),
        }
    )
    return model


def test_pipeline_conserves_coherent_team_draws():
    features = _features()
    pipeline = VolumePipeline(
        team_model=_team_model(features),
        target_model=_share_model(features, "target"),
        carry_model=_share_model(features, "carry"),
    )
    prediction = pipeline.predict_samples(features, seed=9)
    assert (prediction.team["targets"] <= prediction.team["pass_attempts"]).all()
    assert np.array_equal(
        prediction.targets.counts.sum(axis=0), prediction.team["targets"][0]
    )
    expected_carries = prediction.team["plays"][0] - prediction.team["pass_attempts"][0]
    assert np.array_equal(prediction.carries.counts.sum(axis=0), expected_carries)


def test_pipeline_save_and_load_preserves_prediction_metadata(tmp_path):
    features = _features()
    pipeline = VolumePipeline(
        team_model=_team_model(features),
        target_model=_share_model(features, "target"),
        carry_model=_share_model(features, "carry"),
    )
    path = pipeline.save(tmp_path / "posterior")
    restored = VolumePipeline.load(path)
    assert restored.team_model.teams == pipeline.team_model.teams
    assert restored.target_model.feature_names == pipeline.target_model.feature_names
    assert restored.carry_model.players == pipeline.carry_model.players
    assert restored.target_model.position_log_prior == pipeline.target_model.position_log_prior


def test_idata_save_serializes_nested_sampler_metadata(tmp_path):
    idata = az.from_dict(posterior={"x": np.zeros((1, 2))})
    idata.attrs["sample_stats"] = {"inference_library": "nutpie", "settings": {"draws": 2}}
    path = save_idata(idata, tmp_path / "nested.nc")
    restored = load_idata(path)
    assert '"inference_library": "nutpie"' in restored.attrs["sample_stats"]
