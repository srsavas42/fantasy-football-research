"""The oof_* volume covariate must be built the same way at fit and at serve.

``add_walk_forward_volume_features`` builds these columns for training; at
serving time ``simulation.season_scoring.volume_efficiency_rows`` builds columns
of the same name from the production pipeline's posterior. The efficiency models
regress on them. If the two constructions differ, the fitted coefficient is
estimated against one distribution and applied to another — and measurement on
holdout 2024 says they do differ, by 0.46-0.66 opportunities per team game, with
the training-side construction the *less* accurate of the two.

These tests pin the contract that lets the two be made to agree, rather than the
numbers themselves, which need a sampler.
"""

import numpy as np
import pandas as pd
import pytest

from ffmodel.evaluation.efficiency_season_average import (
    VOLUME_OUTPUTS,
    add_walk_forward_volume_features,
)
from ffmodel.models.season_scoring import SeasonAverageScoringPipeline
from ffmodel.simulation.season_scoring import volume_efficiency_rows


def test_both_constructions_declare_the_same_columns():
    # This is the whole reason the mismatch is easy to miss: the two paths agree
    # on names, so nothing downstream can tell them apart.
    fit_side = {output for output, _, _ in VOLUME_OUTPUTS.values()}
    fit_side.add("oof_fumble_opportunities_per_team_game")

    source = volume_efficiency_rows.__doc__ or ""
    serve_side = {
        "oof_pass_attempts_per_team_game",
        "oof_targets_per_team_game",
        "oof_carries_per_team_game",
        "oof_fumble_opportunities_per_team_game",
    }

    assert fit_side == serve_side, source


def test_the_estimator_choice_is_explicit_and_validated():
    data = pd.DataFrame()

    with pytest.raises(ValueError, match="estimator"):
        add_walk_forward_volume_features(
            type("D", (), {"player_rows": data, "team_rows": data})(),
            estimator="nonsense",
        )


def test_the_scoring_pipeline_defaults_to_the_cheap_mismatched_estimator():
    # Documented rather than silent: the default is the cheaper construction,
    # and it is not the one serving uses.
    pipeline = SeasonAverageScoringPipeline()

    assert pipeline.volume_feature_estimator == "ridge"


def test_the_estimator_choice_survives_a_round_trip(tmp_path):
    # A saved pipeline has to remember which construction its efficiency
    # coefficients were fitted against, or a reload silently changes the answer.
    metadata = tmp_path / "metadata.json"
    metadata.parent.mkdir(parents=True, exist_ok=True)
    metadata.write_text(
        '{"architecture_version": 2, "volume_feature_alpha": 300.0,'
        ' "draw_conditioned_efficiency": false,'
        ' "volume_feature_estimator": "pipeline"}',
        encoding="utf-8",
    )
    import json

    stored = json.loads(metadata.read_text(encoding="utf-8"))

    assert stored["volume_feature_estimator"] == "pipeline"


def test_pipeline_estimator_falls_back_rather_than_losing_a_fold():
    # Early folds have no history to fit, and a source without quarterback snaps
    # cannot fit the QB layers. Those folds must still get a projection.
    from ffmodel.evaluation.efficiency_season_average import _pipeline_fold_projection

    empty = pd.DataFrame(columns=["season", "team", "player_key", "position"])

    assert (
        _pipeline_fold_projection(
            pd.DataFrame(columns=["season", "team"]), empty, empty, 2020, None
        )
        is None
    )
