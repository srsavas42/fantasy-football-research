"""The promoted reserve flag has to actually reach the response it was scored on.

Two ways this promotion fails quietly. ``_matrix`` keeps only the features
present in the frame, so a missing column drops the flag without a word and the
model fits as though it were off -- which is how the teammate-quality feature's
first walk-forward came back identical to its baseline. And the save/load path
rebuilds each response from its saved state, so a field it does not persist
reverts to the current default, and a refit from a reloaded pipeline trains on a
different design than the artifact did.

Neither raises on its own. Both would show up as a promoted feature quietly
doing nothing.
"""

from __future__ import annotations

import pandas as pd
import pytest

from ffmodel.models.efficiency_season_average import (
    EFFICIENCY_MODEL_BY_TARGET,
    RESERVE_EFFICIENCY_FEATURES,
    RESERVE_EFFICIENCY_TARGETS,
    PosteriorSeasonEfficiencyModel,
    SeasonAveragePosteriorEfficiencyPipeline,
)


def test_the_flag_is_on_by_default():
    assert SeasonAveragePosteriorEfficiencyPipeline().reserve_efficiency_features


def test_only_the_response_that_was_measured_gets_it():
    """rec_catch_rate admits no covariates and rec_td_rate was never scored."""
    assert RESERVE_EFFICIENCY_TARGETS == ("rec_yards_per_target",)


def test_a_missing_column_is_an_error_not_a_silent_no_op():
    """Otherwise the flag is on, the feature is dropped, and nothing says so."""
    pipeline = SeasonAveragePosteriorEfficiencyPipeline()
    rows = pd.DataFrame({"season": [2023], "position": ["WR"]})
    with pytest.raises(ValueError, match="roster_reserve is not in"):
        pipeline.fit(rows)


def test_turning_the_flag_off_does_not_demand_the_column():
    pipeline = SeasonAveragePosteriorEfficiencyPipeline(
        reserve_efficiency_features=False
    )
    rows = pd.DataFrame({"season": [2023], "position": ["WR"]})
    # Still fails, but on having no usable rows rather than on the guard.
    with pytest.raises(ValueError) as excinfo:
        pipeline.fit(rows)
    assert "roster_reserve" not in str(excinfo.value)


def test_the_design_carries_the_flag_for_the_promoted_response():
    model = PosteriorSeasonEfficiencyModel(
        EFFICIENCY_MODEL_BY_TARGET["rec_yards_per_target"],
        mean_mode="posterior",
        extra_features=RESERVE_EFFICIENCY_FEATURES,
    )
    assert "roster_reserve" in model._candidates()


def test_an_unpromoted_response_does_not_carry_it():
    model = PosteriorSeasonEfficiencyModel(
        EFFICIENCY_MODEL_BY_TARGET["rush_yards_per_carry"], mean_mode="posterior"
    )
    assert "roster_reserve" not in model._candidates()


def test_extra_features_defaults_empty_so_old_artifacts_reproduce():
    model = PosteriorSeasonEfficiencyModel(
        EFFICIENCY_MODEL_BY_TARGET["rec_yards_per_target"]
    )
    assert model.extra_features == ()
