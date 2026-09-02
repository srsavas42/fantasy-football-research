"""Fumbles committed are modeled; fumbles lost are what scoring charges for.

The conversion between them is the whole point of the change, and it has two
quiet failure modes. The league share can be hardcoded, in which case a holdout
sees its own season's recoveries. And it can be dropped somewhere between the
pipeline that fits it and the simulator that applies it, in which case the
projections silently double -- a fumble rate used as a lost rate is roughly
twice the truth, and nothing raises.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ffmodel.features.season_efficiency import EFFICIENCY_BY_NAME
from ffmodel.models.efficiency_season_average import (
    EFFICIENCY_MODEL_BY_TARGET,
    LEAGUE_FUMBLE_LOST_SHARE,
    SeasonAveragePosteriorEfficiencyPipeline,
    _fumble_lost_share,
)
from ffmodel.simulation.season_scoring import REQUIRED_EFFICIENCY_TARGETS


def test_the_modeled_response_is_the_committed_rate():
    spec = EFFICIENCY_MODEL_BY_TARGET["fumble_rate"]
    assert spec.numerator == "eff_fumbles"
    assert "fumble_lost_rate" not in EFFICIENCY_MODEL_BY_TARGET


def test_the_lost_rate_is_still_observable_for_scoring_truth():
    """It is no longer fitted, but a season is still graded on it."""
    assert "fumble_lost_rate" in EFFICIENCY_BY_NAME
    assert EFFICIENCY_BY_NAME["fumble_lost_rate"].numerator == "fumbles_lost"
    assert EFFICIENCY_BY_NAME["fumble_rate"].numerator == "fumbles"


def test_scoring_requires_the_new_response():
    assert "fumble_rate" in REQUIRED_EFFICIENCY_TARGETS
    assert "fumble_lost_rate" not in REQUIRED_EFFICIENCY_TARGETS


def test_the_share_is_counted_from_the_rows_it_is_given():
    rows = pd.DataFrame(
        {"eff_fumbles": [10.0, 6.0], "eff_fumbles_lost": [5.0, 4.0]}
    )
    assert _fumble_lost_share(rows) == 9.0 / 16.0


def test_a_frame_without_fumble_counts_falls_back_to_the_league_share():
    assert _fumble_lost_share(pd.DataFrame({"season": [2023]})) == (
        LEAGUE_FUMBLE_LOST_SHARE
    )


def test_no_fumbles_at_all_does_not_divide_by_zero():
    rows = pd.DataFrame({"eff_fumbles": [0.0], "eff_fumbles_lost": [0.0]})
    assert _fumble_lost_share(rows) == LEAGUE_FUMBLE_LOST_SHARE


def test_missing_counts_do_not_defeat_the_guard():
    """``pd.to_numeric(None)`` is nan, not None, so the guard reads columns."""
    rows = pd.DataFrame({"eff_fumbles": [np.nan], "eff_fumbles_lost": [np.nan]})
    assert _fumble_lost_share(rows) == LEAGUE_FUMBLE_LOST_SHARE


def test_the_share_lands_on_the_pipeline_where_scoring_reads_it():
    pipeline = SeasonAveragePosteriorEfficiencyPipeline()
    assert pipeline.fumble_lost_share == LEAGUE_FUMBLE_LOST_SHARE
    # The simulator reads it by attribute off the fitted efficiency model, so
    # the name matters as much as the value.
    assert hasattr(pipeline, "fumble_lost_share")


def test_the_league_share_is_about_a_coin_flip():
    """Measured 49.8% over 2014-2025; a value far from this is a wiring bug."""
    assert 0.45 < LEAGUE_FUMBLE_LOST_SHARE < 0.55
