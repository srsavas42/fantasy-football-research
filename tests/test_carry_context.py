"""The short-yardage promotion has three quiet ways to do nothing.

The column can fail to reach the panel, in which case the spec computes a ratio
of a missing numerator and the feature is silently absent. The ridge design
keeps only the features present in the frame, so a missing column drops the
covariate without a word and the model fits as though it were off -- which is
how the teammate-quality feature's first walk-forward came back identical to its
baseline. And the merge can turn absent coverage into a zero, which does not
look like an error at all: it says every back ran no short yardage rather than
that nobody looked, and a zero is a legitimate value here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ffmodel.features.carry_context import (
    SHORT_YARDAGE_DISTANCE,
    merge_carry_context,
    _season_carry_context,
)
from ffmodel.features.season_efficiency import (
    ADVANCED_EFFICIENCY_FEATURES,
    EFFICIENCY_BY_NAME,
    _AGGREGATE_COLUMNS,
)
from ffmodel.models.efficiency_season_average import EFFICIENCY_MODEL_BY_TARGET


def _pbp() -> pd.DataFrame:
    return pd.DataFrame({
        "season_type": ["REG"] * 5 + ["POST"],
        "week": [1, 1, 2, 2, 2, 3],
        "rush_attempt": [1, 1, 1, 1, 0, 1],
        "rusher_player_id": ["a", "a", "a", "b", "c", "a"],
        "ydstogo": [1, 10, 2, 7, 1, 1],
    })


def test_short_yardage_is_counted_by_distance_not_by_yardline():
    got = _season_carry_context(_pbp(), 2023)
    week1 = got[(got.week == 1) & (got.player_id == "a")]
    assert float(week1.rush_short_yardage_att.iloc[0]) == 1.0  # the 1, not the 10


def test_the_postseason_is_excluded():
    got = _season_carry_context(_pbp(), 2023)
    assert 3 not in set(got.week)


def test_a_pass_play_is_not_a_carry():
    got = _season_carry_context(_pbp(), 2023)
    assert "c" not in set(got.player_id)


def test_the_cut_is_inclusive_at_the_stated_distance():
    got = _season_carry_context(_pbp(), 2023)
    week2 = got[(got.week == 2) & (got.player_id == "a")]
    assert SHORT_YARDAGE_DISTANCE == 2
    assert float(week2.rush_short_yardage_att.iloc[0]) == 1.0  # ydstogo == 2 counts


def test_a_covered_week_with_no_short_carry_is_zero_not_missing():
    weeks = pd.DataFrame({
        "season": [2023, 2023], "week": [1, 2], "player_id": ["b", "b"],
    })
    context = _season_carry_context(_pbp(), 2023)
    got = merge_carry_context(weeks, context)
    assert float(got.rush_short_yardage_att.iloc[1]) == 0.0


def test_an_uncovered_season_stays_missing_rather_than_becoming_zero():
    """A zero would claim the back never ran in short yardage."""
    weeks = pd.DataFrame({"season": [2011], "week": [1], "player_id": ["a"]})
    got = merge_carry_context(weeks, _season_carry_context(_pbp(), 2023))
    assert np.isnan(got.rush_short_yardage_att.iloc[0])


def test_no_context_at_all_still_yields_the_column():
    weeks = pd.DataFrame({"season": [2023], "week": [1], "player_id": ["a"]})
    got = merge_carry_context(weeks, pd.DataFrame())
    assert "rush_short_yardage_att" in got
    assert np.isnan(got.rush_short_yardage_att.iloc[0])


def test_the_numerator_reaches_the_season_aggregator():
    assert "rush_short_yardage_att" in _AGGREGATE_COLUMNS


def test_the_share_is_lagged_before_a_model_can_see_it():
    assert "prior_rush_short_yardage_share" in ADVANCED_EFFICIENCY_FEATURES


def test_the_denominator_is_carries_so_the_feature_is_a_share():
    spec = EFFICIENCY_BY_NAME["rush_short_yardage_share"]
    assert (spec.numerator, spec.denominator) == ("rush_short_yardage_att", "rush_att")


def test_the_rushing_response_actually_admits_it():
    """Otherwise the ridge design drops it and the promotion is a no-op."""
    spec = EFFICIENCY_MODEL_BY_TARGET["rush_yards_per_carry"]
    assert "prior_rush_short_yardage_share" in spec.advanced_features


def test_only_the_rushing_response_carries_it():
    for target, spec in EFFICIENCY_MODEL_BY_TARGET.items():
        if target == "rush_yards_per_carry":
            continue
        assert "prior_rush_short_yardage_share" not in spec.advanced_features
