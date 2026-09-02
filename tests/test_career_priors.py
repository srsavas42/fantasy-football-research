"""The career priors are a leak risk and a silent-no-op risk at once.

Leak: the value on season Y must summarise seasons strictly before Y. The
efficiency frame is indexed by the season whose history it describes and then
lagged by one, so an off-by-one here would train every response on its own
outcome and look like a spectacular improvement.

Silent no-op: the accumulation is exposure-weighted, and the whole reason for
building it is that the rate-EWMA already in the tree is not. A version that
averages per-season rates would pass a naive smoke test while discarding the
gain -- a 12-target season and a 120-target season would count the same.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ffmodel.features.season_efficiency import (
    CAREER_DECAY,
    CAREER_EFFICIENCY_FEATURES,
    _decayed_history,
    add_career_efficiency_priors,
)


def _frame(seasons, yards, targets):
    return pd.DataFrame({
        "season": seasons,
        "player_key": ["a"] * len(seasons),
        "position": ["WR"] * len(seasons),
        "targets": [float(t) for t in targets],
        "rec_yds": [float(y) for y in yards],
        "shrunk_rec_yards_per_target": [
            y / t if t else np.nan for y, t in zip(yards, targets)
        ],
    })


def test_a_players_first_season_has_no_history():
    got = add_career_efficiency_priors(_frame([2019, 2020], [800, 900], [100, 100]))
    assert np.isnan(got.career_rec_yards_per_target.iloc[0])
    assert np.isfinite(got.career_rec_yards_per_target.iloc[1])


def test_the_value_uses_only_strictly_earlier_seasons():
    """An off-by-one here is the response leaking into its own predictor."""
    got = add_career_efficiency_priors(_frame([2019, 2020], [800, 5000], [100, 100]))
    # 2020's enormous season must not touch 2020's own career prior.
    assert got.career_rec_yards_per_target_exposure.iloc[1] == 100.0 * CAREER_DECAY


def test_a_gap_year_decays_by_years_elapsed_not_by_one_step():
    got = add_career_efficiency_priors(
        _frame([2018, 2020], [1000, 1000], [100, 100])
    )
    # One season, two years back: 100 * 0.7**2, not 100 * 0.7.
    assert got.career_rec_yards_per_target_exposure.iloc[1] == (
        100.0 * CAREER_DECAY**2
    )


def test_seasons_are_weighted_by_exposure_not_counted_equally():
    """The point of the construction; a rate EWMA would fail this.

    Compared as a swap rather than against a threshold, because both arms then
    take identical shrinkage and the comparison isolates the weighting. Under
    equal weighting the two arms would land in the same place; under exposure
    weighting the 150-target season carries each of them.
    """
    big_season_good = add_career_efficiency_priors(
        # 2018: 10 targets at 2.0.  2019: 150 targets at 10.0.
        _frame([2018, 2019, 2020], [20, 1500, 0], [10, 150, 1])
    ).career_rec_yards_per_target.iloc[2]
    big_season_bad = add_career_efficiency_priors(
        # The rates swapped, the exposures left where they were.
        _frame([2018, 2019, 2020], [100, 300, 0], [10, 150, 1])
    ).career_rec_yards_per_target.iloc[2]
    assert big_season_good > big_season_bad
    # And the gap is wide: the large season sets the prior almost by itself.
    assert big_season_good - big_season_bad > 4.0


def test_more_history_raises_the_exposure_behind_the_prior():
    """Which is what shrinks it less; a veteran should be pooled less than a rookie."""
    one = add_career_efficiency_priors(_frame([2019, 2020], [900] * 2, [100] * 2))
    many = add_career_efficiency_priors(
        _frame([2016, 2017, 2018, 2019, 2020], [900] * 5, [100] * 5)
    )
    assert (
        many.career_rec_yards_per_target_exposure.iloc[-1]
        > one.career_rec_yards_per_target_exposure.iloc[-1]
    )


def test_variable_history_needs_no_special_casing():
    """A second-year player's career prior is his one-season prior, not missing."""
    got = add_career_efficiency_priors(_frame([2019, 2020], [900, 900], [100, 100]))
    assert np.isfinite(got.career_rec_yards_per_target.iloc[1])


def test_decayed_history_is_empty_before_any_observation():
    values = pd.Series([np.nan, 10.0, 20.0])
    seasons = pd.Series([2019, 2020, 2021])
    got = _decayed_history(values, seasons, 0.7)
    assert np.isnan(got[0])          # nothing seen yet
    assert np.isnan(got[1])          # the 2019 row was not observed
    assert got[2] == 10.0 * 0.7      # only 2020 contributes


def test_every_response_gets_a_career_feature_name():
    assert "prior_rec_yards_per_target_career" in CAREER_EFFICIENCY_FEATURES
    assert "prior_rush_yards_per_carry_career" in CAREER_EFFICIENCY_FEATURES
    assert len(CAREER_EFFICIENCY_FEATURES) == len(set(CAREER_EFFICIENCY_FEATURES))


def test_only_the_two_responses_that_cleared_the_gate_carry_it():
    """Five of seven were rejected; adding them anyway would be unmeasured."""
    from ffmodel.models.efficiency_season_average import EFFICIENCY_MODEL_BY_TARGET

    promoted = {"rush_yards_per_carry", "pass_completion_rate"}
    for target, spec in EFFICIENCY_MODEL_BY_TARGET.items():
        carries = f"prior_{target}_career" in spec.advanced_features
        assert carries == (target in promoted), target


def test_the_promoted_features_are_the_response_s_own_history():
    """A crossed name would fit yards per carry to completion-rate history."""
    from ffmodel.models.efficiency_season_average import EFFICIENCY_MODEL_BY_TARGET

    for target in ("rush_yards_per_carry", "pass_completion_rate"):
        spec = EFFICIENCY_MODEL_BY_TARGET[target]
        career = [f for f in spec.advanced_features if f.endswith("_career")]
        assert career == [f"prior_{target}_career"]
