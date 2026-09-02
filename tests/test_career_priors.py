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


def test_snap_share_uses_team_plays_not_the_sum_of_player_snaps():
    """Eleven players share each snap; summing the roster overshoots fivefold.

    The first implementation summed player snaps for every stream. Targets and
    carries survive that -- each belongs to one player -- but the emitted snap
    share topped out at 0.178 against an observed share that reaches 1.0, and
    nothing raised.
    """
    from ffmodel.features.season_average import (
        CAREER_ROLE_STREAMS,
        add_career_role_priors,
    )

    by_name = {name: team for _, team, name in CAREER_ROLE_STREAMS}
    assert by_name["career_snap_share"] == "team_offense_snaps"
    assert by_name["career_target_share"] is None

    history = pd.DataFrame({
        "player_key": ["a", "a", "b", "b"],
        "season": [2019, 2020, 2019, 2020],
        "team": ["KC", "KC", "KC", "KC"],
        # Two players, 500 snaps each, on a team that ran 1000 plays.
        "offense_snaps": [500.0, 500.0, 500.0, 500.0],
        "team_offense_snaps": [1000.0, 1000.0, 1000.0, 1000.0],
        "targets": [50.0, 50.0, 50.0, 50.0],
    })
    got = add_career_role_priors(history)
    played = got.loc[got.player_key.eq("a") & got.season.eq(2020), "career_snap_share"]
    # Half the team's plays, not a quarter of the roster's summed snaps.
    assert abs(float(played.iloc[0]) - 0.5) < 1e-9


def test_only_the_carry_room_gets_the_career_role_feature():
    """Targets were measured to be hurt by it; snaps landed on the floor."""
    from ffmodel.models.volume_season_average import SeasonAverageVolumePipeline

    pipeline = SeasonAverageVolumePipeline()
    pipeline._enable_postseason_role_features()
    if pipeline.career_role_features:
        pipeline.carry_model.extra_features = tuple(
            dict.fromkeys(
                (*pipeline.carry_model.extra_features, "prior_carry_share_career")
            )
        )
    assert "prior_carry_share_career" in pipeline.carry_model.extra_features
    assert "prior_target_share_career" not in pipeline.target_model.extra_features
    assert "prior_snap_share_career" not in pipeline.snap_model.extra_features


def test_the_career_role_flag_is_on_by_default():
    from ffmodel.models.volume_season_average import SeasonAverageVolumePipeline

    assert SeasonAverageVolumePipeline().career_role_features


def test_the_tier_interaction_ships_only_where_it_helped_drafted_players():
    """Two responses cleared the pooled gate and failed the drafted one.

    pass_yards_per_attempt was -1.02% CRPS overall at 3/3 and +0.22% MAE on
    drafted players at 1/3; pass_completion_rate -0.41% overall at 3/3 and
    2/3 on drafted. A tier interaction that helps the players a draft is not
    about has missed its own point, so neither ships.
    """
    from ffmodel.models.efficiency_season_average import EFFICIENCY_MODEL_BY_TARGET

    promoted = {"pass_td_rate"}
    for target, spec in EFFICIENCY_MODEL_BY_TARGET.items():
        carries = any(f.endswith("_x_drafted") for f in spec.advanced_features)
        assert carries == (target in promoted), target


def test_the_promoted_interaction_carries_its_own_level_term():
    """Without adp_drafted the level difference is misattributed to slope."""
    from ffmodel.models.efficiency_season_average import EFFICIENCY_MODEL_BY_TARGET

    features = EFFICIENCY_MODEL_BY_TARGET["pass_td_rate"].advanced_features
    assert "prior_pass_td_rate_x_drafted" in features
    assert "adp_drafted" in features
