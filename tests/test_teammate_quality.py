"""Cross-positional teammate quality: who is throwing the ball.

Every efficiency spec has an empty feature list, so receiver efficiency is
modelled from the receiver's own history and nothing about his quarterback.
``prior_rec_team_quality_signal`` does not fill the gap -- it ranks a player's
own quality within his team, which is relative standing, not teammate quality.
"""

import numpy as np
import pandas as pd
import pytest

from ffmodel.features.season_efficiency import add_teammate_quality_features


def _roster():
    """Two teams. KC lists its starter; BUF lists nobody and needs the fallback."""
    return pd.DataFrame(
        {
            "season": 2024,
            "team": ["KC", "KC", "KC", "BUF", "BUF", "BUF"],
            "position": ["QB", "QB", "WR", "QB", "QB", "RB"],
            "player_key": ["kc-qb1", "kc-qb2", "kc-wr1", "buf-qb1", "buf-qb2", "buf-rb1"],
            "qb_listed_starter": [1, 0, np.nan, np.nan, np.nan, np.nan],
            "qb_depth_rank": [1.0, 2.0, np.nan, 2.0, 1.0, np.nan],
            "prior_pass_quality_signal": [0.40, -0.30, np.nan, -0.25, 0.15, np.nan],
        }
    )


def test_skill_players_inherit_the_listed_starter():
    out = add_teammate_quality_features(_roster())

    receiver = out.loc[out.player_key.eq("kc-wr1"), "teammate_qb_quality_signal"]
    assert receiver.iloc[0] == pytest.approx(0.40)


def test_the_fallback_is_the_depth_chart_when_nobody_is_listed():
    """BUF lists no starter, and its shallower entry is qb2, not qb1."""
    out = add_teammate_quality_features(_roster())

    back = out.loc[out.player_key.eq("buf-rb1"), "teammate_qb_quality_signal"]
    assert back.iloc[0] == pytest.approx(0.15)


def test_a_quarterback_does_not_read_his_own_quality_as_a_teammates():
    """Otherwise one signal enters the model twice under two names."""
    out = add_teammate_quality_features(_roster())

    assert out.loc[out.position.eq("QB"), "teammate_qb_quality_signal"].isna().all()


def test_it_refuses_to_run_before_the_quality_composite_exists():
    rows = _roster().drop(columns=["prior_pass_quality_signal"])

    with pytest.raises(ValueError, match="prior_pass_quality_signal"):
        add_teammate_quality_features(rows)


def test_it_ignores_primary_qb_entirely():
    """The listed starter and depth rank are preseason artifacts. ``primary_qb``
    is not -- it is derived from who actually played -- so a feature reading it
    would score well in validation and be unavailable when serving.

    Checked by behaviour rather than by grepping the source: flipping
    ``primary_qb`` to contradict the depth chart must change nothing.
    """
    rows = _roster()
    honest = add_teammate_quality_features(rows.assign(primary_qb=[1, 0, 0, 0, 1, 0]))
    inverted = add_teammate_quality_features(rows.assign(primary_qb=[0, 1, 0, 1, 0, 0]))

    pd.testing.assert_series_equal(
        honest["teammate_qb_quality_signal"], inverted["teammate_qb_quality_signal"]
    )


def test_the_flag_reaches_the_receiving_responses_and_one_rushing_rate():
    """Rushing is admitted one response at a time, on a measured mechanism.

    This test used to exclude rushing entirely, on the grounds that the
    mechanism by which a passer changes yards per carry is indirect at best and
    that testing where there is no story to tell is how a feature earns a fold
    win by chance. Both halves of that still hold, and
    scripts/screen_qb_context_rb.py bears out the first: yards per carry
    measures +0.065 (p=0.12), which is nothing.

    rush_td_rate is the exception it turned up, and it has the story the
    original reasoning asked for. A back's touchdowns per carry is a red-zone
    quantity, a better passer reaches the red zone more often, and the measured
    partial correlation is +0.171 (p=3.8e-05) -- stronger than the receiving
    effect already in the set, and flat across three control sets including the
    back's own prior_rush_short_yardage_share.

    What the screen found nothing for stays out: carry share (-0.024),
    target share (-0.003), yards per carry (+0.065).
    """
    from ffmodel.models.efficiency_season_average import (
        EFFICIENCY_MODEL_SPECS,
        TEAMMATE_QUALITY_TARGETS,
    )

    assert set(TEAMMATE_QUALITY_TARGETS) == {
        "rec_catch_rate",
        "rec_yards_per_target",
        "rec_td_rate",
        "rush_td_rate",
    }
    assert "rush_yards_per_carry" not in TEAMMATE_QUALITY_TARGETS
    targets = {spec.target for spec in EFFICIENCY_MODEL_SPECS}
    assert set(TEAMMATE_QUALITY_TARGETS) <= targets


def test_it_is_off_by_default():
    """Every efficiency spec has run with an empty feature list since the
    pipeline was written. This is the first thing any of them learns about a
    player's teammates, so it stays off until a gate says otherwise."""
    from ffmodel.models.efficiency_season_average import (
        SeasonAveragePosteriorEfficiencyPipeline,
    )
    from ffmodel.models.season_scoring import SeasonAverageScoringPipeline

    assert not SeasonAveragePosteriorEfficiencyPipeline().teammate_quality_features
    assert SeasonAverageScoringPipeline().teammate_quality_features is None


def test_enabling_it_adds_exactly_one_feature_to_each_receiving_spec():
    from dataclasses import replace

    from ffmodel.models.efficiency_season_average import (
        EFFICIENCY_MODEL_SPECS,
        TEAMMATE_QUALITY_TARGETS,
    )

    for spec in EFFICIENCY_MODEL_SPECS:
        if spec.target not in TEAMMATE_QUALITY_TARGETS:
            continue
        widened = replace(
            spec,
            advanced_features=(*spec.advanced_features, "teammate_qb_quality_signal"),
        )
        assert len(widened.advanced_features) == len(spec.advanced_features) + 1
        assert widened.advanced_features[-1] == "teammate_qb_quality_signal"


def test_asking_for_the_feature_without_the_column_fails_loudly():
    """The null result that exposed this.

    ``_matrix`` keeps only features present in the frame, so an absent one is
    dropped silently and the model fits as though the flag were off. The first
    walk-forward of this feature came back identical to its baseline to five
    decimals on every metric across three folds, because the cached frames were
    built before the column existed. Nothing in the run said so.
    """
    from ffmodel.models.efficiency_season_average import (
        SeasonAveragePosteriorEfficiencyPipeline,
    )

    pipeline = SeasonAveragePosteriorEfficiencyPipeline(teammate_quality_features=True)
    rows = pd.DataFrame({"season": [2024], "team": ["KC"], "position": ["WR"]})

    with pytest.raises(ValueError, match="teammate_qb_quality_signal"):
        pipeline.fit(rows)


def test_the_guard_stays_out_of_the_way_when_the_flag_is_off():
    """Frames without the column are ordinary, and most are."""
    from ffmodel.models.efficiency_season_average import (
        SeasonAveragePosteriorEfficiencyPipeline,
    )

    pipeline = SeasonAveragePosteriorEfficiencyPipeline()
    rows = pd.DataFrame({"season": [2024], "team": ["KC"], "position": ["WR"]})

    # Fails later for want of real data, but not on the teammate guard.
    with pytest.raises(Exception) as caught:
        pipeline.fit(rows)
    assert "teammate_qb_quality_signal" not in str(caught.value)
