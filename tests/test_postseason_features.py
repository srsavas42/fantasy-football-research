"""Postseason usage rides alongside the regular season, never inside it.

Season Y's playoffs finish before season Y+1 opens, so they are legitimately
available to a Y+1 preseason projection. What they must not do is join the
regular-season aggregates: ``games``, ``team_games`` and
``observed_availability`` all count regular-season games, and the team totals
every usage share divides by are built from the same rows. Letting playoff weeks
into those would push availability past one for the 40% of teams that qualify
and would feed player numerators and team denominators at different rates.
"""

import numpy as np
import pandas as pd
import pytest

from ffmodel.data import ingest
from ffmodel.features.season_average import (
    POSTSEASON_FEATURES,
    _merge_postseason_features,
    postseason_player_usage,
)


def _post_weeks() -> pd.DataFrame:
    rows = []
    for week in (19, 20):
        rows += [
            {"season": 2023, "week": week, "team": "KC", "player_id": "wr1",
             "player_name": "WR One", "position": "WR", "targets": 8.0,
             "rush_att": 0.0, "pass_att": 0.0},
            {"season": 2023, "week": week, "team": "KC", "player_id": "te1",
             "player_name": "TE One", "position": "TE", "targets": 12.0,
             "rush_att": 0.0, "pass_att": 0.0},
            {"season": 2023, "week": week, "team": "KC", "player_id": "rb1",
             "player_name": "RB One", "position": "RB", "targets": 0.0,
             "rush_att": 15.0, "pass_att": 0.0},
        ]
    return pd.DataFrame(rows)


def test_shares_are_relative_to_the_postseason_team_total():
    usage = postseason_player_usage(_post_weeks())

    assert usage["post_target_share"].sum() == pytest.approx(1.0)
    assert usage["post_carry_share"].sum() == pytest.approx(1.0)
    # 12 of 20 targets went to the tight end across both games.
    tight_end = usage.loc[usage["player_key"].eq("te1"), "post_target_share"]
    assert tight_end.iloc[0] == pytest.approx(0.6)


def test_games_are_counted_so_one_playoff_game_can_be_discounted():
    usage = postseason_player_usage(_post_weeks())

    assert set(usage["post_games"]) == {2}


def test_an_empty_postseason_yields_the_declared_columns():
    usage = postseason_player_usage(pd.DataFrame())

    assert list(usage.columns) == [
        "season", "team", "player_key",
        "post_pass_attempt_share", "post_target_share",
        "post_carry_share", "post_games",
    ]


def _rows_for(prior_team: str) -> pd.DataFrame:
    return pd.DataFrame(
        [{"season": 2024, "team": "KC", "prior_team": prior_team, "player_key": "wr1"}]
    )


def test_a_missing_postseason_is_flagged_rather_than_zero_filled():
    # A team with no postseason row did not qualify. Filling a zero share would
    # read as "played in the postseason and earned nothing", which is a
    # different fact and one the model would be entitled to learn from.
    usage = postseason_player_usage(_post_weeks())

    absent = _merge_postseason_features(_rows_for("BUF"), usage)

    assert absent["prior_post_available"].iloc[0] == 0.0
    assert pd.isna(absent["prior_post_target_share"].iloc[0])
    assert absent["prior_post_games"].iloc[0] == 0.0


def test_a_present_postseason_lands_on_the_following_season():
    usage = postseason_player_usage(_post_weeks())

    present = _merge_postseason_features(_rows_for("KC"), usage)

    assert present["prior_post_available"].iloc[0] == 1.0
    assert present["prior_post_target_share"].iloc[0] == pytest.approx(0.4)
    assert present["prior_post_games"].iloc[0] == 2.0


def test_no_postseason_source_still_declares_every_feature():
    absent = _merge_postseason_features(_rows_for("KC"), pd.DataFrame())

    for feature in POSTSEASON_FEATURES:
        assert feature in absent.columns
    assert absent["prior_post_available"].iloc[0] == 0.0


def test_load_weekly_rejects_an_unknown_season_type():
    with pytest.raises(ValueError, match="season_type"):
        ingest.load_weekly([2023], season_type="PRE")


def test_the_pipeline_flag_is_on_and_reaches_only_skill_role_models():
    from ffmodel.models.volume_season_average import SeasonAverageVolumePipeline

    assert SeasonAverageVolumePipeline().postseason_role_features is True

    pipeline = SeasonAverageVolumePipeline(postseason_role_features=True)
    pipeline._enable_postseason_role_features()

    for model in (
        pipeline.snap_model,
        pipeline.target_role_model,
        pipeline.carry_eligibility_model,
        pipeline.target_model,
        pipeline.carry_model,
    ):
        assert set(POSTSEASON_FEATURES) <= set(model.extra_features)

    # Availability must not see it: qualifying for the postseason is a fact
    # about the team's quality, and it would stand in for staying healthy.
    assert not set(POSTSEASON_FEATURES) & set(
        pipeline.availability_model.extra_features
    )
    # Nor the quarterback room. That one is measured rather than argued: handing
    # it over cost 3.39% pass-attempt MAE across all three holdouts, against a
    # gate that allows no pass-stream regression beyond 0.5%.
    assert not set(POSTSEASON_FEATURES) & set(
        pipeline.workload_model.extra_features
    )
