"""Features must be computable at projection time, not only at fit time.

A feature that varies across training rows and is always missing when the model
is actually used is fit on one distribution and applied to another. The model
fills the gap with a training median, so nothing raises: the feature silently
becomes a constant and its fitted coefficient is applied to data that never
looked like the data it was estimated from.
"""

import numpy as np
import pandas as pd
import pytest

from ffmodel.features.season_pathways import add_player_pathway_features


def _player_seasons(seasons) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "season": list(seasons),
            "team": ["BAL"] * len(seasons),
            "player_key": ["qb-bal"] * len(seasons),
            "position": ["QB"] * len(seasons),
            "prior_snap_share": np.linspace(0.4, 0.8, len(seasons)),
            "prior_availability": [0.9] * len(seasons),
            "prior_pass_role": np.linspace(0.5, 0.9, len(seasons)),
            "prior_target_role": [0.0] * len(seasons),
            "prior_carry_role": [0.02] * len(seasons),
        }
    )


def test_trends_need_a_consecutive_prior_row_in_the_same_frame():
    # With consecutive seasons present the trend is a real difference...
    multi = add_player_pathway_features(_player_seasons([2022, 2023, 2024]))

    assert multi["prior_snap_share_trend"].notna().sum() == 2


@pytest.mark.xfail(
    reason=(
        "Known gap: a projection frame holds one season per player, so the "
        "groupby-shift that builds *_trend has no prior row and every trend is "
        "NaN. prior_snap_share_trend is consumed by SeasonTargetRoleModel, so it "
        "varies during fitting and is always absent at projection time. Fixing "
        "it means computing pathway features over history plus the projection "
        "season and then filtering, rather than over the projection alone."
    ),
    strict=True,
)
def test_trend_is_available_on_a_single_season_projection_frame():
    # ...but a projection frame contains exactly one season per player.
    single = add_player_pathway_features(_player_seasons([2026]))

    assert single["prior_snap_share_trend"].notna().all()


def test_the_ewma_history_column_still_survives_a_single_season():
    # The 3yr EWMA has min_periods=1, so unlike the trend it is defined from one
    # row. Only the differenced features are lost.
    single = add_player_pathway_features(_player_seasons([2026]))

    assert single["prior_snap_share_3yr"].notna().all()
