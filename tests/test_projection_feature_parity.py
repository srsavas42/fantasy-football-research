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


def test_a_single_season_frame_now_raises_instead_of_going_quietly_constant():
    """The gap is closed by refusing, not by inventing a value.

    A projection frame holds one season per player, so the groupby-shift that
    builds ``*_trend`` has no prior row and every trend is NaN.
    ``prior_snap_share_trend`` is consumed by ``SeasonTargetRoleModel``, which
    fills missing values with a training median — so the feature varies during
    fitting and is a constant at projection time, and nothing says so. There is
    no correct value to fill in here; the caller has to featurize history
    alongside the projection season, and the error says that.
    """
    with pytest.raises(ValueError, match="prior_snap_share_trend"):
        add_player_pathway_features(_player_seasons([2026]))


def test_the_error_names_the_fix():
    with pytest.raises(ValueError, match="Featurize history together"):
        add_player_pathway_features(_player_seasons([2026]))


def test_a_caller_that_does_not_consume_trends_can_opt_out():
    single = add_player_pathway_features(
        _player_seasons([2026]), require_trends=False
    )

    assert single["prior_snap_share_trend"].isna().all()


def test_the_production_path_is_unaffected():
    # History and the projection season featurized together, which is what
    # build_season_average_data does; the trend resolves and nothing raises.
    together = add_player_pathway_features(_player_seasons([2024, 2025, 2026]))

    assert together["prior_snap_share_trend"].notna().sum() == 2


def test_an_empty_frame_is_not_an_error():
    empty = add_player_pathway_features(_player_seasons([]))

    assert len(empty) == 0


def test_the_ewma_history_column_still_survives_a_single_season():
    # The 3yr EWMA has min_periods=1, so unlike the trend it is defined from one
    # row. Only the differenced features are lost, which is why the guard is
    # about trends specifically rather than about single-season frames.
    single = add_player_pathway_features(
        _player_seasons([2026]), require_trends=False
    )

    assert single["prior_snap_share_3yr"].notna().all()


@pytest.mark.xfail(
    reason=(
        "Live train/serve gap, now measured rather than assumed. "
        "build_projection_data's row universe comes from roster_snapshot, which "
        "covers the projection season alone, so add_player_pathway_features "
        "receives a single-season frame and every *_trend is NaN. "
        "SeasonTargetRoleModel consumes prior_snap_share_trend, so it varies "
        "while fitting on a backtest frame and is a training-median constant on "
        "every projection. The previous xfail claimed the production path was "
        "unaffected because history and projection are featurized together; "
        "that is true of build_season_average_data over a season range and "
        "false of build_projection_data. Closing it means giving the projection "
        "frame history rows to difference against and filtering afterwards."
    ),
    strict=True,
)
def test_a_projection_build_resolves_the_trend_it_was_fitted_on():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from test_season_projection import _roster_snapshot

    from ffmodel.features.season_average import build_projection_data

    data = build_projection_data(
        2021,
        roster_snapshot=_roster_snapshot(2021),
        history_seasons=[2018, 2019, 2020],
        source="legacy",
    )

    assert data.player_rows["prior_snap_share_trend"].notna().any()
