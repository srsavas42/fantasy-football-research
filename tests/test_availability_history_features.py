"""The availability layer reads the player's own history, and says so if it can't.

``prior_availability_3yr`` has been built for every row since the pathway
features landed and read by nothing. The failure mode this guards is the one
the package has documented twice: ``_matrix`` keeps only the features actually
present in the frame, so a cache built before the pathway work would fit the
single-season layer, report the new configuration, and raise nothing.
"""

import pandas as pd
import pytest

from ffmodel.models.season_availability import (
    AVAILABILITY_FEATURES,
    AVAILABILITY_HISTORY_FEATURES,
)
from ffmodel.models.volume_season_average import SeasonAverageVolumePipeline


def _player_rows(*, with_history: bool) -> pd.DataFrame:
    rows = pd.DataFrame(
        {
            "season": [2023, 2023, 2023, 2023],
            "team": ["SEA", "SEA", "KC", "KC"],
            "player_key": ["a", "b", "c", "d"],
            "position": ["RB", "WR", "QB", "TE"],
            "prior_availability": [0.9, 0.5, 1.0, 0.7],
            "games": [16, 9, 17, 12],
            "team_games": [17, 17, 17, 17],
        }
    )
    if with_history:
        rows["prior_availability_3yr"] = [0.88, 0.62, 0.97, 0.74]
    return rows


def test_history_column_is_not_already_a_base_feature() -> None:
    """Otherwise the flag would be a no-op that looks like a change."""
    assert AVAILABILITY_HISTORY_FEATURES == ("prior_availability_3yr",)
    for name in AVAILABILITY_HISTORY_FEATURES:
        assert name not in AVAILABILITY_FEATURES


def test_enabled_by_default_and_appended_to_the_availability_design() -> None:
    pipeline = SeasonAverageVolumePipeline()
    assert pipeline.availability_history_features is True
    assert pipeline.availability_model.extra_features == ()

    pipeline._enable_availability_history(_player_rows(with_history=True))
    for name in AVAILABILITY_HISTORY_FEATURES:
        assert name in pipeline.availability_model.extra_features


def test_enabling_twice_does_not_duplicate() -> None:
    pipeline = SeasonAverageVolumePipeline()
    rows = _player_rows(with_history=True)
    pipeline._enable_availability_history(rows)
    pipeline._enable_availability_history(rows)
    extra = pipeline.availability_model.extra_features
    assert len(extra) == len(set(extra))


def test_a_frame_without_the_column_raises_rather_than_dropping_it() -> None:
    pipeline = SeasonAverageVolumePipeline()
    with pytest.raises(ValueError, match="prior_availability_3yr"):
        pipeline._enable_availability_history(_player_rows(with_history=False))


def test_the_paired_arm_leaves_the_design_alone() -> None:
    pipeline = SeasonAverageVolumePipeline(availability_history_features=False)
    assert pipeline.availability_history_features is False
    assert pipeline.availability_model.extra_features == ()


def test_history_reaches_the_fitted_design(monkeypatch) -> None:
    """End to end through ``_matrix``, which is where a missing column vanishes."""
    pipeline = SeasonAverageVolumePipeline()
    rows = _player_rows(with_history=True)
    pipeline._enable_availability_history(rows)
    model = pipeline.availability_model
    prepared = model._prepare(rows)
    model._matrix(prepared, fit=True)
    assert "prior_availability_3yr" in model.feature_names
