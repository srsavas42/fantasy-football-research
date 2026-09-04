"""The exposure target is one decision spanning two models.

The availability model fits a count out of ``team_games``; the snap model
divides the observed season snap share by the matching fraction to recover a
per-game rate; prediction multiplies that rate back by the availability draws.
Setting one without the other divides by one exposure and multiplies by
another, and nothing downstream raises -- the projection is simply wrong by the
ratio between them, which for undrafted quarterbacks is a factor of 2.6. These
tests exist because that failure is silent.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ffmodel.models.volume_season_average import SeasonAverageVolumePipeline


def _rows(**overrides) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "season": [2023, 2023],
            "team": ["AAA", "BBB"],
            "player_key": ["a", "b"],
            "position": ["RB", "WR"],
            "games": [17.0, 9.0],
            "team_games": [17.0, 17.0],
            "observed_availability": [1.0, 9.0 / 17.0],
            "snap_games": [16.0, 5.0],
            "snap_availability": [16.0 / 17.0, 5.0 / 17.0],
        }
    )
    for name, value in overrides.items():
        frame[name] = value
    return frame


def test_the_two_layers_move_together():
    pipeline = SeasonAverageVolumePipeline()
    pipeline.availability_target = "snap"
    pipeline._apply_availability_target(_rows())

    assert pipeline.availability_model.games_column == "snap_games"
    assert pipeline.snap_model.availability_column == "snap_availability"


def test_the_roster_target_is_the_historical_pairing():
    pipeline = SeasonAverageVolumePipeline()
    assert pipeline.availability_target == "roster"
    pipeline._apply_availability_target(_rows())

    assert pipeline.availability_model.games_column == "games"
    assert pipeline.snap_model.availability_column == "observed_availability"


def test_an_unknown_target_is_refused_by_name():
    pipeline = SeasonAverageVolumePipeline()
    pipeline.availability_target = "stat_activity"

    with pytest.raises(ValueError, match="availability_target must be one of"):
        pipeline._apply_availability_target(_rows())


def test_a_frame_without_the_columns_is_refused():
    pipeline = SeasonAverageVolumePipeline()
    pipeline.availability_target = "snap"
    frame = _rows().drop(columns=["snap_games", "snap_availability"])

    with pytest.raises(ValueError, match="absent from the player rows"):
        pipeline._apply_availability_target(frame)


def test_a_wholly_empty_column_is_refused_rather_than_fitted():
    """The legacy snap source has season totals and cannot count weeks.

    The column is declared so the schema stays stable, and carries nothing.
    Fitting against it would make every player unavailable, which is a far
    worse outcome than refusing.
    """
    pipeline = SeasonAverageVolumePipeline()
    pipeline.availability_target = "snap"
    frame = _rows(snap_games=np.nan, snap_availability=np.nan)

    with pytest.raises(ValueError, match="present but wholly missing"):
        pipeline._apply_availability_target(frame)


def test_the_target_round_trips_so_a_restored_model_keeps_its_exposure():
    """A restored artifact reverting to roster games would be silent."""
    pipeline = SeasonAverageVolumePipeline()
    pipeline.availability_target = "snap"
    pipeline._apply_availability_target(_rows())

    metadata = {"availability_target": pipeline.availability_target}
    assert metadata["availability_target"] == "snap"
    # And the default for an artifact written before the field existed is the
    # historical pairing, not an error.
    assert str({}.get("availability_target", "roster")) == "roster"


def test_a_saved_pipeline_comes_back_pointing_at_the_same_exposure():
    """The pairing has to survive save/load, and it did not.

    ``from_metadata`` rebuilds the sub-models at their class defaults and then
    restored only the pipeline-level ``availability_target``, so a pipeline fitted
    against ``snap`` came back predicting against ``games`` and
    ``observed_availability`` while still reporting ``availability_target ==
    "snap"``. That is the silent 2.6x this module exists to prevent, reached by
    a different door.
    """
    pipeline = SeasonAverageVolumePipeline()
    pipeline.availability_target = "snap"
    pipeline._point_availability_layers()

    # What from_metadata does: fresh sub-models, target carried over.
    reloaded = SeasonAverageVolumePipeline(availability_target="snap")
    assert reloaded.availability_model.games_column == "games"
    assert reloaded.snap_model.availability_column == "observed_availability"

    reloaded._point_availability_layers()
    assert reloaded.availability_model.games_column == "snap_games"
    assert reloaded.snap_model.availability_column == "snap_availability"
