"""The within-room structure features offered to the target softmax.

The softmax normalises over the whole team roster and never sees a positional
room, so a receiver's standing *within his own room* is information the score
does not carry. These guard what gets offered and that asking for it on a frame
that lacks it fails loudly.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ffmodel.features.season_average import SeasonAverageData
from ffmodel.models.volume_season_average import (
    TARGET_ROOM_FEATURES,
    SeasonAverageVolumePipeline,
)


def test_the_flag_is_off_until_it_clears_the_gate():
    assert not SeasonAverageVolumePipeline().room_structure_features


def test_only_player_varying_candidates_ship():
    """Room- and team-level competition are constants within a room.

    ``prior_target_room_competition`` is ``1 - room_leader``: every receiver on
    a team shares the value, so it cannot separate two players the softmax is
    allocating between. It measured as nothing in the screen and is left out.
    """
    assert "prior_target_role_uncertainty" in TARGET_ROOM_FEATURES
    assert "prior_rec_room_quality_advantage" in TARGET_ROOM_FEATURES
    assert "prior_target_room_competition" not in TARGET_ROOM_FEATURES
    assert "prior_target_team_competition" not in TARGET_ROOM_FEATURES


def _frame(columns: tuple[str, ...]) -> pd.DataFrame:
    """A roster minimal enough to be quick but complete enough to pass preflight.

    ``volume_input_problems`` runs before any feature enablement and needs a
    quarterback with positive snaps, so a two-receiver frame fails on that
    instead of on the guard under test.
    """
    rows = pd.DataFrame({
        "season": [2020, 2020, 2020],
        "team": ["KC", "KC", "KC"],
        "player_key": ["qb", "a", "b"],
        "player_name": ["QB", "A", "B"],
        "position": ["QB", "WR", "WR"],
        "offense_snaps": [1000.0, 900.0, 400.0],
        "games": [17, 17, 17],
        "observed_availability": [1.0, 1.0, 1.0],
        "targets": [0, 90, 40],
    })
    for name in columns:
        rows[name] = np.linspace(0.1, 0.4, len(rows))
    return rows


def test_asking_for_the_features_on_a_frame_without_them_raises():
    """A dropped feature fits as though the flag were off, silently.

    ``_fit_metadata`` keeps only the features present in the frame. Without
    this guard the arm and its baseline come back identical on every metric and
    the run looks like a clean null result.
    """
    pipeline = SeasonAverageVolumePipeline(room_structure_features=True)
    data = SeasonAverageData(
        pd.DataFrame({"season": [2020], "team": ["KC"], "offense_snaps": [1000.0]}),
        _frame(("prior_target_role_uncertainty",)),  # one of the two, not both
    )
    with pytest.raises(ValueError, match="prior_rec_room_quality_advantage"):
        pipeline.fit(data)


def test_enabling_reaches_the_target_room_and_nothing_else():
    pipeline = SeasonAverageVolumePipeline(room_structure_features=True)
    pipeline.target_model.extra_features = tuple(
        dict.fromkeys((*pipeline.target_model.extra_features, *TARGET_ROOM_FEATURES))
    )
    for name in TARGET_ROOM_FEATURES:
        assert name in pipeline.target_model.extra_features
        assert name not in pipeline.carry_model.extra_features
        assert name not in pipeline.snap_model.extra_features
