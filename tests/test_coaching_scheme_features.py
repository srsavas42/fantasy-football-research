"""The scheme carrier's carried backfield tendency, as offered to the softmax.

The screen behind this lives in scripts/screen_coaching_tree_transfer.py; these
guard the properties that make the feature usable at all rather than the size
of the effect.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ffmodel.features.coaching_scheme import (
    COACHING_SCHEME_FEATURES,
    add_coaching_scheme_features,
)
from ffmodel.models.volume_season_average import SeasonAverageVolumePipeline

FEATURE = COACHING_SCHEME_FEATURES[0]


def _rows() -> pd.DataFrame:
    return pd.DataFrame({
        "season": [2024] * 4,
        "team": ["KC"] * 4,
        "position": ["RB", "WR", "TE", "QB"],
        "targets": [40.0, 90.0, 50.0, 0.0],
        "is_replacement_player": [0, 0, 0, 0],
    })


def test_the_flag_is_off_until_the_gate_speaks():
    assert not SeasonAverageVolumePipeline().coaching_scheme_features


def test_it_ships_as_an_interaction_not_a_level():
    """A team-season constant cancels exactly in a within-team softmax.

    This is the same arithmetic that made prior_target_room_competition measure
    as nothing: every player on the team shares the value, so it cannot
    separate the players the allocator is choosing between. Multiplying by the
    back indicator is the whole point of the feature.
    """
    out = add_coaching_scheme_features(_rows())
    non_backs = out.loc[~out.position.eq("RB"), FEATURE]
    assert (non_backs == 0.0).all()


def test_missing_lineage_reads_as_no_information():
    """Zero, not a low value: the feature is centred, so 0 is the neutral point."""
    out = add_coaching_scheme_features(_rows())
    assert out[FEATURE].notna().all()
    assert np.isfinite(out[FEATURE].to_numpy(dtype=float)).all()


def test_it_refuses_rows_without_the_columns_it_needs():
    with pytest.raises(ValueError, match="position"):
        add_coaching_scheme_features(
            pd.DataFrame({"season": [2024], "team": ["KC"]})
        )


def test_asking_for_the_feature_on_a_frame_without_it_raises():
    """A silently dropped feature fits as though the flag were off.

    The same failure the teammate-quality flag hit: _fit_metadata keeps only
    the features present in the frame, so the arm and its baseline come back
    identical on every metric and the run looks like a clean null.
    """
    from ffmodel.features.season_average import SeasonAverageData

    pipeline = SeasonAverageVolumePipeline(coaching_scheme_features=True)
    players = pd.DataFrame({
        "season": [2020, 2020],
        "team": ["KC", "KC"],
        "player_key": ["qb", "rb"],
        "player_name": ["QB", "RB"],
        "position": ["QB", "RB"],
        "offense_snaps": [1000.0, 500.0],
        "games": [17, 17],
        "observed_availability": [1.0, 1.0],
        "targets": [0, 40],
    })
    data = SeasonAverageData(
        pd.DataFrame({"season": [2020], "team": ["KC"], "offense_snaps": [1000.0]}),
        players,
    )
    with pytest.raises(ValueError, match=FEATURE):
        pipeline.fit(data)
