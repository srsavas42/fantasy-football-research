"""Pass-play participation proxy construction."""

import numpy as np
import pandas as pd

from ffmodel.features.participation import (
    attach_lagged_participation_features,
    season_pass_play_participation,
)


def test_participation_counts_only_targeted_pass_plays_and_lags():
    raw = pd.DataFrame(
        {
            "nflverse_game_id": ["2023_01_A_B", "2023_01_A_B", "2023_01_A_B"],
            "offense_players": ["qb;wr;te;c", "qb;wr;rb;c", "qb;wr;rb;c"],
            "offense_positions": ["QB;WR;TE;C", "QB;WR;RB;C", "QB;WR;RB;C"],
            "route": ["GO", "SCREEN", ""],
        }
    )
    usage = season_pass_play_participation(raw).set_index("player_key")
    assert usage.loc["wr", "pass_play_opportunities"] == 2
    assert usage.loc["te", "pass_play_opportunities"] == 1
    assert "c" not in usage.index

    rows = pd.DataFrame(
        {
            "season": [2024, 2024],
            "player_key": ["wr", "te"],
            "prior_targets": [1, 1],
        }
    )
    attached = attach_lagged_participation_features(rows, raw).set_index("player_key")
    assert np.isclose(attached.loc["wr", "prior_targets_per_pass_play"], 0.5)
    assert np.isclose(attached.loc["te", "prior_targets_per_pass_play"], 1.0)
    assert attached["prior_participation_available"].eq(1).all()
