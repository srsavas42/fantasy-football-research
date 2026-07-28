"""Leakage and ordering contracts for promoted season pathway features."""

import numpy as np
import pandas as pd

from ffmodel.features.season_pathways import add_player_pathway_features


def test_pathway_history_uses_lagged_player_state_across_team_changes():
    rows = pd.DataFrame(
        {
            "season": [2024, 2022, 2023],
            "team": ["C", "A", "B"],
            "player_key": ["p", "p", "p"],
            "position": ["RB", "RB", "RB"],
            "prior_snap_share": [0.50, 0.20, 0.35],
            "prior_availability": [0.90, 0.70, 0.80],
            "prior_rush_epa_per_carry": [0.10, -0.05, 0.02],
        }
    )

    result = add_player_pathway_features(rows).set_index("season")

    assert np.isclose(result.loc[2023, "prior_snap_share_trend"], 0.15)
    assert np.isclose(result.loc[2024, "prior_snap_share_trend"], 0.15)
    assert np.isclose(
        result.loc[2024, "prior_snap_share_3yr"],
        pd.Series([0.20, 0.35, 0.50]).ewm(alpha=0.5, adjust=True).mean().iloc[-1],
    )
    assert result.loc[2024, "team"] == "C"
