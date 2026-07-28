"""nflverse-to-canonical mapping edge cases."""

import pandas as pd

from ffmodel.data.ingest import _map_weekly_aliases


def test_weekly_aliases_coalesce_without_duplicate_columns():
    raw = pd.DataFrame(
        {
            "player_name": ["A.Rodgers", "Fallback Name"],
            "player_display_name": ["Aaron Rodgers", None],
            "team": ["NYJ", "BUF"],
            "attempts": [30, 0],
            "sacks_suffered": [2, 0],
            "passing_interceptions": [1, 0],
            "receiving_air_yards": [0, 42],
            "receiving_epa": [None, 3.5],
        }
    )
    out = _map_weekly_aliases(raw)
    assert not out.columns.duplicated().any()
    assert out["player_name"].tolist() == ["Aaron Rodgers", "Fallback Name"]
    assert out["pass_att"].tolist() == [30, 0]
    assert out["pass_sacks"].tolist() == [2, 0]
    assert out["pass_int"].tolist() == [1, 0]
    assert out["rec_air_yds"].tolist() == [0, 42]
    assert out["rec_epa"].isna().tolist() == [True, False]
