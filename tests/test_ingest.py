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
        }
    )
    out = _map_weekly_aliases(raw)
    assert not out.columns.duplicated().any()
    assert out["player_name"].tolist() == ["Aaron Rodgers", "Fallback Name"]
    assert out["pass_att"].tolist() == [30, 0]
