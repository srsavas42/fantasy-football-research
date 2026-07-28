"""Player ID and snap-share integration."""

import pandas as pd

from ffmodel.data.identity import canonicalize_player_dim
from ffmodel.features.snaps import canonicalize_snaps, merge_snap_usage


def _players():
    return canonicalize_player_dim(
        pd.DataFrame(
            {
                "gsis_id": ["00-1", "00-2"],
                "pfr_id": ["PlayEx00", "BackEx00"],
                "display_name": ["Example Player", "Backup Player"],
            }
        )
    )


def test_canonicalize_snap_percent_and_pfr_id():
    snaps = pd.DataFrame(
        {
            "pfr_player_id": ["PlayEx00", "BackEx00"],
            "season": [2024, 2024],
            "week": [1, 1],
            "team": ["BUF", "BUF"],
            "position": ["WR", "WR"],
            "offense_snaps": [50, 25],
            "offense_pct": ["80%", 0.4],
        }
    )
    out = canonicalize_snaps(snaps, _players())
    assert out["player_id"].tolist() == ["00-1", "00-2"]
    assert out["snap_share"].tolist() == [0.8, 0.4]


def test_snap_trailing_excludes_current_week():
    pw = pd.DataFrame(
        {
            "player_id": ["00-1", "00-1"],
            "player_name": ["Example Player", "Example Player"],
            "position": ["WR", "WR"],
            "team": ["BUF", "BUF"],
            "season": [2024, 2024],
            "week": [1, 2],
        }
    )
    snaps = pd.DataFrame(
        {
            "pfr_player_id": ["PlayEx00", "PlayEx00"],
            "season": [2024, 2024],
            "week": [1, 2],
            "team": ["BUF", "BUF"],
            "position": ["WR", "WR"],
            "offense_snaps": [40, 60],
            "offense_pct": [0.5, 0.9],
        }
    )
    out = merge_snap_usage(pw, snaps, _players())
    assert pd.isna(out.loc[0, "ewma_snap_share"])
    assert out.loc[1, "ewma_snap_share"] == 0.5
