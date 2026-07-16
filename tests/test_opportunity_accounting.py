"""Team opportunity accounting and provider-position normalization."""

import pandas as pd

from ffmodel.features.volume import (
    opportunity_accounting_summary,
    opportunity_position,
    team_game_totals,
)


def _player_weeks():
    return pd.DataFrame(
        {
            "season": [2024] * 6,
            "week": [1] * 6,
            "team": ["A", "A", "A", "A", "B", "B"],
            "position": ["QB", "RB", "FB", "WR/RS", "QB", "WR"],
            "pass_att": [10, 0, 0, 0, 2, 0],
            "rush_att": [1, 4, 1, 0, 0, 1],
            "targets": [0, 1, 2, 5, 0, 3],
        }
    )


def test_provider_position_labels_are_mapped_to_volume_support():
    actual = opportunity_position(pd.Series(["HB", "FB", "WR/RS", "QB", "DEF"]))
    assert actual.tolist() == ["RB", "RB", "WR", "QB", "OTHER"]


def test_team_target_total_matches_share_support_and_flags_impossible_rows():
    totals = team_game_totals(_player_weeks()).set_index("team")
    assert totals.loc["A", "team_targets"] == 8
    assert totals.loc["A", "team_unallocated_targets"] == 0
    assert bool(totals.loc["A", "team_opportunity_valid"])
    assert not bool(totals.loc["B", "team_opportunity_valid"])
    assert totals.loc["B", "team_targets"] > totals.loc["B", "team_pass_att"]


def test_accounting_summary_reports_invalid_team_weeks():
    summary = opportunity_accounting_summary(_player_weeks())
    assert summary["team_weeks"] == 2
    assert summary["invalid_team_weeks"] == 1
    assert summary["invalid_team_week_rate"] == 0.5
