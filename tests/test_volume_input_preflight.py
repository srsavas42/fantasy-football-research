"""The volume fit checks its inputs before it spends a posterior on them.

The pipeline fits eight components in sequence and the quarterback layers come
fourth, so a source that cannot support them used to burn three fits before
failing — and then failed from inside whichever model noticed first. For the
committed CSVs that surfaced as a PyTensor ``MemoryError`` on an empty softmax,
which names neither the column nor the source that caused it.
"""

import pandas as pd
import pytest

from ffmodel.features.season_average import SeasonAverageData
from ffmodel.models.volume_season_average import (
    SeasonAverageVolumePipeline,
    volume_input_problems,
)


def _player_rows(qb_snaps: float) -> pd.DataFrame:
    records = []
    for position, snaps in (("QB", qb_snaps), ("RB", 500.0), ("WR", 700.0)):
        records.append(
            {
                "season": 2020,
                "team": "A",
                "player_key": f"A-{position}",
                "player_name": position,
                "position": position,
                "offense_snaps": snaps,
                "snap_counts_observed": 1,
                "team_games": 16,
                "games": 16,
                "pass_att": 500.0 if position == "QB" else 0.0,
                "targets": 0.0 if position == "QB" else 90.0,
                "rush_att": 40.0 if position == "QB" else 120.0,
            }
        )
    return pd.DataFrame(records)


def _team_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [{"season": 2020, "team": "A", "games": 16, "opportunity_plays": 1000}]
    )


def test_quarterbacks_without_snaps_are_reported_as_a_source_limitation():
    data = SeasonAverageData(_team_rows(), _player_rows(qb_snaps=0.0))

    problems = volume_input_problems(data)

    assert len(problems) == 1
    # The message has to name the response and the source, not the symptom.
    assert "offense_snaps" in problems[0]
    assert "legacy" in problems[0]


def test_a_source_with_quarterback_snaps_passes_preflight():
    data = SeasonAverageData(_team_rows(), _player_rows(qb_snaps=950.0))

    assert volume_input_problems(data) == []


def test_missing_rosters_are_reported_without_indexing_into_them():
    empty = pd.DataFrame(columns=["season", "team", "player_key", "position"])

    problems = volume_input_problems(SeasonAverageData(_team_rows(), empty))

    assert any("player_rows is empty" in problem for problem in problems)


def test_fit_refuses_before_sampling_anything():
    data = SeasonAverageData(_team_rows(), _player_rows(qb_snaps=0.0))
    pipeline = SeasonAverageVolumePipeline()

    with pytest.raises(ValueError, match="not fittable"):
        pipeline.fit(data)

    # Nothing was sampled: the team model is the first fit and it never ran.
    assert pipeline.team_model.idata is None
    assert pipeline.fit_seconds == {}
