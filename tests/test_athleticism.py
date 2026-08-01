"""One athletic score per player: combine composite, overridden by RAS.

The composite is a fallback and a slot, not a validated predictor — see the
module docstring. These cover the mechanics: orientation, partial testing,
leakage in the percentile baseline, and precedence.
"""

import numpy as np
import pandas as pd

from ffmodel.features.athleticism import (
    ATHLETIC_FEATURES,
    combine_athletic_score,
    load_ras_scores,
    merge_athletic_score,
)


def _features() -> pd.DataFrame:
    # Two classes of receivers; the 2024 pair straddle the 2023 pair.
    return pd.DataFrame(
        {
            "season": [2023, 2023, 2024, 2024],
            "position": ["WR", "WR", "WR", "WR"],
            "player_id": ["00-0000001", "00-0000002", "00-0000003", "00-0000004"],
            "combine_forty": [4.30, 4.70, 4.25, 4.75],
            "combine_vertical": [40.0, 30.0, 41.0, 29.0],
            "combine_broad_jump": [130.0, 110.0, 132.0, 108.0],
            "combine_bench": [np.nan] * 4,
            "combine_cone": [np.nan] * 4,
            "combine_shuttle": [np.nan] * 4,
            "combine_ht": [74.0, 72.0, 75.0, 71.0],
            "combine_wt": [210.0, 190.0, 215.0, 185.0],
        }
    )


def test_timed_drills_are_oriented_so_faster_scores_higher():
    scored = combine_athletic_score(_features()).set_index("player_id")

    # A 4.30 forty is better than a 4.70, even though the number is smaller.
    assert scored.loc["00-0000001", "athletic_score"] > scored.loc["00-0000002", "athletic_score"]


def test_score_counts_only_the_drills_actually_recorded():
    scored = combine_athletic_score(_features()).set_index("player_id")

    # Five of eight metrics present, so a skipped drill is not a zero.
    assert scored.loc["00-0000001", "athletic_metrics_used"] == 5


def test_percentiles_do_not_use_later_classes():
    scored = combine_athletic_score(_features()).set_index("player_id")

    # The fastest 2023 receiver is top of the pool known in 2023, so a quicker
    # player arriving in 2024 must not retroactively demote him.
    assert scored.loc["00-0000001", "athletic_score"] == 10.0


def test_supplied_ras_takes_precedence_over_the_composite():
    rows = pd.DataFrame({"player_key": ["00-0000001", "00-0000002"], "position": ["WR", "WR"]})
    ras = pd.DataFrame({"player_id": ["00-0000001"], "ras": [9.4]})

    merged = merge_athletic_score(rows, _features(), ras=ras)

    assert merged.loc[0, "athletic_score"] == 9.4
    assert merged.loc[0, "athletic_score_is_ras"] == 1.0
    # The player RAS does not cover keeps the composite.
    assert merged.loc[1, "athletic_score_is_ras"] == 0.0
    assert np.isfinite(merged.loc[1, "athletic_score"])


def test_missing_ras_file_is_not_an_error(tmp_path):
    # Absence is the normal state: RAS is not fetched, it is dropped in by hand.
    assert load_ras_scores(tmp_path / "nope.csv").empty


def test_ras_file_without_a_score_column_is_rejected(tmp_path):
    path = tmp_path / "ras_scores.csv"
    path.write_text("player_name,score\nSomeone,9.1\n")

    try:
        load_ras_scores(path)
    except ValueError as error:
        assert "ras" in str(error)
    else:
        raise AssertionError("a table with no ras column must not load silently")


def test_players_with_no_testing_get_no_score():
    rows = pd.DataFrame({"player_key": ["00-0009999"], "position": ["WR"]})

    merged = merge_athletic_score(rows, _features(), ras=pd.DataFrame())

    assert set(ATHLETIC_FEATURES) <= set(merged.columns)
    # Never measured is unknown, not unathletic.
    assert pd.isna(merged.loc[0, "athletic_score"])
    assert merged.loc[0, "athletic_metrics_used"] == 0.0
