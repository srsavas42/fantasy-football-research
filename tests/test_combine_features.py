"""Combine measurables, and the two different shapes of their absence.

A player outside the combine feed was not invited, which is a scouting verdict.
A player inside it with nothing recorded was invited and did not test. Neither
is a slow time, so no measurement is imputed and both facts are carried as
their own features.
"""

import numpy as np
import pandas as pd

from ffmodel.features.combine import (
    COMBINE_FEATURES,
    combine_feature_rows,
    merge_combine_features,
    parse_height_inches,
)


def _measurables() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "season": [2026, 2026],
            "player_name": ["Tested Player", "Invited Only"],
            "position": ["WR", "QB"],
            "player_id": ["00-0000001", "00-0000002"],
            "ht": ["6-2", "6-5"],
            "wt": [200.0, 225.0],
            "forty": [4.45, np.nan],
            "vertical": [38.0, np.nan],
            "broad_jump": [125.0, np.nan],
            "bench": [np.nan, np.nan],
            "cone": [np.nan, np.nan],
            "shuttle": [np.nan, np.nan],
        }
    )


def test_height_is_parsed_from_feet_and_inches():
    parsed = parse_height_inches(pd.Series(["6-2", "5-11", "6-0", None, "74"]))

    assert parsed.tolist()[:3] == [74.0, 71.0, 72.0]
    assert pd.isna(parsed.iloc[3])
    # Already-numeric values pass through rather than being discarded.
    assert parsed.iloc[4] == 74.0


def test_completed_drills_are_counted_not_imputed():
    rows = combine_feature_rows(_measurables()).set_index("player_name")

    assert rows.loc["Tested Player", "combine_drills_completed"] == 3
    assert rows.loc["Tested Player", "combine_tested"] == 1
    # Invited, tested nothing: a real and distinct state.
    assert rows.loc["Invited Only", "combine_drills_completed"] == 0
    assert rows.loc["Invited Only", "combine_tested"] == 0
    assert rows.loc["Invited Only", "combine_invited"] == 1
    # No time was recorded, so none is invented.
    assert pd.isna(rows.loc["Invited Only", "combine_forty"])


def test_players_the_feed_never_listed_are_marked_uninvited():
    rows = pd.DataFrame(
        {
            "player_key": ["00-0000001", "00-0000002", "00-0009999"],
            "position": ["WR", "QB", "RB"],
        }
    )

    merged = merge_combine_features(rows, combine_feature_rows(_measurables()))

    # Not invited is a known fact, so it is zero rather than missing...
    assert merged["combine_invited"].tolist() == [1.0, 1.0, 0.0]
    assert merged["combine_tested"].tolist() == [1.0, 0.0, 0.0]
    # ...while the measurements stay missing, because none were taken.
    assert pd.isna(merged.loc[2, "combine_forty"])


def test_measurables_survive_the_merge():
    rows = pd.DataFrame({"player_key": ["00-0000001"], "position": ["WR"]})

    merged = merge_combine_features(rows, combine_feature_rows(_measurables()))

    assert merged.loc[0, "combine_forty"] == 4.45
    assert merged.loc[0, "combine_ht"] == 74.0


def test_an_empty_feed_still_yields_the_full_feature_contract():
    rows = pd.DataFrame({"player_key": ["00-0000001"], "position": ["WR"]})

    merged = merge_combine_features(rows, pd.DataFrame())

    assert set(COMBINE_FEATURES) <= set(merged.columns)
    # Unknown invitation status must not read as a confirmed non-invite... but
    # with no feed at all there is nothing to contradict, so it stays zero.
    assert merged.loc[0, "combine_invited"] == 0.0
    assert pd.isna(merged.loc[0, "combine_forty"])
