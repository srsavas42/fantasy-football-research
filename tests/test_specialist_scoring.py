"""Kicker and defense scoring, and the two places it is easy to get wrong.

The distance tiers are the reason a kicker projection is worth building, so a
tier that silently collapses would remove the signal while leaving every metric
looking plausible. And the points-allowed function is a step function whose
boundaries are inclusive on the upper bound -- an off-by-one there moves a whole
tier of games and is invisible in an average.
"""

import pandas as pd

from ffmodel.config import DefenseRules, KickerRules
from ffmodel.simulation.scoring import (
    defense_points,
    kicker_points,
    points_allowed_score,
)


def test_field_goals_are_scored_by_distance_not_by_count():
    """Three makes are worth 9, 12 or 15 depending only on where from."""
    frame = pd.DataFrame(
        {
            "fg_made_30_39": [3.0, 0.0, 0.0],
            "fg_made_40_49": [0.0, 3.0, 0.0],
            "fg_made_50_59": [0.0, 0.0, 3.0],
        }
    )
    assert kicker_points(frame).tolist() == [9.0, 12.0, 15.0]


def test_the_short_buckets_collapse_into_one_tier():
    """0-19, 20-29 and 30-39 all score three, and all three are read."""
    frame = pd.DataFrame(
        {
            "fg_made_0_19": [1.0, 0.0, 0.0],
            "fg_made_20_29": [0.0, 1.0, 0.0],
            "fg_made_30_39": [0.0, 0.0, 1.0],
        }
    )
    assert kicker_points(frame).tolist() == [3.0, 3.0, 3.0]


def test_a_blocked_kick_is_a_miss():
    """The points did not go up, so it is scored as a miss rather than ignored."""
    blocked = pd.DataFrame({"fg_blocked": [1.0]})
    missed = pd.DataFrame({"fg_missed": [1.0]})
    assert kicker_points(blocked).tolist() == kicker_points(missed).tolist() == [-1.0]


def test_the_60_yard_bucket_is_not_dropped():
    """``fg_made_60_`` has a trailing underscore and is easy to miss by name."""
    frame = pd.DataFrame({"fg_made_60_": [1.0]})
    assert kicker_points(frame).tolist() == [5.0]


def test_kicker_rules_are_configurable():
    """A league with flat scoring changes the rules, not the model."""
    flat = KickerRules(fg_0_39=3.0, fg_40_49=3.0, fg_50_plus=3.0, fg_miss=0.0)
    frame = pd.DataFrame({"fg_made_50_59": [2.0], "fg_missed": [1.0]})
    assert kicker_points(frame, flat).tolist() == [6.0]


def test_points_allowed_tiers_are_inclusive_on_the_upper_bound():
    """Each boundary belongs to the tier it bounds, not the one above it."""
    allowed = pd.Series([0, 1, 6, 7, 13, 14, 20, 21, 27, 28, 34, 35, 60])
    expected = [10.0, 7.0, 7.0, 4.0, 4.0, 1.0, 1.0, 0.0, 0.0, -1.0, -1.0, -4.0, -4.0]
    assert points_allowed_score(allowed).tolist() == expected


def test_an_unknown_final_score_is_not_a_shutout():
    """A missing opponent score has to stay missing; zero would score it a 10."""
    assert points_allowed_score(pd.Series([None])).isna().all()


def test_defense_points_add_events_to_the_points_allowed_tier():
    frame = pd.DataFrame(
        {
            "def_sacks": [4.0],
            "def_interceptions": [2.0],
            "fumble_recovery_opp": [1.0],
            "def_tds": [1.0],
            "def_safeties": [0.0],
            "points_allowed": [10.0],
        }
    )
    # 4 sacks + 2x2 interceptions + 2 fumble + 6 touchdown = 16, plus the 7-13
    # tier's 4.
    assert defense_points(frame).tolist() == [20.0]


def test_return_touchdowns_count_alongside_defensive_ones():
    """nflverse splits them across two columns; both are worth six."""
    defensive = pd.DataFrame({"def_tds": [1.0], "points_allowed": [24.0]})
    special = pd.DataFrame({"special_teams_tds": [1.0], "points_allowed": [24.0]})
    assert defense_points(defensive).tolist() == defense_points(special).tolist() == [6.0]


def test_defense_rules_are_configurable():
    rules = DefenseRules(sack=2.0, points_allowed_tiers=((13, 5.0),), points_allowed_worst=0.0)
    frame = pd.DataFrame({"def_sacks": [3.0], "points_allowed": [10.0]})
    assert defense_points(frame, rules).tolist() == [11.0]
