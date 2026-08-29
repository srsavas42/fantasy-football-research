"""Expected points, and the two ways a week can have none.

The panel carries honest zeros: a rostered player who did not play is a real
zero-point week, not a missing one. The expected-points feed has no such
convention -- it simply omits the row. So the merge has to distinguish a week
that was worth nothing from a week the feed never priced, and those two cases
land in the same empty cell.
"""

import numpy as np
import pandas as pd
import pytest

from ffmodel.weekly import expected as expected_module
from ffmodel.weekly.expected import SOURCE_COLUMN, attach_expected, load_expected_points


def _panel() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "player_key": ["played", "benched", "unpriced"],
            "season": [2022, 2022, 2022],
            "week": [4, 4, 4],
            "position": ["RB", "WR", "TE"],
            "played": [1, 0, 1],
            "points": [18.4, 0.0, 6.1],
        }
    )


def _stub(monkeypatch, rows: pd.DataFrame) -> None:
    monkeypatch.setattr(expected_module.ingest, "load_ff_opportunity", lambda s: rows)


def _feed(**overrides) -> pd.DataFrame:
    base = pd.DataFrame(
        {
            "season": [2022],
            "week": [4],
            "player_id": ["played"],
            SOURCE_COLUMN: [12.0],
        }
    )
    for key, value in overrides.items():
        base[key] = value
    return base


def test_a_week_not_played_is_zero_expected_not_unknown(monkeypatch) -> None:
    _stub(monkeypatch, _feed())
    got = attach_expected(_panel()).set_index("player_key")

    # The feed has no row for a player who did not dress, and neither case can
    # be read off the merge alone -- the convention has to be applied.
    assert got.loc["benched", "points_exp"] == 0.0
    assert got.loc["benched", "points_luck"] == 0.0


def test_a_played_week_the_feed_never_priced_stays_missing(monkeypatch) -> None:
    """The other empty cell. Filling this one with zero would invent a signal."""
    _stub(monkeypatch, _feed())
    got = attach_expected(_panel()).set_index("player_key")

    assert np.isnan(got.loc["unpriced", "points_exp"])
    assert np.isnan(got.loc["unpriced", "points_luck"])


def test_luck_is_what_the_bounce_added(monkeypatch) -> None:
    _stub(monkeypatch, _feed())
    got = attach_expected(_panel()).set_index("player_key")

    assert got.loc["played", "points_exp"] == pytest.approx(12.0)
    assert got.loc["played", "points_luck"] == pytest.approx(18.4 - 12.0)


def test_two_games_in_a_week_add_rather_than_average(monkeypatch) -> None:
    """A mid-season move puts two feed rows on one panel row; the opportunities
    are both his."""
    doubled = pd.DataFrame(
        {
            "season": [2022, 2022],
            "week": [4, 4],
            "player_id": ["played", "played"],
            SOURCE_COLUMN: [7.0, 5.0],
        }
    )
    _stub(monkeypatch, doubled)
    got = load_expected_points([2022])

    assert len(got) == 1
    assert got.iloc[0]["points_exp"] == pytest.approx(12.0)


def test_a_missing_feed_leaves_no_signal_rather_than_a_fabricated_one(
    monkeypatch,
) -> None:
    _stub(monkeypatch, pd.DataFrame())
    got = attach_expected(_panel()).set_index("player_key")

    assert got.loc["played", "points_exp"] != got.loc["played", "points_exp"]  # NaN
    assert got.loc["benched", "points_exp"] == 0.0


def test_an_unreachable_feed_is_an_empty_frame_not_a_crash(monkeypatch) -> None:
    def _boom(seasons):
        raise RuntimeError("network")

    monkeypatch.setattr(expected_module.ingest, "load_ff_opportunity", _boom)
    assert load_expected_points([2022]).empty
    # And the panel still comes back whole.
    assert len(attach_expected(_panel())) == 3


def test_the_merge_does_not_duplicate_panel_rows(monkeypatch) -> None:
    """A one-to-many merge would silently inflate every downstream count."""
    _stub(
        monkeypatch,
        pd.DataFrame(
            {
                "season": [2022, 2022],
                "week": [4, 4],
                "player_id": ["played", "played"],
                SOURCE_COLUMN: [7.0, 5.0],
            }
        ),
    )
    assert len(attach_expected(_panel())) == 3


def _career(weeks: int = 6) -> pd.DataFrame:
    """One player, one season, with expected points attached."""
    rows = []
    for week in range(1, weeks + 1):
        played = int(week != 3)
        rows.append(
            {
                "player_key": "P0",
                "player_id": "P0",
                "season": 2022,
                "week": week,
                "team": "T",
                "opponent": f"O{week}",
                "position": "RB",
                "played": played,
                "points": 10.0 * week if played else 0.0,
                "points_exp": 1.0 * week if played else 0.0,
                "points_luck": 9.0 * week if played else 0.0,
                "targets": 0.0,
                "rush_att": 0.0,
                "pass_att": 0.0,
                "receptions": 0.0,
                "rush_yds": 0.0,
                "rec_yds": 0.0,
                "rush_epa": 0.0,
                "rec_epa": 0.0,
                "team_targets": 30.0,
                "team_rush_att": 25.0,
                "team_plays": 60.0,
                "team_points": 40.0,
                "team_pass_att": 35.0,
            }
        )
    return pd.DataFrame(rows)


def test_expected_history_is_lagged_and_skips_weeks_he_sat_out() -> None:
    from ffmodel.weekly.features import add_features

    frame = add_features(_career()).sort_values("week").set_index("week")

    # Week 1 has no history at all.
    assert np.isnan(frame.loc[1, "prior_points_exp_last"])
    # Week 2 sees week 1 and nothing of its own.
    assert frame.loc[2, "prior_points_exp_last"] == pytest.approx(1.0)
    # Week 4 follows the week he sat out, so it still sees week 2 -- a zero
    # there would be a benching read as an opportunity-free game.
    assert frame.loc[4, "prior_points_exp_last"] == pytest.approx(2.0)
    assert frame.loc[4, "prior_luck_last"] == pytest.approx(18.0)


def test_expected_history_does_not_see_its_own_week() -> None:
    from ffmodel.weekly.features import add_features

    panel = _career()
    base = add_features(panel).sort_values("week").reset_index(drop=True)
    perturbed = panel.copy()
    hit = perturbed["week"] == 4
    perturbed.loc[hit, ["points_exp", "points_luck"]] += 1000.0
    after = add_features(perturbed).sort_values("week").reset_index(drop=True)

    columns = [
        "prior_points_exp_recent",
        "prior_points_exp_last",
        "prior_luck_recent",
        "prior_luck_last",
    ]
    upto = base["week"] <= 4
    pd.testing.assert_frame_equal(base.loc[upto, columns], after.loc[upto, columns])
    # And it does reach the following week, or the column would be inert.
    assert not np.allclose(
        base.loc[base["week"] == 5, columns].to_numpy(),
        after.loc[after["week"] == 5, columns].to_numpy(),
    )
