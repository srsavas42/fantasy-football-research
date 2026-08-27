"""The variance split, and the correction that decides its answer.

A share measured over eleven carries is not a precise quantity. Its week-to-week
wobble is mostly the binomial noise of a small denominator, and counting that as
role drift is what makes a perfectly stable role look unstable. On this panel the
correction removes 45% of the raw week-to-week variance -- without it the
conclusion of the whole decomposition flips from "between-season and week-to-week
are comparable" to "week-to-week dominates".

So the tests are built around constructed players whose true role is known:
one whose share never changes, one who changes only between seasons, and one who
changes only inside a season.
"""

import numpy as np
import pandas as pd
import pytest

from scripts.decompose_role_change import (
    lag_comparison,
    share_columns,
    variance_components,
)


def _shares(rows) -> pd.DataFrame:
    return pd.DataFrame(
        rows, columns=["player_key", "season", "week", "position", "played",
                       "share", "opportunities", "model_lag"]
    )


def _player(key, season_shares, opportunities=30.0, weeks=10):
    """One row per week at exactly the stated share -- no sampling wobble."""
    out = []
    for season, share in season_shares.items():
        for week in range(1, weeks + 1):
            out.append([key, season, week, "WR", 1, share, opportunities, share])
    return out


def test_a_stable_role_shows_no_drift() -> None:
    rows = []
    for i, share in enumerate([0.10, 0.20, 0.30]):
        rows += _player(f"p{i}", {2020: share, 2021: share, 2022: share})
    got = variance_components(_shares(rows))
    # All the variance is between players; neither within-player term survives.
    assert got["pct_between_player"] > 99.0
    assert got["between_season_within_player"] == pytest.approx(0.0, abs=1e-9)
    assert got["within_season_drift"] == pytest.approx(0.0, abs=1e-9)


def test_a_role_that_only_moves_between_seasons_is_attributed_there() -> None:
    rows = []
    for i in range(4):
        base = 0.10 + 0.02 * i
        rows += _player(f"p{i}", {2020: base, 2021: base + 0.15, 2022: base + 0.30})
    got = variance_components(_shares(rows))
    assert got["between_season_within_player"] > 0.0
    assert got["within_season_drift"] == pytest.approx(0.0, abs=1e-9)
    assert got["pct_between_season"] > got["pct_within_season"]


def test_a_role_that_only_moves_inside_a_season_is_attributed_there() -> None:
    """Same season mean every year, but a mid-season step up each time."""
    rows = []
    for i in range(4):
        low, high = 0.10 + 0.02 * i, 0.40 + 0.02 * i
        for season in (2020, 2021, 2022):
            for week in range(1, 11):
                share = low if week <= 5 else high
                rows.append([f"p{i}", season, week, "WR", 1, share, 30.0, share])
    got = variance_components(_shares(rows))
    assert got["within_season_drift"] > 0.0
    assert got["between_season_within_player"] == pytest.approx(0.0, abs=1e-9)
    assert got["pct_within_season"] > got["pct_between_season"]


def test_sampling_noise_is_removed_from_the_week_to_week_term() -> None:
    """A fixed true role sampled over few opportunities must not read as drift.

    This is the correction the decomposition's conclusion depends on. Each
    player's underlying share is constant; the observed weekly shares wobble only
    because the denominator is small.
    """
    rng = np.random.default_rng(0)
    rows = []
    opportunities = 25.0
    for i, truth in enumerate([0.15, 0.25, 0.35, 0.45]):
        for season in (2020, 2021, 2022):
            for week in range(1, 13):
                drawn = rng.binomial(int(opportunities), truth) / opportunities
                rows.append([f"p{i}", season, week, "WR", 1, drawn, opportunities, truth])
    got = variance_components(_shares(rows))

    # The raw figure is inflated by binomial noise; the corrected one is not.
    assert got["within_season_sampling_noise"] > 0.0
    assert got["within_season_drift"] < 0.25 * got["within_season_raw"]
    # And the noise estimate should land near the binomial value it is modelling.
    expected = np.mean([p * (1 - p) / opportunities for p in (0.15, 0.25, 0.35, 0.45)])
    assert got["within_season_sampling_noise"] == pytest.approx(expected, rel=0.35)


def test_a_negative_remainder_is_floored_at_zero() -> None:
    """Noise can exceed the observed spread; drift is never reported negative."""
    rows = []
    for i, share in enumerate([0.2, 0.3, 0.4]):
        # Constant observed share with a tiny denominator: implied noise exceeds
        # the (zero) observed within-season variance.
        rows += _player(f"p{i}", {2020: share, 2021: share}, opportunities=3.0)
    got = variance_components(_shares(rows))
    assert got["within_season_drift"] == 0.0


def test_lag_comparison_excludes_the_current_week() -> None:
    """This season's average must not contain the week it is predicting."""
    rows = []
    for week in range(1, 8):
        # A step up at week 4; if the current week leaked in, the week-4 error
        # would be zero rather than large.
        share = 0.10 if week < 4 else 0.50
        rows.append(["p0", 2021, week, "WR", 1, share, 30.0, 0.10])
    table = lag_comparison(_shares(rows)).set_index("week")
    assert table.loc[4, "this_season_mae"] == pytest.approx(0.40, abs=1e-9)
