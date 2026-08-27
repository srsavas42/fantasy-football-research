"""What the weekly estimators must produce, regardless of how well they score.

Three properties the metrics cannot check for themselves: draws have the shape
the scorer expects, the hurdle's zero is a real atom rather than a small number,
and the rest-of-season target is the sum it claims to be.

The last test here is the one with an argument behind it. Summing independent
weeks understates a season total's spread, because most of what is unknown about
a player is unknown in all of his weeks at once. The hierarchy exists to fix
that, so a test pins the direction: same means, wider intervals.
"""

import numpy as np
import pandas as pd
import pytest

from ffmodel.weekly.fitting import Logistic, Ridge
from ffmodel.weekly.nextweek import Hurdle, HistoryMean, PositionClimatology
from ffmodel.weekly.restofseason import (
    OFFSET,
    TARGET,
    HierarchicalSeason,
    add_rest_of_season_target,
)
from ffmodel.weekly.features import add_features


def _training_panel(seasons=(2019, 2020), weeks=16, players=40) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    rows = []
    for season in seasons:
        for player in range(players):
            skill = rng.uniform(2.0, 20.0)
            durability = rng.uniform(0.55, 0.98)
            for week in range(1, weeks + 1):
                played = int(rng.random() < durability)
                rows.append(
                    {
                        "player_key": f"P{player}",
                        "player_id": f"P{player}",
                        "season": season,
                        "week": week,
                        "team": f"T{player % 8}",
                        "opponent": f"T{(player + 3) % 8}",
                        "position": ["QB", "RB", "WR", "TE"][player % 4],
                        "played": played,
                        "points": float(max(rng.normal(skill, 6.0), -3.0)) if played else 0.0,
                        "targets": float(rng.integers(0, 12)) if played else 0.0,
                        "rush_att": float(rng.integers(0, 18)) if played else 0.0,
                        "pass_att": 0.0,
                        "receptions": 0.0,
                        "rush_yds": float(rng.integers(0, 100)) if played else 0.0,
                        "rec_yds": float(rng.integers(0, 100)) if played else 0.0,
                        "rush_epa": float(rng.normal()) if played else 0.0,
                        "rec_epa": float(rng.normal()) if played else 0.0,
                        "team_targets": 32.0,
                        "team_rush_att": 26.0,
                        "team_plays": 62.0,
                        "team_points": 45.0,
                        "team_pass_att": 36.0,
                    }
                )
    return add_features(pd.DataFrame(rows))


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    return _training_panel()


@pytest.mark.parametrize(
    "estimator",
    [
        PositionClimatology(),
        HistoryMean(column="prior_points_recent", name="recency"),
        Hurdle(),
    ],
)
def test_draws_have_the_shape_the_scorer_expects(estimator, frame) -> None:
    target = frame["points"].to_numpy(float)
    fitted = estimator.fit(frame, target)
    samples = fitted.predict_samples(frame.head(50), draws=64, seed=1)
    assert samples.shape == (50, 64)
    assert np.isfinite(samples).all()


def test_the_hurdle_zero_is_an_atom(frame) -> None:
    """Weeks the draw says he sat out must be exactly zero, not nearly zero."""
    fitted = Hurdle().fit(frame, frame["points"].to_numpy(float))
    samples = fitted.predict_samples(frame.head(200), draws=256, seed=2)
    zeros = samples == 0.0
    assert zeros.any(), "no draw put any player on the bench"
    # Nothing should land just next to the atom by accident.
    tiny = (np.abs(samples) < 1e-9) & ~zeros
    assert not tiny.any()


def test_the_hurdle_is_reproducible(frame) -> None:
    fitted = Hurdle().fit(frame, frame["points"].to_numpy(float))
    first = fitted.predict_samples(frame.head(30), draws=32, seed=5)
    second = fitted.predict_samples(frame.head(30), draws=32, seed=5)
    np.testing.assert_array_equal(first, second)


def test_rest_of_season_target_is_the_remaining_sum(frame) -> None:
    seasonal = add_rest_of_season_target(frame)
    for (_, _), block in seasonal.groupby(["player_key", "season"]):
        block = block.sort_values("week")
        points = block["points"].to_numpy(float)
        got = block[TARGET].to_numpy(float)
        expected = points[::-1].cumsum()[::-1]
        np.testing.assert_allclose(got, expected)


def test_games_remaining_counts_down_to_one(frame) -> None:
    seasonal = add_rest_of_season_target(frame)
    for (_, _), block in seasonal.groupby(["season", "team"]):
        block = block.drop_duplicates("week").sort_values("week")
        got = block[OFFSET].to_numpy(int)
        np.testing.assert_array_equal(got, np.arange(len(got), 0, -1))


def test_the_hierarchy_widens_intervals_without_moving_the_mean(frame) -> None:
    """The argument for drawing the player once, made as a test.

    Independent weeks and correlated weeks agree about the expected total and
    disagree about how sure of it to be. If the hierarchy ever stops widening
    the interval, its latent draws have collapsed and it has silently become the
    thing it was built to replace.
    """
    seasonal = add_rest_of_season_target(frame)
    rows = seasonal[seasonal["week"] == 4].head(120)
    weekly_target = seasonal["points"].to_numpy(float)

    independent = HierarchicalSeason(persistent=False).fit(seasonal, weekly_target)
    correlated = HierarchicalSeason(persistent=True).fit(seasonal, weekly_target)
    assert correlated.level_sd > 0.0
    assert correlated.concentration < 400.0

    a = independent.predict_samples(rows, draws=400, seed=3)
    b = correlated.predict_samples(rows, draws=400, seed=3)

    spread_a = a.std(axis=1).mean()
    spread_b = b.std(axis=1).mean()
    assert spread_b > spread_a * 1.05

    # Means stay comparable: this is a statement about uncertainty, not level.
    assert abs(a.mean() - b.mean()) < 0.15 * max(a.mean(), 1.0)


def test_ridge_recovers_a_known_line() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(400, 2))
    y = 3.0 + 2.0 * x[:, 0] - 1.5 * x[:, 1] + rng.normal(0, 0.01, size=400)
    fitted = Ridge.fit(x, y, penalty=1e-6)
    np.testing.assert_allclose(fitted.predict(x), y, atol=0.05)


def test_logistic_separates_a_clean_boundary() -> None:
    rng = np.random.default_rng(1)
    x = rng.normal(size=(500, 1))
    y = (x[:, 0] > 0).astype(float)
    fitted = Logistic.fit(x, y, penalty=1e-3)
    probability = fitted.predict_proba(x)
    assert probability[x[:, 0] > 1].mean() > 0.9
    assert probability[x[:, 0] < -1].mean() < 0.1
