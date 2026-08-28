"""The fitted role process, and the two ways it could be silently wrong.

A share is bounded in [0, 1] and the dynamics are fitted in logit space, so the
round trip has to be exact or every simulated role is quietly mis-scaled. And the
process must revert toward the *player's own* standing level rather than the
population's -- reverting to the pooled mean would drag every starter down and
every backup up, producing a simulation in which roles converge, which is the
opposite of what they do.
"""

import numpy as np
import pandas as pd
import pytest

from ffmodel.weekly.usage import (
    EPSILON,
    ShareDynamics,
    UsageProcess,
    expit,
    logit,
)


def test_logit_round_trips_and_survives_the_boundary() -> None:
    values = np.array([0.0, 0.001, 0.25, 0.5, 0.9, 1.0])
    back = expit(logit(values))
    # Interior values return unchanged; the boundary is squeezed, not infinite.
    np.testing.assert_allclose(back[2:5], values[2:5], atol=1e-9)
    assert np.isfinite(logit(values)).all()
    assert back[0] == pytest.approx(EPSILON, abs=1e-6)
    assert back[-1] == pytest.approx(1.0 - EPSILON, abs=1e-6)


def test_the_process_reverts_toward_the_players_own_level() -> None:
    """A high-usage player must not be dragged toward the population mean."""
    dynamics = ShareDynamics(
        intercept=0.0, persistence=0.35, reversion=0.58, innovation=0.0
    )
    star = logit(np.array([0.30]))
    backup = logit(np.array([0.05]))
    noise = np.zeros(1)
    # Each held at his own level: the step should leave him near where he is.
    assert expit(dynamics.step(star, star, noise))[0] > 0.20
    assert expit(dynamics.step(backup, backup, noise))[0] < 0.12


def test_a_displaced_share_is_pulled_back_toward_the_level() -> None:
    dynamics = ShareDynamics(
        intercept=0.0, persistence=0.35, reversion=0.58, innovation=0.0
    )
    level = logit(np.array([0.25]))
    high = logit(np.array([0.45]))
    low = logit(np.array([0.08]))
    assert expit(dynamics.step(high, level, np.zeros(1)))[0] < 0.45
    assert expit(dynamics.step(low, level, np.zeros(1)))[0] > 0.08


def _panel(seasons=(2019, 2020, 2021), weeks=15, players=40) -> pd.DataFrame:
    rng = np.random.default_rng(2)
    rows = []
    for season in seasons:
        for player in range(players):
            role = rng.uniform(0.05, 0.35)
            for week in range(1, weeks + 1):
                share = float(np.clip(role + rng.normal(0, 0.04), 0.01, 0.9))
                rows.append(
                    {
                        "player_key": f"P{player}",
                        "season": season,
                        "week": week,
                        "position": ["QB", "RB", "WR", "TE"][player % 4],
                        "played": 1,
                        "targets": share * 32.0,
                        "team_targets": 32.0,
                        "rush_att": share * 26.0,
                        "team_rush_att": 26.0,
                        "snap_share": float(np.clip(share * 2.2, 0.02, 0.98)),
                    }
                )
    return pd.DataFrame(rows)


def test_a_persistent_role_fits_as_persistent() -> None:
    """Shares generated around a stable level must not fit as a random walk."""
    process = UsageProcess().fit(_panel())
    for name in ("primary_share", "snap_share"):
        dynamics = process.pooled[name]
        # Persistence and reversion together describe where next week sits
        # between last week and the standing level; neither should run away.
        assert 0.0 <= dynamics.persistence < 1.2
        assert dynamics.innovation > 0.0
        assert np.isfinite(dynamics.reversion)


def test_an_unfitted_position_falls_back_to_the_pooled_process() -> None:
    process = UsageProcess().fit(_panel())
    fallback = process.get("primary_share", "ZZ")
    assert fallback is process.pooled["primary_share"]


def test_shares_are_masked_to_weeks_he_played() -> None:
    """A week on the sideline is not evidence about his role."""
    panel = _panel(seasons=(2020,), weeks=6, players=8)
    panel.loc[panel["week"] == 3, ["played", "targets", "rush_att"]] = 0
    observed = UsageProcess.observed_shares(panel)
    assert (observed["week"] != 3).all()


def test_a_random_walk_is_detected_and_a_stationary_process_is_not() -> None:
    """The variance-ratio test, on two constructed series with known answers.

    This is the measurement that decided the role question, so it needs to be
    demonstrably able to tell the two cases apart rather than merely to run. A
    mean-reverting share and a wandering one look identical one week apart; the
    horizon is what separates them, and a test that only checked one horizon
    would pass on either.
    """
    from ffmodel.weekly.usage import estimate_random_walk

    def panel(walk_sd: float, seed: int) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        rows = []
        for player in range(90):
            level = rng.uniform(-2.0, -0.5)
            for season in (2020, 2021):
                state = level
                for week in range(1, 17):
                    # Mean-reverting around a level that may itself wander.
                    level = level + rng.normal(0.0, walk_sd)
                    state = level + 0.5 * (state - level) + rng.normal(0.0, 0.35)
                    share = float(1.0 / (1.0 + np.exp(-state)))
                    rows.append(
                        {
                            "player_key": f"P{player}",
                            "season": season,
                            "week": week,
                            "position": "WR",
                            "played": 1,
                            "targets": share * 30.0,
                            "team_targets": 30.0,
                            "rush_att": 0.0,
                            "team_rush_att": 25.0,
                            "snap_share": 0.5,
                        }
                    )
        return pd.DataFrame(rows)

    stationary, _ = estimate_random_walk(panel(0.0, 1), "primary_share")
    wandering, table = estimate_random_walk(panel(0.30, 2), "primary_share")

    assert wandering > stationary
    assert wandering > 0.10, "a real random walk was not detected"
    assert stationary < 0.10, "a stationary process was reported as wandering"
    # The wandering series must still be climbing at the long horizons, which is
    # the shape the estimator reads.
    tail = table[table["horizon"] >= 6].sort_values("horizon")
    assert tail["variance"].iloc[-1] > tail["variance"].iloc[0]


def test_the_estimator_declines_without_enough_pairs() -> None:
    from ffmodel.weekly.usage import estimate_random_walk

    tiny = pd.DataFrame(
        {
            "player_key": ["A"] * 6,
            "season": [2020] * 6,
            "week": range(1, 7),
            "position": ["WR"] * 6,
            "played": [1] * 6,
            "targets": [5.0] * 6,
            "team_targets": [30.0] * 6,
            "rush_att": [0.0] * 6,
            "team_rush_att": [25.0] * 6,
            "snap_share": [0.5] * 6,
        }
    )
    value, table = estimate_random_walk(tiny, "primary_share")
    assert value == 0.0
    assert len(table) < 3
