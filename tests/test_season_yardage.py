"""Yards follow the exposure their rate was fitted against.

``pass_yards_per_attempt`` means yards per *attempt*, so season yards are
attempts times that rate. The simulator instead rescaled the rate by a clipped
completion probability and multiplied by the completion count. That preserved
the mean, but it added a second noise source the model never estimated — worth
about 120 yards of standard deviation per efficiency draw — and it did so only
for passing and receiving, while rushing used the matched form. The rate draw
already carries per-opportunity noise through its
``sqrt(season_sigma^2 + opportunity_sigma^2 / exposure)`` scale, so that second
source was double-counting.
"""

import numpy as np
import pytest

from ffmodel.simulation.season_scoring import _event_yards


def test_yards_are_the_rate_times_its_own_exposure():
    exposure = np.array([[500.0]])
    rate = np.array([[7.2]])
    events = np.array([[330]])

    assert _event_yards(exposure, rate, events)[0, 0] == pytest.approx(3600)


def test_the_event_count_gates_but_does_not_scale():
    # Two draws with the same attempts and rate but very different completion
    # counts must produce the same yards: the rate is per attempt, and scaling
    # by completions as well would charge the completion randomness twice.
    exposure = np.array([[500.0], [500.0]])
    rate = np.array([[7.2], [7.2]])
    few = _event_yards(exposure, rate, np.array([[290], [290]]))
    many = _event_yards(exposure, rate, np.array([[360], [360]]))

    assert np.array_equal(few, many)


def test_no_completed_events_means_no_yards():
    # The one thing the event count must still enforce: you cannot gain
    # receiving yards without a reception.
    yards = _event_yards(np.array([[40.0]]), np.array([[8.0]]), np.array([[0]]))

    assert yards[0, 0] == 0


def test_yards_are_never_negative():
    # rush_yards_per_carry has a lower bound of -2.0, so a low draw against a
    # small carry count can go negative; a season yardage total that does is a
    # scoring bug rather than a plausible outcome.
    yards = _event_yards(np.array([[6.0]]), np.array([[-1.5]]), np.array([[6]]))

    assert yards[0, 0] == 0


def test_the_construction_adds_no_variance_of_its_own():
    """Conditional on one efficiency draw, yards are determined.

    All the season-to-season spread should come from the efficiency posterior,
    where it is estimated, rather than from an arithmetic choice in the
    simulator.
    """
    draws = 5000
    exposure = np.full((1, draws), 500.0)
    rate = np.full((1, draws), 7.2)
    events = np.random.default_rng(0).binomial(500, 0.65, size=(1, draws))

    yards = _event_yards(exposure, rate, events)

    assert yards.std() == 0.0
