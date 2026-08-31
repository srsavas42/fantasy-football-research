"""Season shares should fall with availability at elasticity one, not less.

The default allocation multiplies a player's weight by season-average
availability and renormalises once, which is ``softmax(E[presence])`` standing
in for ``E[softmax(presence)]``. Those differ, in a direction that hands a
concentrated player back most of what his absence should have cost him, and
nothing in the pipeline raises when it happens -- the share simply comes out
too high. Within-player observed elasticities are 1.08 (RB carries), 0.99
(targets) and 1.20 (QB pass attempts) on snap-based availability, so one is the
number to reproduce.
"""

from __future__ import annotations

import numpy as np
import pytest

from ffmodel.models.base import simplex_shares
from ffmodel.models.volume_season_average import _per_game_shares


def _room(weights, exposure, *, games=17, seed=0, draws=4000):
    """One room, repeated across draws so the Monte Carlo error is small."""
    w = np.asarray(weights, dtype=float)
    eta = np.log(w)[None, :, None] * np.ones((1, len(w), draws))
    live = np.ones((1, len(w), draws), dtype=bool)
    e = np.asarray(exposure, dtype=float)[None, :, None] * np.ones((1, len(w), draws))
    per_game = _per_game_shares(eta, live, e, games=games, seed=seed)
    default = simplex_shares(eta + np.log(e), live)
    return default[0, :, :].mean(axis=1), per_game[0, :, :].mean(axis=1)


def _closed_form_bias(s: float, e: float) -> float:
    return 1.0 / (1.0 - s * (1.0 - e))


@pytest.mark.parametrize("share,exposure", [(0.10, 0.55), (0.50, 0.55), (0.90, 0.55), (0.67, 0.75)])
def test_the_default_overshoots_by_the_closed_form(share, exposure):
    """Two players: the bias is 1/(1-s(1-e)) and the fix removes it."""
    default, per_game = _room([share, 1 - share], [exposure, 1.0])
    truth = share * exposure
    assert per_game[0] == pytest.approx(truth, rel=0.03)
    assert default[0] / truth == pytest.approx(_closed_form_bias(share, exposure), rel=0.03)


def test_elasticity_is_one_for_a_workhorse_back():
    """A back at 67% of his room, halving his availability, halves his share."""
    _, full = _room([0.67, 0.13, 0.11, 0.09], [1.0, 0.9, 0.9, 0.9])
    _, half = _room([0.67, 0.13, 0.11, 0.09], [0.5, 0.9, 0.9, 0.9])
    assert half[0] / full[0] == pytest.approx(0.5, rel=0.05)


def test_elasticity_is_one_for_a_starting_quarterback():
    """The most concentrated room, where the default is worst."""
    _, full = _room([0.92, 0.08], [1.0, 1.0])
    _, half = _room([0.92, 0.08], [0.5, 1.0])
    assert half[0] / full[0] == pytest.approx(0.5, rel=0.05)


def test_elasticity_is_one_for_a_dilute_receiver():
    """Least affected, but it must not be made worse."""
    weights = [0.10, 0.22, 0.20, 0.18, 0.16, 0.14]
    _, full = _room(weights, [1.0] + [0.9] * 5)
    _, half = _room(weights, [0.5] + [0.9] * 5)
    assert half[0] / full[0] == pytest.approx(0.5, rel=0.06)


def test_the_default_understates_that_elasticity_most_where_share_is_high():
    """The ordering the streams inherit: QB pass worst, targets least."""
    def default_elasticity(weights):
        full, _ = _room(weights, [1.0] + [0.9] * (len(weights) - 1))
        half, _ = _room(weights, [0.5] + [0.9] * (len(weights) - 1))
        return np.log(half[0] / full[0]) / np.log(0.5)

    quarterback = default_elasticity([0.92, 0.08])
    workhorse = default_elasticity([0.67, 0.13, 0.11, 0.09])
    receiver = default_elasticity([0.10, 0.22, 0.20, 0.18, 0.16, 0.14])
    assert quarterback < workhorse < receiver < 1.0


def test_shares_still_sum_to_one():
    """The team's carries all go somewhere; the fix may not lose volume."""
    _, per_game = _room([0.67, 0.13, 0.11, 0.09], [0.5, 0.9, 0.9, 0.9])
    assert per_game.sum() == pytest.approx(1.0, rel=1e-6)


def test_freed_share_goes_to_teammates():
    _, full = _room([0.67, 0.13, 0.11, 0.09], [1.0, 0.9, 0.9, 0.9])
    _, half = _room([0.67, 0.13, 0.11, 0.09], [0.5, 0.9, 0.9, 0.9])
    assert (half[1:] > full[1:]).all()


def test_full_availability_reproduces_the_default():
    """With nobody missing games the two paths are the same allocation."""
    default, per_game = _room([0.5, 0.3, 0.2], [1.0, 1.0, 1.0])
    assert per_game == pytest.approx(default, rel=1e-6)


def test_an_emptied_room_still_allocates_its_volume():
    """A week where everyone is out must not drop the team's carries."""
    _, per_game = _room([0.5, 0.5], [0.02, 0.02], draws=2000)
    assert per_game.sum() == pytest.approx(1.0, rel=1e-6)


def test_zero_games_is_rejected():
    eta = np.zeros((1, 2, 3))
    live = np.ones((1, 2, 3), dtype=bool)
    with pytest.raises(ValueError, match="allocation_games"):
        _per_game_shares(eta, live, np.full((1, 2, 3), 0.5), games=0, seed=0)
