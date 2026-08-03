"""Role innovation should widen a room without reallocating it.

The role models perturb a log-odds vector with Gaussian noise and take a
softmax. The softmax is not linear, so renormalization takes probability mass
from whoever leads the room and spreads it over everyone else — a transfer no
model term claims is real, and one that grows with how concentrated the room is.
A quarterback room whose starter holds 0.90 gives up 0.0267 of share at the
default innovation scale of 0.60, about nine tenths of an attempt per game.
"""

import numpy as np
import pytest

from ffmodel.models.base import mean_preserving_shares, simplex_shares


def _room(prior, draws, scale, seed=0):
    eta = np.repeat(np.log(np.asarray(prior))[None, :, None], draws, axis=2)
    live = np.ones(eta.shape, dtype=bool)
    noise = np.random.default_rng(seed).normal(size=eta.shape) * scale
    return eta, eta + noise, live


def test_the_uncorrected_softmax_taxes_the_leader():
    prior = [0.90, 0.08, 0.02]
    _, perturbed, live = _room(prior, 20000, 0.60)

    naive = simplex_shares(perturbed, live).mean(axis=2)[0]

    assert naive[0] == pytest.approx(0.8736, abs=5e-4)
    assert naive[0] < prior[0]
    # The mass does not vanish; it is handed to the players behind him.
    assert naive[1] > prior[1] and naive[2] > prior[2]
    assert naive.sum() == pytest.approx(1.0)


def test_the_correction_returns_the_draw_average_to_the_allocation():
    prior = [0.90, 0.08, 0.02]
    baseline, perturbed, live = _room(prior, 20000, 0.60)

    fixed = mean_preserving_shares(baseline, perturbed, live).mean(axis=2)[0]

    assert fixed == pytest.approx(prior, abs=1e-6)


def test_a_flat_room_needed_less_correction_than_a_concentrated_one():
    # Worth stating because it predicts where this shows up: the quarterback
    # room is the most concentrated simplex in the pipeline, and target and
    # carry rooms are much flatter.
    flat = [0.26, 0.22, 0.18, 0.14, 0.10, 0.06, 0.04]
    _, perturbed, live = _room(flat, 20000, 0.60, seed=1)

    naive = simplex_shares(perturbed, live).mean(axis=2)[0]

    assert abs(naive[0] - flat[0]) < 0.01


def test_the_correction_is_a_location_shift_and_keeps_every_contrast():
    """The churn the innovation exists to represent survives untouched.

    The correction adds a per-player constant in log space, so every pairwise
    log-odds contrast between two players is exactly what it was before. That is
    the quantity the innovation actually models — how far the room can reshuffle
    — and it is invariant to eight decimal places.
    """
    baseline, perturbed, live = _room([0.90, 0.08, 0.02], 20000, 0.60)

    naive = simplex_shares(perturbed, live)
    fixed = mean_preserving_shares(baseline, perturbed, live)

    contrast = lambda p: np.log(p[0, 0]) - np.log(p[0, 1])
    assert contrast(fixed).std() == pytest.approx(contrast(naive).std(), abs=1e-8)


def test_share_scale_spread_compresses_because_the_mean_moved_toward_one():
    """A consequence worth naming rather than asserting away.

    Restoring the leader to 0.90 puts him nearer the ceiling of 1.0, and the
    simplex leaves less room above than below, so his share-scale standard
    deviation falls — here 0.086 to 0.072. This is not the correction damping
    the model: on the log-odds scale where the innovation is defined, nothing
    changed at all. It does mean the leader's predictive interval narrows
    slightly along with the point estimate rising, which is the coverage
    behaviour to watch on the walk-forward.
    """
    baseline, perturbed, live = _room([0.90, 0.08, 0.02], 20000, 0.60)

    naive = simplex_shares(perturbed, live)
    fixed = mean_preserving_shares(baseline, perturbed, live)

    assert fixed[0, 0].std() < naive[0, 0].std()
    assert fixed[0, 0].std() > 0.05


def test_every_draw_still_lands_on_the_simplex():
    baseline, perturbed, live = _room([0.5, 0.3, 0.2], 500, 0.75)

    fixed = mean_preserving_shares(baseline, perturbed, live)

    assert np.allclose(fixed.sum(axis=1), 1.0)
    assert (fixed >= 0).all()


def test_unsupported_players_get_exactly_zero_not_a_rounding_crumb():
    baseline, perturbed, live = _room([0.5, 0.3, 0.2], 200, 0.60)
    live = live.copy()
    live[0, 2, :] = False

    fixed = mean_preserving_shares(baseline, perturbed, live)

    assert (fixed[0, 2] == 0.0).all()
    assert np.allclose(fixed.sum(axis=1), 1.0)


def test_a_gate_that_closes_on_some_draws_still_moves_the_mean():
    # The gate is an estimated component and is supposed to change the expected
    # share. Only the innovation is corrected, so a player gated out of half the
    # draws must still average well below his allocation.
    baseline, perturbed, live = _room([0.6, 0.4], 4000, 0.60)
    live = live.copy()
    live[0, 0, ::2] = False

    fixed = mean_preserving_shares(baseline, perturbed, live).mean(axis=2)[0]

    assert fixed[0] < 0.35
    assert fixed.sum() == pytest.approx(1.0)


def test_a_group_with_nobody_supported_yields_zeros_rather_than_nan():
    baseline, perturbed, live = _room([0.5, 0.5], 10, 0.60)
    live = np.zeros_like(live)

    fixed = mean_preserving_shares(baseline, perturbed, live)

    assert np.isfinite(fixed).all()
    assert (fixed == 0.0).all()


def test_a_one_player_room_is_left_alone():
    baseline, perturbed, live = _room([1.0], 100, 0.60)

    fixed = mean_preserving_shares(baseline, perturbed, live)

    assert np.allclose(fixed, 1.0)


def test_the_pipeline_flag_reaches_the_three_allocation_layers():
    from ffmodel.models.volume_season_average import SeasonAverageVolumePipeline

    pipeline = SeasonAverageVolumePipeline(mean_preserving_innovation=True)
    pipeline._enable_mean_preserving_innovation()

    assert pipeline.workload_model.mean_preserving_innovation
    assert pipeline.target_model.mean_preserving_innovation
    assert pipeline.carry_model.mean_preserving_innovation
    # The snap and eligibility layers are per-player: nothing renormalizes, so
    # there is no Jensen gap to close and no flag to carry.
    assert not hasattr(pipeline.snap_model, "mean_preserving_innovation")
    assert not hasattr(
        pipeline.carry_eligibility_model, "mean_preserving_innovation"
    )
