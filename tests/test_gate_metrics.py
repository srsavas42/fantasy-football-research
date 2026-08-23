"""The four metrics added to the acceptance gate.

The decomposition gets the most attention because it is the one that can be
subtly wrong and still look plausible: every part is a positive number of about
the right size whether or not the algebra is right. The identity against
``empirical_crps`` is what makes it checkable.
"""

from __future__ import annotations

import numpy as np
import pytest

from ffmodel.evaluation.metrics import (
    concordance,
    crps_decomposition,
    crps_skill_score,
    empirical_crps,
    ordering_metrics,
    pit_calibration,
    spearman,
    top_k_hit_rate,
)


def _forecast(rng, n=3000, members=200, sd=1.0, shift=0.0, informative=True):
    signal = rng.normal(0.0, 3.0, n)
    truth = signal + rng.normal(0.0, 1.0, n)
    centre = signal if informative else np.zeros(n)
    return truth, centre[:, None] + shift + rng.normal(0.0, sd, (n, members))


@pytest.mark.parametrize(
    "sd,shift,informative",
    [(1.0, 0.0, True), (0.25, 0.0, True), (4.0, 0.0, True), (1.0, 2.0, True), (3.2, 0.0, False)],
)
def test_the_decomposition_sums_back_to_the_crps(sd, shift, informative):
    """reliability + potential == CRPS, or the split means nothing."""
    truth, draws = _forecast(np.random.default_rng(0), sd=sd, shift=shift, informative=informative)
    parts = crps_decomposition(truth, draws)

    assert parts["crps"] == pytest.approx(empirical_crps(truth, draws).mean(), rel=1e-9)
    assert parts["potential"] == pytest.approx(
        parts["uncertainty"] - parts["resolution"], rel=1e-9
    )


def test_a_climatological_forecast_is_calibrated_and_has_no_resolution():
    """The case that pins both ends of the decomposition at once."""
    truth, draws = _forecast(np.random.default_rng(1), sd=3.2, informative=False)
    parts = crps_decomposition(truth, draws)

    assert parts["reliability"] < 0.02
    assert abs(parts["resolution"]) < 0.02
    assert parts["potential"] == pytest.approx(parts["uncertainty"], abs=0.02)


def test_an_informative_forecast_earns_resolution():
    truth, draws = _forecast(np.random.default_rng(2), sd=1.0)
    informative = crps_decomposition(truth, draws)
    truth2, flat = _forecast(np.random.default_rng(2), sd=3.2, informative=False)
    climatology = crps_decomposition(truth2, flat)

    assert informative["resolution"] > climatology["resolution"] + 0.5


def test_a_biased_forecast_pays_in_reliability_not_resolution():
    """Bias is a calibration fault; it should not look like lost information."""
    rng = np.random.default_rng(3)
    truth, honest = _forecast(rng, sd=1.0)
    shifted = honest + 2.0
    a, b = crps_decomposition(truth, honest), crps_decomposition(truth, shifted)

    assert b["reliability"] > a["reliability"] + 0.5
    assert b["resolution"] == pytest.approx(a["resolution"], abs=0.15)


def test_the_skill_score_is_zero_against_itself_and_one_when_perfect():
    assert crps_skill_score(5.0, 5.0) == pytest.approx(0.0)
    assert crps_skill_score(0.0, 5.0) == pytest.approx(1.0)
    assert crps_skill_score(10.0, 5.0) == pytest.approx(-1.0)
    with pytest.raises(ValueError, match="must be positive"):
        crps_skill_score(1.0, 0.0)


def test_ordering_is_perfect_on_itself_and_chance_on_noise():
    rng = np.random.default_rng(4)
    observed = rng.normal(100, 30, 300)
    assert spearman(observed, observed) == pytest.approx(1.0)
    assert concordance(observed, observed) == pytest.approx(1.0)
    assert top_k_hit_rate(observed, observed, 12) == pytest.approx(1.0)

    noise = rng.normal(100, 30, 300)
    assert abs(spearman(noise, observed)) < 0.2
    assert concordance(noise, observed) == pytest.approx(0.5, abs=0.06)


def test_ordering_is_scored_within_position():
    """A pooled rank correlation is inflated by the gap between positions.

    Here the projection knows only which position a player plays and nothing
    about individuals. Pooled that looks like real skill; within position it is
    correctly worthless.
    """
    rng = np.random.default_rng(5)
    positions = np.repeat(["QB", "RB", "WR", "TE"], 60)
    level = {"QB": 300.0, "RB": 200.0, "WR": 180.0, "TE": 110.0}
    observed = np.array([level[p] + rng.normal(0, 15) for p in positions])
    projected = np.array([level[p] for p in positions])

    out = ordering_metrics(projected, observed, positions, k=12)
    assert out["spearman"] > 0.85
    assert abs(out["within_group_spearman"]) < 0.2


def test_a_tie_in_the_projection_is_worth_half_a_coin_flip():
    projected = np.zeros(40)
    observed = np.arange(40, dtype=float)
    assert concordance(projected, observed) == pytest.approx(0.5)


def test_pit_names_the_shape_it_sees():
    rng = np.random.default_rng(6)
    truth, calibrated = _forecast(rng, sd=1.0)
    assert pit_calibration(truth, calibrated)["shape"] == "flat"

    truth, tight = _forecast(rng, sd=0.25)
    assert "over-confident" in pit_calibration(truth, tight)["shape"]

    truth, wide = _forecast(rng, sd=4.0)
    assert "over-dispersed" in pit_calibration(truth, wide)["shape"]


def test_a_shifted_forecast_is_reported_as_shifted_not_over_confident():
    """A shifted forecast also piles mass in one tail.

    Reporting that as over-confidence would send a reader to widen the
    posterior when the centre is what is wrong.
    """
    rng = np.random.default_rng(7)
    truth, draws = _forecast(rng, sd=1.0, shift=2.0)
    out = pit_calibration(truth, draws)

    assert out["shape"] == "shifted (projections run high)"
    assert out["mean_pit"] < 0.3
