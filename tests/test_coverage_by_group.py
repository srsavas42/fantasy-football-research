"""Coverage decomposed by group.

A pooled coverage number can pass a promotion gate while a subgroup fails it.
Rescaling dispersion globally then trades a passing group against a failing one,
which reads as an unfixable sharpness/calibration trade-off but is a
misdiagnosis. These cover the decomposition itself.
"""

import numpy as np
import pytest

from ffmodel.evaluation.metrics import coverage_by_group


def _samples(centres, spread, draws=400, seed=0):
    rng = np.random.default_rng(seed)
    return np.asarray(centres, dtype=float)[:, None] + rng.normal(
        0.0, spread, size=(len(centres), draws)
    )


def test_a_failing_group_is_visible_behind_a_passing_pool():
    # One group is well calibrated; the other is far too confident. Pooled
    # coverage lands between them and hides the second.
    good = _samples([0.0] * 100, 1.0)
    bad = _samples([0.0] * 100, 0.05, seed=1)
    samples = np.vstack([good, bad])
    observed = np.concatenate(
        [np.random.default_rng(2).normal(0, 1, 100), np.random.default_rng(3).normal(0, 1, 100)]
    )
    groups = ["good"] * 100 + ["bad"] * 100

    out = coverage_by_group(observed, samples, groups, levels=(0.95,))

    assert out["good"]["coverage"][0.95] > 0.85
    assert out["bad"]["coverage"][0.95] < 0.30
    assert out["good"]["n"] == 100 and out["bad"]["n"] == 100


def test_miss_direction_separates_the_two_tails():
    # Every observation sits far below its interval: a collapsed role, not a
    # mis-set spread. The direction is what distinguishes them.
    samples = _samples([100.0] * 50, 5.0)
    observed = np.zeros(50)

    out = coverage_by_group(observed, samples, ["qb"] * 50, levels=(0.95,))

    assert out["qb"]["below"] == 50
    assert out["qb"]["above"] == 0


def test_every_requested_level_is_reported():
    samples = _samples([0.0] * 40, 1.0)
    observed = np.zeros(40)

    out = coverage_by_group(observed, samples, ["a"] * 40, levels=(0.5, 0.8, 0.95))

    assert sorted(out["a"]["coverage"]) == [0.5, 0.8, 0.95]


def test_group_labels_must_align_with_observations():
    with pytest.raises(ValueError, match="one label per observation"):
        coverage_by_group(np.zeros(5), _samples([0.0] * 5, 1.0), ["a", "b"])
