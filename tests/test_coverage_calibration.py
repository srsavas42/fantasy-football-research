"""Coverage is a count of rare events, and a per-fold rate hides that.

A 95% interval over 84 quarterback rows expects about four misses. A fold that
records 1.000 has zero, which reads as "the intervals cover everything" and
looks like over-widening — but P(0 | n=84, p=0.05) is 0.0135, so seeing it in at
least one of three folds has probability 0.04. Judging that by eye off a rate is
how a noise draw gets written up as a defect, which is what happened once.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from coverage_calibration import _binomial_interval, report


def _run(misses_by_fold, n=84, nominal=0.95, metric="cov95"):
    return {
        fold: {"qb_workload": {"n": n, metric: 1.0 - missed / n}}
        for fold, missed in misses_by_fold.items()
    }


def test_the_interval_brackets_the_expected_count():
    low, high = _binomial_interval(253, 0.05)

    assert low <= 253 * 0.05 <= high
    assert (low, high) == (6, 20)


def test_a_large_n_does_not_overflow():
    """math.comb(1754, k) is an integer too large to convert to a float.

    The target and carry streams are that size, so the direct product overflows
    before it can be scaled down; the mass is built in log space instead.
    """
    low, high = _binomial_interval(1754, 0.05)

    assert 0 < low < 1754 * 0.05 < high < 1754


def test_a_fold_with_zero_misses_still_pools_as_calibrated():
    # The 2024 shape: 8, 5 and 0 misses against 4.2 expected each.
    text = report(_run({"2022": 8, "2023": 5, "2024": 0}), "qb_workload")

    assert "MISCALIBRATED" not in text
    assert "calibrated" in text


def test_a_genuinely_overconfident_run_is_named():
    # The pre-calibration shape: 18, 16 and 11 misses against 4.2 expected.
    text = report(_run({"2022": 18, "2023": 16, "2024": 11}), "qb_workload")

    assert "MISCALIBRATED" in text


def test_over_covering_is_caught_as_well_as_under_covering():
    """Too wide is a calibration failure too, and the carry stream is one.

    Its 80% intervals contain 88% of outcomes, which no test looking only for
    overconfidence would report.
    """
    text = report(
        _run({"2022": 0, "2023": 0, "2024": 0}, n=600, metric="cov95"), "qb_workload"
    )

    assert "MISCALIBRATED" in text


def test_a_stream_the_run_does_not_carry_is_reported_not_crashed():
    assert "not present" in report(_run({"2022": 4}), "nonexistent")
