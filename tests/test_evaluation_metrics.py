import numpy as np
import pytest

from ffmodel.evaluation.metrics import empirical_crps, interval_coverage, pit_values


def test_crps_is_zero_for_perfect_deterministic_forecast():
    score = empirical_crps([2.0], np.full((1, 20), 2.0))
    assert score[0] == pytest.approx(0.0)


def test_crps_equals_absolute_error_for_wrong_deterministic_forecast():
    score = empirical_crps([1.0], np.zeros((1, 20)))
    assert score[0] == pytest.approx(1.0)


def test_interval_coverage_and_pit():
    samples = np.array([[0, 1, 2, 3, 4], [0, 1, 2, 3, 4]], dtype=float)
    coverage = interval_coverage([2, 10], samples, level=0.8)
    assert coverage["coverage"] == pytest.approx(0.5)
    pit = pit_values([2, 10], samples)
    assert pit.tolist() == pytest.approx([0.5, 1.0])


def test_metrics_validate_shape():
    with pytest.raises(ValueError, match="shape"):
        empirical_crps([1, 2], np.ones((3, 4)))
