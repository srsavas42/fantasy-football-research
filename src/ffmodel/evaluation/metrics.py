"""Metrics for posterior predictive samples.

All functions accept samples with shape ``(n_observations, n_draws)``.  They are
kept NumPy-only so validation scripts do not depend on a scoring framework.
"""

from __future__ import annotations

import numpy as np


def _inputs(observed, samples) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(observed, dtype=float).reshape(-1)
    draws = np.asarray(samples, dtype=float)
    if draws.ndim != 2 or draws.shape[0] != len(y):
        raise ValueError("samples must have shape (n_observations, n_draws)")
    if draws.shape[1] == 0:
        raise ValueError("samples must contain at least one posterior draw")
    return y, draws


def empirical_crps(observed, samples) -> np.ndarray:
    """Continuous Ranked Probability Score for each empirical distribution.

    Lower is better.  The sorted-sample identity avoids allocating the
    quadratic pairwise-difference matrix used by the definition.
    """
    y, draws = _inputs(observed, samples)
    ordered = np.sort(draws, axis=1)
    n = ordered.shape[1]
    weights = 2 * np.arange(1, n + 1) - n - 1
    dispersion = (ordered * weights[None, :]).sum(axis=1) / (n * n)
    absolute_error = np.abs(draws - y[:, None]).mean(axis=1)
    return absolute_error - dispersion


def interval_coverage(observed, samples, level: float = 0.8) -> dict[str, object]:
    """Central posterior interval coverage plus its row-level bounds."""
    if not 0.0 < level < 1.0:
        raise ValueError("level must be strictly between 0 and 1")
    y, draws = _inputs(observed, samples)
    tail = (1.0 - level) / 2.0
    lower = np.quantile(draws, tail, axis=1)
    upper = np.quantile(draws, 1.0 - tail, axis=1)
    covered = (y >= lower) & (y <= upper)
    return {
        "level": level,
        "coverage": float(covered.mean()),
        "covered": covered,
        "lower": lower,
        "upper": upper,
    }


def pit_values(observed, samples) -> np.ndarray:
    """Empirical probability-integral-transform ranks in [0, 1]."""
    y, draws = _inputs(observed, samples)
    below = (draws < y[:, None]).sum(axis=1)
    equal = (draws == y[:, None]).sum(axis=1)
    return (below + 0.5 * equal) / draws.shape[1]
