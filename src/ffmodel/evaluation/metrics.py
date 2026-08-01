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


def coverage_by_group(
    observed, samples, groups, levels=(0.8, 0.95)
) -> dict[object, dict[str, object]]:
    """Interval coverage within each group, and which tail its misses fall in.

    A pooled coverage number can sit comfortably inside a promotion gate while a
    subgroup sits well outside it. Rescaling dispersion globally then trades a
    group that already passes against one that does not, which looks like an
    unfixable sharpness/calibration trade-off but is really a misdiagnosis.

    ``below`` counts observations under the widest interval and ``above`` counts
    those over it, because the two call for different fixes: a heavy lower tail
    is a role that failed to materialise, not a mis-set spread.
    """
    y, draws = _inputs(observed, samples)
    labels = np.asarray(groups).reshape(-1)
    if len(labels) != len(y):
        raise ValueError("groups must have one label per observation")
    widest = max(levels)
    out: dict[object, dict[str, object]] = {}
    for label in dict.fromkeys(labels.tolist()):
        mask = labels == label
        entry: dict[str, object] = {"n": int(mask.sum()), "coverage": {}}
        for level in levels:
            entry["coverage"][float(level)] = interval_coverage(
                y[mask], draws[mask], level=level
            )["coverage"]
        bounds = interval_coverage(y[mask], draws[mask], level=widest)
        entry["below"] = int((y[mask] < bounds["lower"]).sum())
        entry["above"] = int((y[mask] > bounds["upper"]).sum())
        out[label] = entry
    return out


def pit_values(observed, samples) -> np.ndarray:
    """Empirical probability-integral-transform ranks in [0, 1]."""
    y, draws = _inputs(observed, samples)
    below = (draws < y[:, None]).sum(axis=1)
    equal = (draws == y[:, None]).sum(axis=1)
    return (below + 0.5 * equal) / draws.shape[1]
