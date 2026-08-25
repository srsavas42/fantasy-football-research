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


def crps_decomposition(observed, samples) -> dict[str, float]:
    """Split the CRPS into reliability and potential CRPS (Hersbach 2000).

    ``CRPS = reliability + potential``, and ``potential = uncertainty -
    resolution``. The three parts answer different questions:

    ``reliability``
        Are the stated probabilities honest? Zero for a calibrated forecast,
        whatever its sharpness. This is the part a wider or narrower posterior
        moves.
    ``resolution``
        Does the forecast successfully tell different outcomes apart? Higher is
        better, and it is bounded above by ``uncertainty``. A forecast that
        issues the climatological distribution to everyone has resolution zero
        no matter how well calibrated it is.
    ``uncertainty``
        The CRPS of the climatological forecast -- the score of knowing nothing
        beyond the marginal distribution of outcomes. A property of the
        population, not of the model, so it is the yardstick the other two are
        read against.

    This matters here because the defect this package spent a session on was a
    *resolution* failure: the availability layer was unbiased pooled while
    missing in opposite directions on drafted and undrafted players, which is
    what a forecast does when it cannot separate two groups. That was diagnosed
    indirectly, from bias splits and an in-sample refit. This measures it.

    Implemented over the ensemble intervals rather than by binning the PIT, so
    the identity above holds exactly against :func:`empirical_crps` -- which the
    tests check, because a decomposition that does not sum back is a decoration.
    """
    y, draws = _inputs(observed, samples)
    ordered = np.sort(draws, axis=1)
    cases, members = ordered.shape

    # Interior intervals [x_i, x_(i+1)), each carrying forecast probability i/n.
    left = ordered[:, :-1]
    right = ordered[:, 1:]
    width = right - left
    target = y[:, None]
    # Below the interval, inside it, or above it -- the three cases Hersbach
    # splits each interval's contribution into.
    alpha = np.where(target > right, width, np.where(target < left, 0.0, target - left))
    beta = np.where(target > right, 0.0, np.where(target < left, width, right - target))
    alpha = np.clip(alpha, 0.0, None)
    beta = np.clip(beta, 0.0, None)

    probability = np.arange(1, members) / members
    alpha_bar = alpha.mean(axis=0)
    beta_bar = beta.mean(axis=0)

    # The two outlier bins, where the observation falls outside the ensemble.
    # They carry no interval width of their own, so their frequency has to be
    # counted directly rather than read off a width.
    below_all = y < ordered[:, 0]
    above_all = y > ordered[:, -1]
    outlier_low = float(below_all.mean())
    outlier_high = float(above_all.mean())
    beta_low = float(np.where(below_all, ordered[:, 0] - y, 0.0).mean())
    alpha_high = float(np.where(above_all, y - ordered[:, -1], 0.0).mean())

    spread = alpha_bar + beta_bar
    # Observed frequency that the outcome fell above each interval.
    with np.errstate(divide="ignore", invalid="ignore"):
        frequency = np.where(spread > 0, beta_bar / np.where(spread > 0, spread, 1.0), 0.0)
        # Each outlier bin's average width *given* that the observation landed
        # there, so that reliability and potential sum back to the bin's own
        # contribution to the CRPS. Both denominators are the outlier rate: the
        # bin only exists on the cases that fall in it. Using (1 - rate) for the
        # upper bin is the natural-looking mistake and breaks the identity by
        # exactly the amount the ensemble is over-confident.
        low_spread = beta_low / outlier_low if outlier_low > 0 else 0.0
        high_spread = alpha_high / outlier_high if outlier_high > 0 else 0.0

    reliability = float((spread * (frequency - probability) ** 2).sum())
    reliability += low_spread * outlier_low**2
    reliability += high_spread * outlier_high**2

    potential = float((spread * frequency * (1.0 - frequency)).sum())
    potential += low_spread * outlier_low * (1.0 - outlier_low)
    potential += high_spread * outlier_high * (1.0 - outlier_high)

    # Climatology: score each outcome against the pooled distribution of all
    # outcomes. Computed on the same rows being scored, so it is the internal
    # uncertainty of this population rather than an outside reference.
    uncertainty = float(
        empirical_crps(y, np.tile(np.sort(y)[None, :], (cases, 1))).mean()
    )
    return {
        "crps": float(reliability + potential),
        "reliability": reliability,
        "potential": potential,
        "uncertainty": uncertainty,
        "resolution": uncertainty - potential,
    }


def crps_skill_score(candidate, reference) -> float:
    """Fraction of a reference forecast's CRPS that a candidate removes.

    ``1 - candidate / reference``: zero means no better than the reference, one
    means perfect, negative means worse. Reported because a relative change is
    hard to size on its own -- "CRPS 1.1% better" says nothing about how much
    room there was, while a skill score against the draft board says how much of
    the achievable gap was closed.
    """
    candidate = float(np.mean(candidate))
    reference = float(np.mean(reference))
    if reference <= 0:
        raise ValueError("the reference CRPS must be positive to form a skill score")
    return 1.0 - candidate / reference


def _ranks(values: np.ndarray) -> np.ndarray:
    """Average ranks, so ties do not manufacture an ordering."""
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(1, len(values) + 1, dtype=float)
    unique, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    if (counts > 1).any():
        sums = np.zeros(len(unique))
        np.add.at(sums, inverse, ranks)
        ranks = (sums / counts)[inverse]
    return ranks


def spearman(projected, observed) -> float:
    """Rank correlation, computed as Pearson on average ranks."""
    projected = np.asarray(projected, dtype=float).reshape(-1)
    observed = np.asarray(observed, dtype=float).reshape(-1)
    if len(projected) != len(observed):
        raise ValueError("projected and observed must be the same length")
    if len(projected) < 3:
        return float("nan")
    a, b = _ranks(projected), _ranks(observed)
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def concordance(projected, observed) -> float:
    """Share of player pairs the projection puts in the right order.

    Ties in the projection count as half, which is what a coin flip between two
    equally-projected players is worth. 0.5 is no ordering skill.
    """
    projected = np.asarray(projected, dtype=float).reshape(-1)
    observed = np.asarray(observed, dtype=float).reshape(-1)
    if len(projected) < 2:
        return float("nan")
    dp = projected[:, None] - projected[None, :]
    do = observed[:, None] - observed[None, :]
    upper = np.triu(np.ones_like(dp, dtype=bool), k=1)
    comparable = upper & (do != 0)
    if not comparable.any():
        return float("nan")
    agree = np.sign(dp[comparable]) == np.sign(do[comparable])
    tied = dp[comparable] == 0
    return float((agree.sum() + 0.5 * tied.sum()) / comparable.sum())


def top_k_hit_rate(projected, observed, k: int) -> float:
    """Share of the projected top ``k`` that finished in the observed top ``k``.

    The metric a drafter lives with: not how close the numbers were, but whether
    the players named were the right players.
    """
    projected = np.asarray(projected, dtype=float).reshape(-1)
    observed = np.asarray(observed, dtype=float).reshape(-1)
    k = int(min(k, len(projected)))
    if k < 1:
        return float("nan")
    picked = set(np.argsort(-projected, kind="mergesort")[:k].tolist())
    actual = set(np.argsort(-observed, kind="mergesort")[:k].tolist())
    return len(picked & actual) / k


def ordering_metrics(projected, observed, groups=None, k: int = 12) -> dict[str, object]:
    """Rank agreement overall and within each group.

    Ordering is scored *within position* because that is how the projection is
    consumed -- a drafter compares receivers to receivers. A pooled rank
    correlation is inflated by the gap between positions, which no one needs a
    model to know: quarterbacks outscore tight ends, and getting that right is
    not skill.
    """
    projected = np.asarray(projected, dtype=float).reshape(-1)
    observed = np.asarray(observed, dtype=float).reshape(-1)
    keep = np.isfinite(projected) & np.isfinite(observed)
    out: dict[str, object] = {
        "n": int(keep.sum()),
        "spearman": spearman(projected[keep], observed[keep]),
        "concordance": concordance(projected[keep], observed[keep]),
        "top_k": top_k_hit_rate(projected[keep], observed[keep], k),
        "k": int(k),
    }
    if groups is None:
        return out
    labels = np.asarray(groups).reshape(-1)
    if len(labels) != len(projected):
        raise ValueError("groups must have one label per observation")
    per_group: dict[str, dict[str, float]] = {}
    weights: list[tuple[float, float, float]] = []
    for label in dict.fromkeys(labels[keep].tolist()):
        mask = keep & (labels == label)
        if mask.sum() < 3:
            continue
        entry = {
            "n": int(mask.sum()),
            "spearman": spearman(projected[mask], observed[mask]),
            "concordance": concordance(projected[mask], observed[mask]),
            "top_k": top_k_hit_rate(projected[mask], observed[mask], k),
        }
        per_group[str(label)] = entry
        weights.append(
            (entry["n"], entry["spearman"], entry["concordance"], entry["top_k"])
        )
    out["by_group"] = per_group
    if weights:
        total = sum(w[0] for w in weights)
        # Size-weighted, so a twelve-man position does not count as much as a
        # three-hundred-man one.
        out["within_group_spearman"] = float(
            sum(w[0] * w[1] for w in weights if np.isfinite(w[1])) / total
        )
        out["within_group_concordance"] = float(
            sum(w[0] * w[2] for w in weights if np.isfinite(w[2])) / total
        )
        # Top-k has to be within group or it is not a real question. Pooled
        # across positions the top twelve is just the twelve highest scorers,
        # which is mostly "quarterbacks outscore tight ends" -- a fact no model
        # is needed for, and one that makes the metric near-constant between
        # arms. Within position it is the set a drafter is actually choosing.
        out["within_group_top_k"] = float(
            sum(w[0] * w[3] for w in weights if np.isfinite(w[3])) / total
        )
    return out


def pit_calibration(observed, samples, bins: int = 10) -> dict[str, object]:
    """Shape of the calibration curve, not two points on it.

    Interval coverage at 80% and 95% samples the curve twice. A forecast can hit
    both and still be wrong in between, or be wrong in a way the two levels
    cannot distinguish -- a uniform widening and a heavy one tail move 95%
    coverage identically.

    The PIT histogram is flat when calibrated, U-shaped when over-confident
    (outcomes keep landing in the tails), and humped when over-dispersed.
    ``deviation`` is the mean absolute departure from flat, in probability
    units, and ``shape`` names the pattern so a table can carry it.
    """
    y, draws = _inputs(observed, samples)
    values = pit_values(y, draws)
    counts, _ = np.histogram(values, bins=bins, range=(0.0, 1.0))
    density = counts / max(counts.sum(), 1) * bins
    deviation = float(np.abs(density - 1.0).mean())
    edge = float(density[0] + density[-1]) / 2.0
    middle = float(density[bins // 2 - 1 : bins // 2 + 1].mean())
    # Bias is checked before shape, because a shifted forecast also piles mass
    # into one tail and would otherwise be reported as over-confident -- the
    # right diagnosis is the centre, not the spread, and the two want different
    # fixes.
    mean_pit = float(values.mean())
    if deviation < 0.15:
        shape = "flat"
    elif abs(mean_pit - 0.5) > 0.05:
        shape = (
            "shifted (projections run low)"
            if mean_pit > 0.5
            else "shifted (projections run high)"
        )
    elif edge > 1.0 and edge > middle:
        shape = "over-confident (tails too heavy)"
    elif middle > 1.0 and middle > edge:
        shape = "over-dispersed (too much mass in the middle)"
    else:
        shape = "skewed"
    return {
        "bins": int(bins),
        "density": [float(v) for v in density],
        "deviation": deviation,
        "shape": shape,
        "mean_pit": mean_pit,
    }
