"""Small fitting primitives, NumPy only.

The package's metrics are deliberately framework-free, and these follow the same
rule: a weekly walk-forward fits several hundred small models, and a dependency
that has to be installed before a validation script runs is a dependency that
stops the validation script from running.

Three pieces, each with one job:

:func:`ridge`
    Penalised least squares with the intercept left unpenalised and the columns
    standardised, so the penalty means the same thing for a feature measured in
    targets and one measured in share-of-team.

:func:`logistic`
    Newton-Raphson with a ridge penalty. Weekly availability separates almost
    perfectly for some players -- a starter with forty consecutive appearances
    drives an unpenalised coefficient to infinity -- so the penalty is load
    bearing rather than decorative.

:func:`local_residual_pool`
    The residual machinery the season layer's rank curve already uses: draw the
    spread from training rows whose fitted value is near this row's, rather than
    from one global pool. It buys the right shape without naming a family, and
    the right heteroskedasticity for free -- a player projected for twenty
    points has a wider spread than one projected for four, and no variance
    function has to be specified to say so.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Standardizer:
    """Column means and scales, with zero-variance columns left alone."""

    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, x: np.ndarray) -> "Standardizer":
        x = np.asarray(x, float)
        mean = x.mean(axis=0)
        scale = x.std(axis=0)
        scale = np.where(scale > 1e-12, scale, 1.0)
        return cls(mean=mean, scale=scale)

    def apply(self, x: np.ndarray) -> np.ndarray:
        return (np.asarray(x, float) - self.mean) / self.scale


@dataclass
class Ridge:
    """Penalised least squares with an unpenalised intercept."""

    coefficients: np.ndarray
    intercept: float
    standardizer: Standardizer

    @classmethod
    def fit(cls, x: np.ndarray, y: np.ndarray, penalty: float = 1.0) -> "Ridge":
        x = np.asarray(x, float)
        y = np.asarray(y, float).reshape(-1)
        if len(x) != len(y):
            raise ValueError("design and response must have the same length")
        if len(x) == 0:
            raise ValueError("cannot fit a ridge on no rows")
        standardizer = Standardizer.fit(x)
        z = standardizer.apply(x)
        centre = float(y.mean())
        gram = z.T @ z + penalty * np.eye(z.shape[1])
        coefficients = np.linalg.solve(gram, z.T @ (y - centre))
        return cls(coefficients=coefficients, intercept=centre, standardizer=standardizer)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.standardizer.apply(x) @ self.coefficients + self.intercept


@dataclass
class Logistic:
    """Ridge-penalised logistic regression by Newton-Raphson."""

    coefficients: np.ndarray
    intercept: float
    standardizer: Standardizer

    @classmethod
    def fit(
        cls,
        x: np.ndarray,
        y: np.ndarray,
        penalty: float = 1.0,
        *,
        iterations: int = 40,
        tolerance: float = 1e-8,
    ) -> "Logistic":
        x = np.asarray(x, float)
        y = np.asarray(y, float).reshape(-1)
        if len(x) != len(y):
            raise ValueError("design and response must have the same length")
        if len(x) == 0:
            raise ValueError("cannot fit a logistic on no rows")
        standardizer = Standardizer.fit(x)
        z = np.column_stack([np.ones(len(x)), standardizer.apply(x)])
        beta = np.zeros(z.shape[1])
        # The intercept is not penalised, so it can move to the base rate freely.
        ridge_diagonal = np.full(z.shape[1], penalty)
        ridge_diagonal[0] = 0.0
        for _ in range(iterations):
            eta = np.clip(z @ beta, -30.0, 30.0)
            p = 1.0 / (1.0 + np.exp(-eta))
            weight = np.clip(p * (1.0 - p), 1e-9, None)
            gradient = z.T @ (y - p) - ridge_diagonal * beta
            hessian = (z * weight[:, None]).T @ z + np.diag(ridge_diagonal)
            try:
                step = np.linalg.solve(hessian, gradient)
            except np.linalg.LinAlgError:
                step = np.linalg.lstsq(hessian, gradient, rcond=None)[0]
            beta = beta + step
            if np.max(np.abs(step)) < tolerance:
                break
        return cls(
            coefficients=beta[1:], intercept=float(beta[0]), standardizer=standardizer
        )

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        eta = self.standardizer.apply(x) @ self.coefficients + self.intercept
        return 1.0 / (1.0 + np.exp(-np.clip(eta, -30.0, 30.0)))


# Below this many rows a local pool is noise rather than a local spread; the
# whole residual set is used instead. Mirrors the season layer's rank curve.
MIN_RESIDUALS = 60


@dataclass
class LocalResiduals:
    """Residuals indexed by the fitted value they came from.

    Draws are taken from training rows whose fit is near the row being
    predicted, which is what makes the spread scale with the projection without
    a variance model.
    """

    fitted: np.ndarray
    residual: np.ndarray
    quantiles: int = 12

    @classmethod
    def fit(
        cls, fitted: np.ndarray, observed: np.ndarray, quantiles: int = 12
    ) -> "LocalResiduals":
        fitted = np.asarray(fitted, float).reshape(-1)
        observed = np.asarray(observed, float).reshape(-1)
        keep = np.isfinite(fitted) & np.isfinite(observed)
        order = np.argsort(fitted[keep], kind="mergesort")
        return cls(
            fitted=fitted[keep][order],
            residual=(observed[keep] - fitted[keep])[order],
            quantiles=quantiles,
        )

    def draw(self, fitted: np.ndarray, draws: int, rng: np.random.Generator) -> np.ndarray:
        """``(len(fitted), draws)`` residuals drawn from nearby fitted values."""
        fitted = np.asarray(fitted, float).reshape(-1)
        pool = self.residual
        if len(pool) < MIN_RESIDUALS:
            raise ValueError(
                f"a residual pool needs at least {MIN_RESIDUALS} rows, got {len(pool)}"
            )
        # Bucket by rank of the fitted value so every bucket holds the same
        # number of training rows regardless of how the projections pile up.
        edges = np.quantile(self.fitted, np.linspace(0.0, 1.0, self.quantiles + 1))
        edges[0], edges[-1] = -np.inf, np.inf
        bucket = np.clip(
            np.searchsorted(edges, fitted, side="right") - 1, 0, self.quantiles - 1
        )
        out = np.empty((len(fitted), draws), dtype=float)
        source = np.searchsorted(edges, self.fitted, side="right") - 1
        source = np.clip(source, 0, self.quantiles - 1)
        for index in range(self.quantiles):
            want = bucket == index
            if not want.any():
                continue
            local = pool[source == index]
            if len(local) < MIN_RESIDUALS:
                local = pool
            # Index with random integers rather than calling ``rng.choice``.
            # Identical semantics -- uniform, with replacement -- and around an
            # order of magnitude faster, which matters because the rest-of-season
            # simulator calls this once per remaining game per fold.
            picks = rng.integers(0, len(local), size=(int(want.sum()), draws))
            out[want] = local[picks]
        return out
