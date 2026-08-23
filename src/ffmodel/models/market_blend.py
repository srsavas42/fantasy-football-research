"""Combine the pipeline's projection with the draft board's.

On the players people actually draft, this package does not beat a rank curve.
Four least-squares parameters -- a log-rank slope and three position dummies --
match it on MAE and beat it on CRPS, and four hypotheses for why have been
tested and eliminated (see docs/adp-ablation-2026-08.md). What survives is not a
defect anyone can go and fix: the component projections are simply less
accurate than consensus.

That makes the question worth asking differently. Two forecasts that make
*different* mistakes combine into a better one even when neither dominates. The
error correlation between these two is about 0.79, and regressing the board's
error on the two forecasts' disagreement gives a slope of +0.41 -- so a
meaningful part of what the model says when it disagrees with the board is
right. Blending beat the board on every scored holdout.

Three choices here are load-bearing.

**Mixture, not averaging.** Both give the same mean, so MAE cannot tell them
apart and the pooled point metrics look interchangeable -- averaging is even
marginally ahead. Intervals are not interchangeable. Averaging paired draws
produces a distribution narrower than either input and wrecks calibration:

    combination   cov80   cov95
    average       0.689   0.892
    mixture       0.847   0.974

against nominal 0.80 and 0.95, over 2023-2025. This is a posterior-predictive
package whose output is a distribution, so the narrow one is not a candidate
regardless of its MAE.

**The weight is the variance-optimal one, estimated as a regression slope**
rather than by scanning a grid. For two forecasts the weight minimising combined
error variance is the coefficient in ``observed - curve = a + b * (model -
curve)``. The grid rule estimates the same thing from a handful of fold-level
summaries and picks the minimum of a noisy curve; the slope uses every row. On
2025 the slope rule beat the grid on both metrics.

**Undrafted players get the model unchanged.** The curve has no rank for them
and nothing to say.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

MODEL_POSITIONS = ("QB", "RB", "WR", "TE")

# Residuals are pooled from ranks within this window of the row being projected,
# so the spread deep on the board is not borrowed from the first round.
RANK_WINDOW = 24

# Below this many rows a local pool is noise; fall back to the position's whole
# residual set rather than resampling a handful of seasons.
MIN_RESIDUALS = 40


def slope_weight(
    observed: np.ndarray, model_mean: np.ndarray, curve_mean: np.ndarray
) -> float:
    """How much of the model's disagreement with the board is right.

    Clipped to [0, 1]: a negative weight means betting against the model, which
    is a different claim needing its own evidence, and above one means
    extrapolating past it.
    """
    observed = np.asarray(observed, float)
    model_mean = np.asarray(model_mean, float)
    curve_mean = np.asarray(curve_mean, float)
    keep = np.isfinite(observed) & np.isfinite(model_mean) & np.isfinite(curve_mean)
    if keep.sum() < 2:
        raise ValueError("the slope weight needs at least two complete rows")
    x = model_mean[keep] - curve_mean[keep]
    y = observed[keep] - curve_mean[keep]
    design = np.column_stack([np.ones(len(x)), x])
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    return float(np.clip(beta[1], 0.0, 1.0))


@dataclass
class RankCurve:
    """Season points from draft rank alone: per-position log fit, local spread.

    The functional form was checked rather than assumed. A kernel-smoothed local
    mean over nearby ranks -- no functional form at all -- scores *worse* than
    the log fit (54.88 against 54.48 MAE), so there is no hidden nonlinear
    structure in the rank-to-points relationship and a more flexible learner
    would be fitting noise.
    """

    coefficients: dict[str, np.ndarray] = field(default_factory=dict)
    residuals: dict[str, np.ndarray] = field(default_factory=dict)
    ranks: dict[str, np.ndarray] = field(default_factory=dict)

    def fit(self, rows: pd.DataFrame, points: np.ndarray) -> "RankCurve":
        """Fit on drafted rows only.

        Fitting on every rostered player instead costs the curve 3.6 CRPS points
        on drafted players -- almost all of its distributional advantage -- so
        the training population is part of the method, not an incidental choice.
        """
        rank = pd.to_numeric(rows["adp_rank"], errors="coerce").to_numpy(float)
        points = np.asarray(points, float)
        drafted = pd.to_numeric(rows["adp_drafted"], errors="coerce").eq(1).to_numpy()
        usable = drafted & np.isfinite(rank) & (rank > 0) & np.isfinite(points)
        if usable.sum() < MIN_RESIDUALS:
            raise ValueError(
                f"the rank curve needs at least {MIN_RESIDUALS} drafted rows, "
                f"got {int(usable.sum())}"
            )
        position = rows["position"].to_numpy()
        for name in MODEL_POSITIONS:
            at = usable & (position == name)
            # Not enough history for this position: use the whole board rather
            # than inventing a curve from a handful of rows.
            fit = at if at.sum() >= MIN_RESIDUALS else usable
            log_rank = np.log(rank[fit])
            coefficients = np.polyfit(log_rank, points[fit], 1)
            self.coefficients[name] = coefficients
            self.residuals[name] = points[fit] - np.polyval(coefficients, log_rank)
            self.ranks[name] = rank[fit]
        return self

    def predict_samples(
        self, rows: pd.DataFrame, draws: int, seed: int = 0
    ) -> np.ndarray:
        """Predictive samples per row; NaN for rows the board does not rank."""
        if not self.coefficients:
            raise RuntimeError("fit the rank curve before predicting")
        rng = np.random.default_rng(seed)
        rank = pd.to_numeric(rows["adp_rank"], errors="coerce").to_numpy(float)
        drafted = pd.to_numeric(rows["adp_drafted"], errors="coerce").eq(1).to_numpy()
        position = rows["position"].to_numpy()
        samples = np.full((len(rows), draws), np.nan, dtype=float)

        for name in MODEL_POSITIONS:
            want = drafted & (position == name) & np.isfinite(rank) & (rank > 0)
            if not want.any():
                continue
            coefficients = self.coefficients.get(name)
            if coefficients is None:
                continue
            residuals = self.residuals[name]
            fit_ranks = self.ranks[name]
            block = rank[want]
            centre = np.polyval(coefficients, np.log(block))
            drawn = np.zeros((len(block), draws), dtype=float)
            for i, value in enumerate(block):
                near = np.abs(fit_ranks - value) <= RANK_WINDOW
                pool = residuals[near] if near.sum() >= MIN_RESIDUALS else residuals
                drawn[i] = centre[i] + rng.choice(pool, size=draws, replace=True)
            # A season total cannot be negative, and a curve linear in log rank
            # goes under zero deep on the board.
            samples[want] = np.maximum(drawn, 0.0)
        return samples


def blend_samples(
    model: np.ndarray, curve: np.ndarray, weight: float, seed: int = 0
) -> np.ndarray:
    """Mixture of the two forecasts: each draw comes from one or the other.

    Rows where the curve has nothing to say -- undrafted, or a position the
    curve could not be fitted for -- keep the model's draws untouched.
    """
    model = np.asarray(model, float)
    curve = np.asarray(curve, float)
    if model.shape != curve.shape:
        raise ValueError(
            f"model and curve samples disagree in shape: {model.shape} vs {curve.shape}"
        )
    if not 0.0 <= weight <= 1.0:
        raise ValueError(f"weight must lie in [0, 1], got {weight}")
    rng = np.random.default_rng(seed)
    take_model = rng.random(model.shape) < weight
    blended = np.where(take_model, model, curve)
    # A whole row of NaN means the board did not rank this player. Any NaN
    # reaching the output would silently poison a mean or a CRPS, so fall back
    # per row rather than trusting the caller to mask afterwards.
    unranked = ~np.isfinite(curve).all(axis=1)
    blended[unranked] = model[unranked]
    return blended


@dataclass
class MarketBlend:
    """Fitted curve plus the weight to give the model against it."""

    weight: float
    curve: RankCurve

    @classmethod
    def fit(
        cls,
        train_rows: pd.DataFrame,
        train_points: np.ndarray,
        *,
        weight: float,
    ) -> "MarketBlend":
        return cls(weight=float(weight), curve=RankCurve().fit(train_rows, train_points))

    def predict_samples(
        self, rows: pd.DataFrame, model_samples: np.ndarray, seed: int = 0
    ) -> np.ndarray:
        curve = self.curve.predict_samples(
            rows, draws=np.asarray(model_samples).shape[1], seed=seed
        )
        return blend_samples(model_samples, curve, self.weight, seed=seed + 1)
