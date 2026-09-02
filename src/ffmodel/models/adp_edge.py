"""Probability that a player beats his draft board, from the model's disagreement.

The projection comparison established three things. The standalone model loses
to an ADP rank curve on both MAE and CRPS. Its disagreement with the board is
nonetheless informative -- corr(model - ADP, actual - ADP) = +0.256 on 696
drafted player-seasons at p = 6.8e-12. And the disagreement's *magnitude* is
overstated about 2.5-fold: regressing actual-minus-ADP on model-minus-ADP gives a
slope near +0.39.

A points projection is therefore the wrong way to use the disagreement at a draft
table, because the number it produces is systematically too extreme. A
probability is the right shape: not "this player will score 193" but "this player
is 61% to beat his ADP", which is a claim the measured relationship actually
supports.

The gap is scaled by the board's own spread at that rank before it enters the
model. A twenty-point disagreement about the fourth pick and a twenty-point
disagreement about the two-hundredth are not the same claim, and an unscaled gap
would let early picks dominate the fit purely because more points live there.

Deliberately a two-parameter logistic on one scaled input rather than a
classifier over the whole feature set. The signal being encoded is one measured
correlation, and a richer model over the same 700 rows would be fitting the
noise around it. If a feature belongs in the probability, it belongs in the
projection first, where it can be tested properly.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# Below this many rows the fit is not worth attempting; the caller gets the
# base rate instead of a curve through noise.
MIN_ROWS = 120


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))


@dataclass
class AdpEdgeModel:
    """Logistic on the model's scaled disagreement with the board."""

    intercept: float = 0.0
    slope: float = 0.0
    spread: float = 1.0
    base_rate: float = 0.5
    fitted: bool = False
    history: dict = field(default_factory=dict)

    def _scaled(self, gap: np.ndarray) -> np.ndarray:
        return np.asarray(gap, dtype=float) / (self.spread if self.spread > 0 else 1.0)

    def fit(
        self,
        gap: np.ndarray,
        beat: np.ndarray,
        *,
        max_iter: int = 200,
        tolerance: float = 1e-8,
    ) -> "AdpEdgeModel":
        """Newton-Raphson on the two-parameter logistic.

        Hand-rolled because this environment has no scikit-learn, and because a
        two-parameter fit does not justify a dependency.
        """
        gap = np.asarray(gap, dtype=float)
        beat = np.asarray(beat, dtype=float)
        keep = np.isfinite(gap) & np.isfinite(beat)
        gap, beat = gap[keep], beat[keep]
        self.base_rate = float(beat.mean()) if len(beat) else 0.5
        if len(gap) < MIN_ROWS or len(np.unique(beat)) < 2:
            self.fitted = False
            return self
        self.spread = float(np.std(gap)) or 1.0
        x = np.column_stack([np.ones(len(gap)), self._scaled(gap)])
        weights = np.zeros(2)
        for _ in range(max_iter):
            probability = _sigmoid(x @ weights)
            variance = np.clip(probability * (1 - probability), 1e-9, None)
            gradient = x.T @ (beat - probability)
            hessian = (x * variance[:, None]).T @ x + 1e-6 * np.eye(2)
            step = np.linalg.solve(hessian, gradient)
            weights = weights + step
            if np.max(np.abs(step)) < tolerance:
                break
        self.intercept, self.slope = float(weights[0]), float(weights[1])
        self.fitted = True
        return self

    def predict(self, gap: np.ndarray) -> np.ndarray:
        """Probability of finishing above the board's expectation."""
        gap = np.asarray(gap, dtype=float)
        if not self.fitted:
            return np.full(len(gap), self.base_rate)
        out = _sigmoid(self.intercept + self.slope * self._scaled(gap))
        # A gap the caller could not compute is not an opinion; hand back the
        # base rate rather than the value a zero gap happens to imply.
        return np.where(np.isfinite(gap), out, self.base_rate)


def calibration_table(
    probability: np.ndarray, beat: np.ndarray, bins: int = 5
) -> pd.DataFrame:
    """Predicted rate against realised rate, in equal-count buckets."""
    probability = np.asarray(probability, dtype=float)
    beat = np.asarray(beat, dtype=float)
    keep = np.isfinite(probability) & np.isfinite(beat)
    probability, beat = probability[keep], beat[keep]
    if len(probability) < bins * 5:
        return pd.DataFrame()
    edges = np.quantile(probability, np.linspace(0, 1, bins + 1))
    rows = []
    for i in range(bins):
        low, high = edges[i], edges[i + 1]
        mask = (probability >= low) & (
            probability <= high if i == bins - 1 else probability < high
        )
        if mask.sum() < 5:
            continue
        rows.append({
            "bucket": f"{low:.0%}-{high:.0%}",
            "n": int(mask.sum()),
            "predicted": float(probability[mask].mean()),
            "actual": float(beat[mask].mean()),
        })
    return pd.DataFrame(rows)


def brier_score(probability: np.ndarray, beat: np.ndarray) -> float:
    probability = np.asarray(probability, dtype=float)
    beat = np.asarray(beat, dtype=float)
    keep = np.isfinite(probability) & np.isfinite(beat)
    return float(np.mean((probability[keep] - beat[keep]) ** 2))


def auc(probability: np.ndarray, beat: np.ndarray) -> float:
    """Rank-based AUC; ties share their averaged rank."""
    probability = np.asarray(probability, dtype=float)
    beat = np.asarray(beat, dtype=float)
    keep = np.isfinite(probability) & np.isfinite(beat)
    probability, beat = probability[keep], beat[keep]
    positive, negative = beat > 0.5, beat <= 0.5
    if positive.sum() == 0 or negative.sum() == 0:
        return float("nan")
    order = probability.argsort()
    ranks = np.empty(len(probability), dtype=float)
    ranks[order] = np.arange(1, len(probability) + 1)
    # average ranks within ties so a constant prediction scores 0.5, not 1.0
    frame = pd.DataFrame({"p": probability, "r": ranks})
    ranks = frame.groupby("p").r.transform("mean").to_numpy()
    total = ranks[positive].sum()
    n_pos, n_neg = positive.sum(), negative.sum()
    return float((total - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))
