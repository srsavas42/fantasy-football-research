"""Leakage-safe player-season role regime labels and probabilities.

The realized regime is a *training label*: it summarizes whether a player was a
replacement, inactive, part of a committee, or a lead option that season.  At
projection time the model only uses information available before Week 1 and
returns a probability distribution over the same states.  This makes it a
useful common driver for subsequent role-only, efficiency-only, and joint
ablations without feeding realized usage back into the prediction features.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


REGIME_NAMES = ("replacement", "inactive", "committee", "lead")
LEARNED_REGIME_NAMES = REGIME_NAMES[1:]
REGIME_LIKELIHOOD_FEATURES = tuple(
    f"regime_probability_{name}" for name in LEARNED_REGIME_NAMES
)

# These are deliberately a conservative subset of the preseason contract.  A
# regime screen should be easy to audit before it is allowed to mediate both
# volume and efficiency.
REGIME_NUMERIC_FEATURES = (
    "prior_pass_role",
    "prior_target_role",
    "prior_carry_role",
    "prior_availability",
    "prior_snap_share",
    "prior_qb_snap_share",
    "prior_target_per_snap",
    "prior_carry_per_snap",
    "prior_qb_attempts_per_snap",
    "age",
    "experience",
    "team_change",
    "cold_start",
    "roster_active",
    "roster_reserve",
    "depth_rank",
    "qb_depth_rank",
    "qb_listed_starter",
    "draft_target_prior",
    "draft_carry_prior",
    "draft_pass_prior",
    "prior_role_continuity",
)


@dataclass
class RegimeThresholds:
    """Training-fold thresholds that define an interpretable lead role."""

    lead_role_threshold: dict[str, float]
    inactive_availability: float = 0.25
    inactive_role: float = 0.02


@dataclass
class SeasonRegimePrediction:
    rows: pd.DataFrame
    probability: np.ndarray
    samples: np.ndarray

    @property
    def most_likely(self) -> np.ndarray:
        return np.asarray(REGIME_NAMES, dtype=object)[self.probability.argmax(axis=1)]


def _number(rows: pd.DataFrame, name: str, default: float = 0.0) -> np.ndarray:
    value = rows.get(name)
    if value is None:
        return np.full(len(rows), default, dtype=float)
    return pd.to_numeric(value, errors="coerce").fillna(default).to_numpy(float)


def _role_metric(rows: pd.DataFrame) -> np.ndarray:
    """Position-specific realized role metric, used only to construct labels."""

    position = rows.get("position", pd.Series("", index=rows.index)).fillna("")
    snap = _number(rows, "snap_share")
    qb_workload = _number(rows, "observed_qb_workload_share")
    targets = _number(rows, "target_share")
    carries = _number(rows, "carry_share")
    out = np.zeros(len(rows), dtype=float)
    qb = position.eq("QB").to_numpy()
    rb = position.eq("RB").to_numpy()
    wr = position.eq("WR").to_numpy()
    te = position.eq("TE").to_numpy()
    out[qb] = qb_workload[qb]
    # Scale opportunity components onto the same rough range as snap share,
    # then retain a transparent blend of playing time and direct role.
    out[rb] = 0.5 * snap[rb] + 0.5 * np.minimum(1.0, 2.0 * carries[rb])
    out[wr] = 0.6 * snap[wr] + 0.4 * np.minimum(1.0, 4.0 * targets[wr])
    out[te] = 0.65 * snap[te] + 0.35 * np.minimum(1.0, 5.0 * targets[te])
    other = ~(qb | rb | wr | te)
    out[other] = snap[other]
    return np.clip(out, 0.0, 1.0)


def fit_regime_thresholds(rows: pd.DataFrame) -> RegimeThresholds:
    """Fit label thresholds exclusively on a training fold's realized usage."""

    replacement = _number(rows, "is_replacement_player").astype(bool)
    availability = _number(rows, "observed_availability")
    role = _role_metric(rows)
    threshold: dict[str, float] = {}
    position = rows.get("position", pd.Series("", index=rows.index)).fillna("")
    eligible = (~replacement) & (availability >= 0.25) & (role >= 0.02)
    for value in ("QB", "RB", "WR", "TE"):
        values = role[(position.eq(value).to_numpy()) & eligible]
        threshold[value] = float(np.quantile(values, 0.75)) if len(values) else 1.0
    return RegimeThresholds(threshold)


def realized_regimes(rows: pd.DataFrame, thresholds: RegimeThresholds) -> np.ndarray:
    """Return realized labels.  Never call this on future rows as a feature."""

    replacement = _number(rows, "is_replacement_player").astype(bool)
    availability = _number(rows, "observed_availability")
    role = _role_metric(rows)
    position = rows.get("position", pd.Series("", index=rows.index)).fillna("")
    labels = np.full(len(rows), "committee", dtype=object)
    labels[replacement] = "replacement"
    inactive = ~replacement & (
        (availability < thresholds.inactive_availability) | (role < thresholds.inactive_role)
    )
    labels[inactive] = "inactive"
    lead_cutoff = np.array(
        [thresholds.lead_role_threshold.get(value, 1.0) for value in position], dtype=float
    )
    lead = ~replacement & ~inactive & (role >= lead_cutoff)
    labels[lead] = "lead"
    return labels


def add_walk_forward_regime_probabilities(
    rows: pd.DataFrame, *, classifier_steps: int = 500
) -> pd.DataFrame:
    """Attach chronologically out-of-fold regime probabilities to training rows.

    For response season ``Y``, the classifier is fit only on seasons before
    ``Y``. These columns can therefore enter a downstream volume likelihood
    without exposing the realized regime label of the response season.
    """

    if "season" not in rows:
        raise ValueError("regime feature rows require a season column")
    out = rows.copy().reset_index(drop=True)
    probability = np.zeros((len(out), len(LEARNED_REGIME_NAMES)), dtype=float)
    seasons = sorted(pd.to_numeric(out["season"], errors="coerce").dropna().unique())
    fallback = np.array([0.25, 0.60, 0.15], dtype=float)
    for season in seasons:
        current = out["season"].eq(season).to_numpy()
        history = out.loc[out["season"].lt(season)].reset_index(drop=True)
        if history.empty:
            probability[current] = fallback
            continue
        model = SeasonRegimeModel(steps=classifier_steps).fit(history)
        probability[current] = model.predict_proba(out.loc[current])[:, 1:]
    for index, name in enumerate(REGIME_LIKELIHOOD_FEATURES):
        out[name] = probability[:, index]
    return out


def add_regime_probabilities(
    rows: pd.DataFrame, model: "SeasonRegimeModel"
) -> pd.DataFrame:
    """Attach future-season regime probabilities from a fitted classifier."""

    out = rows.copy().reset_index(drop=True)
    probability = model.predict_proba(out)[:, 1:]
    for index, name in enumerate(REGIME_LIKELIHOOD_FEATURES):
        out[name] = probability[:, index]
    return out


@dataclass
class SeasonRegimeModel:
    """Regularized multinomial pre-season classifier for player-season regimes."""

    l2: float = 2.0
    steps: int = 2_000
    learning_rate: float = 0.15
    feature_names: tuple[str, ...] = field(default_factory=tuple, init=False)
    fill: np.ndarray | None = field(default=None, init=False)
    mean: np.ndarray | None = field(default=None, init=False)
    scale: np.ndarray | None = field(default=None, init=False)
    coefficients: np.ndarray | None = field(default=None, init=False)
    thresholds: RegimeThresholds | None = field(default=None, init=False)

    def _matrix(self, rows: pd.DataFrame, *, fit: bool) -> np.ndarray:
        numeric = []
        names: list[str] = []
        for name in REGIME_NUMERIC_FEATURES:
            value = rows.get(name)
            if value is None:
                value = np.full(len(rows), np.nan, dtype=float)
            else:
                value = pd.to_numeric(value, errors="coerce").to_numpy(float)
            numeric.append(value)
            names.extend((name, f"{name}_missing"))
        raw = np.column_stack(numeric)
        if fit:
            self.fill = pd.DataFrame(raw).median(axis=0).fillna(0.0).to_numpy(float)
        if self.fill is None:
            raise RuntimeError("fit the regime model before predicting")
        missing = ~np.isfinite(raw)
        raw = np.where(missing, self.fill, raw)
        expanded = np.empty((len(rows), raw.shape[1] * 2), dtype=float)
        expanded[:, 0::2] = raw
        expanded[:, 1::2] = missing.astype(float)
        position = rows.get("position", pd.Series("", index=rows.index)).fillna("")
        categories = ("QB", "RB", "WR", "TE")
        one_hot = np.column_stack([position.eq(value).to_numpy(float) for value in categories])
        matrix = np.column_stack((expanded, one_hot))
        if fit:
            self.mean = matrix.mean(axis=0)
            self.scale = matrix.std(axis=0)
            self.scale = np.where(self.scale > 1e-8, self.scale, 1.0)
            self.feature_names = tuple(names + [f"position_{value}" for value in categories])
        if self.mean is None or self.scale is None:
            raise RuntimeError("fit the regime model before predicting")
        return (matrix - self.mean) / self.scale

    def fit(self, rows: pd.DataFrame) -> "SeasonRegimeModel":
        self.thresholds = fit_regime_thresholds(rows)
        labels = realized_regimes(rows, self.thresholds)
        learn = labels != "replacement"
        x = self._matrix(rows.loc[learn].reset_index(drop=True), fit=True)
        y = np.array([LEARNED_REGIME_NAMES.index(value) for value in labels[learn]], dtype=int)
        weights = np.zeros((x.shape[1] + 1, len(LEARNED_REGIME_NAMES)), dtype=float)
        design = np.column_stack((np.ones(len(x)), x))
        for _ in range(self.steps):
            logits = design @ weights
            logits -= logits.max(axis=1, keepdims=True)
            probability = np.exp(logits)
            probability /= probability.sum(axis=1, keepdims=True)
            target = np.zeros_like(probability)
            target[np.arange(len(y)), y] = 1.0
            # Keep the likelihood unweighted: the output is consumed as a
            # probability distribution, so calibration matters more than
            # balancing hard class predictions.
            residual = probability - target
            gradient = design.T @ residual / len(y)
            gradient[1:] += self.l2 * weights[1:] / len(y)
            weights -= self.learning_rate * gradient
        self.coefficients = weights
        return self

    def predict_proba(self, rows: pd.DataFrame) -> np.ndarray:
        if self.coefficients is None:
            raise RuntimeError("fit the regime model before predicting")
        x = self._matrix(rows.reset_index(drop=True), fit=False)
        logits = np.column_stack((np.ones(len(x)), x)) @ self.coefficients
        logits -= logits.max(axis=1, keepdims=True)
        learned = np.exp(logits)
        learned /= learned.sum(axis=1, keepdims=True)
        probability = np.zeros((len(rows), len(REGIME_NAMES)), dtype=float)
        replacement = _number(rows, "is_replacement_player").astype(bool)
        probability[~replacement, 1:] = learned[~replacement]
        probability[replacement, 0] = 1.0
        return probability

    def predict_samples(
        self, rows: pd.DataFrame, *, draws: int = 200, seed: int = 0
    ) -> SeasonRegimePrediction:
        probability = self.predict_proba(rows)
        rng = np.random.default_rng(seed)
        uniforms = rng.random((len(rows), draws))
        samples = (uniforms[..., None] > np.cumsum(probability, axis=1)[:, None, :]).sum(axis=2)
        samples = np.minimum(samples, len(REGIME_NAMES) - 1).astype(int)
        return SeasonRegimePrediction(rows.reset_index(drop=True), probability, samples)

    def state_dict(self) -> dict[str, object]:
        """Return JSON-safe state for a fitted, prediction-only regime model."""

        if any(
            value is None
            for value in (self.fill, self.mean, self.scale, self.coefficients, self.thresholds)
        ):
            raise RuntimeError("fit the regime model before serializing it")
        assert self.thresholds is not None
        return {
            "l2": self.l2,
            "steps": self.steps,
            "learning_rate": self.learning_rate,
            "feature_names": list(self.feature_names),
            "fill": self.fill.tolist(),
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "coefficients": self.coefficients.tolist(),
            "thresholds": {
                "lead_role_threshold": self.thresholds.lead_role_threshold,
                "inactive_availability": self.thresholds.inactive_availability,
                "inactive_role": self.thresholds.inactive_role,
            },
        }

    @classmethod
    def from_state(cls, state: dict[str, object]) -> "SeasonRegimeModel":
        """Restore a prediction-only model from :meth:`state_dict`."""

        model = cls(
            l2=float(state["l2"]),
            steps=int(state["steps"]),
            learning_rate=float(state["learning_rate"]),
        )
        model.feature_names = tuple(state["feature_names"])
        model.fill = np.asarray(state["fill"], dtype=float)
        model.mean = np.asarray(state["mean"], dtype=float)
        model.scale = np.asarray(state["scale"], dtype=float)
        model.coefficients = np.asarray(state["coefficients"], dtype=float)
        threshold = state["thresholds"]
        model.thresholds = RegimeThresholds(
            lead_role_threshold={
                str(name): float(value)
                for name, value in threshold["lead_role_threshold"].items()
            },
            inactive_availability=float(threshold["inactive_availability"]),
            inactive_role=float(threshold["inactive_role"]),
        )
        return model
