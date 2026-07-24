"""Point baselines and challenger models for season-average player roles."""

from __future__ import annotations

from dataclasses import dataclass, field
import importlib.util

import numpy as np
import pandas as pd

from ffmodel.features.volume import MODEL_POSITIONS
from ffmodel.models.volume_season_average import (
    BASE_ADJUSTMENT_FEATURES,
    GROUP_KEYS,
    STREAMS,
    volume_adjustment_features,
)


def xgboost_available() -> bool:
    """Whether the optional XGBoost challenger dependency is installed."""
    return importlib.util.find_spec("xgboost") is not None


def persistence_shares(rows: pd.DataFrame, stream: str) -> np.ndarray:
    """Normalized prior/late-season role baseline over each projected roster."""
    role = _role_prior(rows, stream)
    return _softmax_by_group(rows, np.log(role) + np.log(_projected_availability(rows)))


def persistence_volume(rows: pd.DataFrame, team_rows: pd.DataFrame) -> pd.DataFrame:
    """Prior team rates × normalized prior player roles."""
    out = rows.copy().reset_index(drop=True)
    pass_share = persistence_shares(out, "pass")
    target_share = persistence_shares(out, "target")
    carry_share = persistence_shares(out, "carry")
    teams = team_rows.set_index(GROUP_KEYS)
    keys = pd.MultiIndex.from_frame(out[GROUP_KEYS])
    pass_pg = teams["prior_pass_attempts_per_game"].reindex(keys).to_numpy(dtype=float)
    target_pg = teams["prior_targets_per_game"].reindex(keys).to_numpy(dtype=float)
    carry_pg = teams["prior_rush_attempts_per_game"].reindex(keys).to_numpy(dtype=float)
    out["pred_target_share"] = target_share
    out["pred_carry_share"] = carry_share
    out["pred_targets_per_game"] = target_share * target_pg
    out["pred_carries_per_game"] = carry_share * carry_pg
    return out


@dataclass
class RidgeRosterBaseline:
    """Regularized linear residual model with roster-level softmax output."""

    stream: str = "target"
    alpha: float = 10.0
    include_experimental_efficiency: bool = False
    extra_volume_features: tuple[str, ...] | None = None
    feature_names: list[str] = field(default_factory=list)
    fill: dict[str, float] = field(default_factory=dict)
    mean: dict[str, float] = field(default_factory=dict)
    scale: dict[str, float] = field(default_factory=dict)
    coefficients: np.ndarray | None = None

    def fit(self, rows: pd.DataFrame) -> "RidgeRosterBaseline":
        d = rows.copy().reset_index(drop=True)
        candidates = (
            BASE_ADJUSTMENT_FEATURES + self.extra_volume_features
            if self.extra_volume_features is not None
            else volume_adjustment_features(
                self.stream,
                include_experimental=self.include_experimental_efficiency,
            )
        )
        self.feature_names = [name for name in candidates if name in d]
        X = self._matrix(d, fit=True)
        role = _role_prior(d, self.stream)
        observed = _availability_adjusted_share(d, self.stream)
        y = np.log(np.clip(observed, 1e-5, None)) - np.log(role)
        penalty = np.eye(X.shape[1]) * self.alpha
        penalty[0, 0] = 0.0
        self.coefficients = np.linalg.solve(X.T @ X + penalty, X.T @ y)
        return self

    def predict_shares(self, rows: pd.DataFrame) -> np.ndarray:
        if self.coefficients is None:
            raise RuntimeError("fit the ridge baseline before predicting")
        d = rows.copy().reset_index(drop=True)
        score = (
            np.log(_role_prior(d, self.stream))
            + np.log(_projected_availability(d))
            + self._matrix(d) @ self.coefficients
        )
        return _softmax_by_group(d, score)

    def _matrix(self, rows: pd.DataFrame, *, fit: bool = False) -> np.ndarray:
        columns = [np.ones(len(rows), dtype=float)]
        for name in self.feature_names:
            values = pd.to_numeric(rows[name], errors="coerce")
            if fit:
                fill = float(values.median()) if values.notna().any() else 0.0
                filled = values.fillna(fill)
                mean = float(filled.mean())
                scale = float(filled.std(ddof=0))
                self.fill[name] = fill
                self.mean[name] = mean
                self.scale[name] = scale if scale > 1e-8 else 1.0
            filled = values.fillna(self.fill[name])
            columns.append(
                (filled.to_numpy(dtype=float) - self.mean[name]) / self.scale[name]
            )
        for position in MODEL_POSITIONS[:-1]:
            columns.append((rows["position"].astype(str) == position).to_numpy(dtype=float))
        return np.column_stack(columns)


@dataclass
class XGBoostRosterBaseline:
    """Optional nonlinear challenger with the same roster-softmax contract."""

    stream: str = "target"
    params: dict[str, object] = field(default_factory=dict)
    include_experimental_efficiency: bool = False
    extra_volume_features: tuple[str, ...] | None = None
    _ridge_encoder: RidgeRosterBaseline = field(init=False)
    model: object = None

    def fit(self, rows: pd.DataFrame) -> "XGBoostRosterBaseline":
        if not xgboost_available():
            raise RuntimeError(
                "XGBoost is optional; install ffmodel[ml] to run the challenger"
            )
        from xgboost import XGBRegressor

        self._ridge_encoder = RidgeRosterBaseline(
            self.stream,
            include_experimental_efficiency=self.include_experimental_efficiency,
            extra_volume_features=self.extra_volume_features,
        )
        candidates = (
            BASE_ADJUSTMENT_FEATURES + self.extra_volume_features
            if self.extra_volume_features is not None
            else volume_adjustment_features(
                self.stream,
                include_experimental=self.include_experimental_efficiency,
            )
        )
        self._ridge_encoder.feature_names = [name for name in candidates if name in rows]
        X = self._ridge_encoder._matrix(rows.reset_index(drop=True), fit=True)
        role = _role_prior(rows, self.stream)
        observed = _availability_adjusted_share(rows, self.stream)
        y = np.log(np.clip(observed, 1e-5, None)) - np.log(role)
        settings = {
            "n_estimators": 300,
            "max_depth": 3,
            "learning_rate": 0.03,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "objective": "reg:squarederror",
            "n_jobs": 1,
            "random_state": 42,
        }
        settings.update(self.params)
        self.model = XGBRegressor(**settings).fit(X, y)
        return self

    def predict_shares(self, rows: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("fit the XGBoost baseline before predicting")
        d = rows.copy().reset_index(drop=True)
        X = self._ridge_encoder._matrix(d)
        score = (
            np.log(_role_prior(d, self.stream))
            + np.log(_projected_availability(d))
            + self.model.predict(X)
        )
        return _softmax_by_group(d, score)


def _role_prior(rows: pd.DataFrame, stream: str) -> np.ndarray:
    if stream not in STREAMS:
        raise ValueError(f"stream must be one of {sorted(STREAMS)}")
    spec = STREAMS[stream]
    role = pd.to_numeric(rows.get(spec["role"]), errors="coerce")
    draft = pd.to_numeric(rows.get(spec["draft"]), errors="coerce")
    fallback = rows["position"].map(spec["fallback"]).astype(float)
    prior = role.where(role > 0)
    prior = prior.where(prior.notna(), draft.where(draft > 0))
    prior = prior.where(prior.notna(), fallback)
    return np.clip(prior.to_numpy(dtype=float), 1e-5, 1.0)


def _availability_adjusted_share(rows: pd.DataFrame, stream: str) -> np.ndarray:
    count = pd.to_numeric(rows[STREAMS[stream]["count"]], errors="coerce").fillna(0.0)
    availability = pd.to_numeric(
        rows.get("observed_availability", pd.Series(1.0, index=rows.index)),
        errors="coerce",
    ).fillna(1.0)
    rate = count.to_numpy(dtype=float) / np.clip(
        availability.to_numpy(dtype=float), 0.03, 1.0
    )
    total = pd.Series(rate, index=rows.index).groupby(
        [rows[key] for key in GROUP_KEYS]
    ).transform("sum")
    return np.divide(
        rate,
        total.to_numpy(dtype=float),
        out=np.zeros(len(rows), dtype=float),
        where=total.to_numpy(dtype=float) > 0,
    )


def _projected_availability(rows: pd.DataFrame) -> np.ndarray:
    source = rows.get(
        "projected_availability",
        rows.get("prior_availability", pd.Series(np.nan, index=rows.index)),
    )
    values = pd.to_numeric(source, errors="coerce")
    position_fill = values.groupby(rows["position"]).transform("median")
    values = values.fillna(position_fill).fillna(0.75)
    return np.clip(values.to_numpy(dtype=float), 0.03, 1.0)


def _softmax_by_group(rows: pd.DataFrame, score: np.ndarray) -> np.ndarray:
    out = np.zeros(len(rows), dtype=float)
    frame = rows.reset_index(drop=True)
    for _, group in frame.groupby(GROUP_KEYS, sort=False, dropna=False):
        indices = group.index.to_numpy(dtype=int)
        centered = score[indices] - np.max(score[indices])
        weights = np.exp(centered)
        out[indices] = weights / weights.sum()
    return out
    out["pred_pass_attempt_share"] = pass_share
    out["pred_pass_attempts_per_game"] = pass_share * pass_pg
