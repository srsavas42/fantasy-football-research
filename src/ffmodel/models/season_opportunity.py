"""Playing-time and per-snap opportunity models for season projections."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ffmodel.features.volume import MODEL_POSITIONS
from ffmodel.models.base import logit, sample_model
from ffmodel.models.volume_team import _sum_to_zero_basis

PLAYER_KEYS = ["season", "team", "player_key"]

SNAP_FEATURES = (
    "prior_snap_share",
    "prior_availability",
    "age",
    "experience",
    "team_change",
    "cold_start",
    "roster_active",
    "roster_reserve",
    "depth_rank",
    "qb_listed_starter",
    "is_replacement_qb",
    "is_replacement_player",
)

QB_PROPENSITY_FEATURES = (
    "prior_qb_attempts_per_snap",
    "prior_qb_snap_share",
    "age",
    "experience",
    "team_change",
    "cold_start",
    "qb_listed_starter",
    "is_replacement_qb",
    "is_replacement_player",
)

CARRY_ELIGIBILITY_FEATURES = (
    "prior_carry_per_snap",
    "prior_carry_role",
    "prior_snap_share",
    "age",
    "experience",
    "team_change",
    "cold_start",
    "depth_rank",
    "is_replacement_qb",
    "is_replacement_player",
)

TARGET_ROLE_FEATURES = (
    "prior_target_per_snap",
    "prior_target_role",
    "prior_snap_share",
    "prior_availability",
    "age",
    "experience",
    "team_change",
    "cold_start",
    "roster_active",
    "roster_reserve",
    "depth_rank",
    "is_replacement_player",
)

SNAP_HISTORY_FEATURES = (
    "prior_snap_share_3yr",
    "prior_snap_share_trend",
    "prior_availability_3yr",
)

CARRY_ELIGIBILITY_EFFICIENCY_FEATURES = ("prior_rush_epa_per_carry",)


def _stack(posterior, name: str) -> np.ndarray:
    return posterior[name].stack(sample=("chain", "draw")).to_numpy()


def _position_effect(pm, name: str, scale: float, size: int):
    raw = pm.Normal(f"{name}_raw", 0.0, scale, shape=size - 1)
    return pm.Deterministic(name, pm.math.dot(_sum_to_zero_basis(size), raw))


@dataclass
class _FeatureModel:
    positions: list[str] = field(default_factory=lambda: list(MODEL_POSITIONS))
    extra_features: tuple[str, ...] = ()
    feature_names: list[str] = field(default_factory=list)
    feature_fill: dict[str, float] = field(default_factory=dict)
    feature_mean: dict[str, float] = field(default_factory=dict)
    feature_scale: dict[str, float] = field(default_factory=dict)
    feature_projection: np.ndarray | None = None
    idata: object = None

    def _candidates(self, base: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(dict.fromkeys((*base, *self.extra_features)))

    def _prepare(self, rows: pd.DataFrame) -> pd.DataFrame:
        required = set(PLAYER_KEYS + ["position"])
        missing = required - set(rows.columns)
        if missing:
            raise ValueError(f"opportunity rows are missing columns: {sorted(missing)}")
        out = rows.copy()
        out["position"] = out["position"].astype(str).str.upper()
        out = out[out["position"].isin(MODEL_POSITIONS)].copy()
        defaults = {
            "prior_snap_share": np.nan,
            "prior_availability": np.nan,
            "prior_qb_attempts_per_snap": np.nan,
            "prior_qb_snap_share": np.nan,
            "prior_carry_per_snap": np.nan,
            "prior_carry_role": np.nan,
            "prior_target_per_snap": np.nan,
            "prior_target_role": np.nan,
            "age": np.nan,
            "experience": np.nan,
            "team_change": 0.0,
            "cold_start": 1.0,
            "roster_active": 1.0,
            "roster_reserve": 0.0,
            "depth_rank": np.nan,
            "qb_listed_starter": 0.0,
            "is_replacement_qb": 0.0,
            "is_replacement_player": 0.0,
        }
        for name, value in defaults.items():
            if name not in out:
                out[name] = value
        return out.sort_values(PLAYER_KEYS).reset_index(drop=True)

    def _matrix(
        self, rows: pd.DataFrame, candidates: tuple[str, ...], *, fit: bool = False
    ) -> np.ndarray:
        if fit:
            self.feature_names = [
                name
                for name in candidates
                if name in rows
                and pd.to_numeric(rows[name], errors="coerce").notna().any()
                and pd.to_numeric(rows[name], errors="coerce")
                .fillna(0)
                .std(ddof=0)
                > 1e-8
            ]
        columns = []
        for name in self.feature_names:
            values = pd.to_numeric(rows[name], errors="coerce")
            if fit:
                fill = float(values.median()) if values.notna().any() else 0.0
                filled = values.fillna(fill)
                scale = float(filled.std(ddof=0))
                self.feature_fill[name] = fill
                self.feature_mean[name] = float(filled.mean())
                self.feature_scale[name] = scale if scale > 1e-8 else 1.0
            filled = values.fillna(self.feature_fill[name])
            columns.append(
                (filled.to_numpy(dtype=float) - self.feature_mean[name])
                / self.feature_scale[name]
            )
        matrix = (
            np.column_stack(columns) if columns else np.zeros((len(rows), 0))
        )
        if fit:
            if matrix.shape[1]:
                _, singular_values, right = np.linalg.svd(matrix, full_matrices=False)
                tolerance = (
                    max(matrix.shape)
                    * np.finfo(float).eps
                    * singular_values.max(initial=0.0)
                )
                rank = int((singular_values > tolerance).sum())
                self.feature_projection = right[:rank].T
            else:
                self.feature_projection = np.zeros((0, 0), dtype=float)
        if self.feature_projection is None:
            return matrix
        return matrix @ np.asarray(self.feature_projection, dtype=float)


@dataclass
class SnapPrediction:
    rows: pd.DataFrame
    conditional_share: np.ndarray
    snap_share: np.ndarray


@dataclass
class SeasonSnapShareModel(_FeatureModel):
    """Conditional offensive-snap share, gated by projected active games."""

    extra_features: tuple[str, ...] = SNAP_HISTORY_FEATURES

    def fit(self, rows: pd.DataFrame, **sample_kwargs) -> "SeasonSnapShareModel":
        import pymc as pm

        out = self._prepare(rows)
        snap_observed = pd.to_numeric(
            out.get("snap_counts_observed", pd.Series(0, index=out.index)),
            errors="coerce",
        ).fillna(0).gt(0)
        snap_share = pd.to_numeric(
            out.get("snap_share", pd.Series(np.nan, index=out.index)),
            errors="coerce",
        )
        availability = pd.to_numeric(
            out.get("observed_availability", pd.Series(np.nan, index=out.index)),
            errors="coerce",
        )
        valid = snap_observed & snap_share.gt(0) & availability.gt(0)
        out = out[valid].reset_index(drop=True)
        response = np.clip(
            snap_share[valid].to_numpy(dtype=float)
            / availability[valid].to_numpy(dtype=float),
            1e-4,
            1.0 - 1e-4,
        )
        if out.empty:
            raise ValueError("snap-share fitting requires observed positive snap counts")
        X = self._matrix(out, self._candidates(SNAP_FEATURES), fit=True)
        position_index = pd.Categorical(out["position"], categories=self.positions).codes
        center = float(logit(np.array([np.clip(response.mean(), 0.03, 0.97)]))[0])
        with pm.Model() as model:
            intercept = pm.Normal("intercept", center, 0.50)
            position_effect = _position_effect(
                pm, "position_effect", 0.40, len(self.positions)
            )
            beta = pm.Normal("beta", 0.0, 0.35, shape=X.shape[1])
            concentration = pm.Gamma("concentration", alpha=3.0, beta=0.08)
            eta = intercept + position_effect[position_index]
            eta = eta + pm.math.sum(X * beta, axis=1)
            mean = pm.math.sigmoid(eta)
            pm.Beta(
                "snap_share_obs",
                alpha=mean * concentration,
                beta=(1.0 - mean) * concentration,
                observed=response,
            )
            sample_kwargs.setdefault("target_accept", 0.92)
            self.idata = sample_model(model, **sample_kwargs)
        return self

    def predict_samples(
        self,
        rows: pd.DataFrame,
        *,
        active_fraction_samples: np.ndarray | None = None,
        seed: int = 0,
    ) -> SnapPrediction:
        if self.idata is None:
            raise RuntimeError("fit the snap-share model before predicting")
        out = self._prepare(rows)
        X = self._matrix(out, self._candidates(SNAP_FEATURES))
        position_index = pd.Categorical(out["position"], categories=self.positions).codes
        post = self.idata.posterior
        eta = _stack(post, "intercept")[None, :]
        eta = eta + _stack(post, "position_effect")[position_index, :]
        eta = eta + X @ _stack(post, "beta")
        mean = 1.0 / (1.0 + np.exp(-np.clip(eta, -20.0, 20.0)))
        concentration = _stack(post, "concentration")[None, :]
        rng = np.random.default_rng(seed)
        conditional = rng.beta(
            np.clip(mean * concentration, 1e-4, None),
            np.clip((1.0 - mean) * concentration, 1e-4, None),
        )
        if active_fraction_samples is None:
            prior = pd.to_numeric(out["prior_availability"], errors="coerce")
            prior = prior.groupby(out["position"]).transform("median").fillna(0.75)
            active_fraction_samples = np.repeat(
                prior.to_numpy(dtype=float)[:, None], conditional.shape[1], axis=1
            )
        active_fraction_samples = np.asarray(active_fraction_samples, dtype=float)
        if active_fraction_samples.shape != conditional.shape:
            raise ValueError("active-fraction samples must align to rows and draws")
        snap_share = np.clip(conditional * active_fraction_samples, 0.0, 1.0)
        return SnapPrediction(out, conditional, snap_share)


@dataclass
class QBPassPropensityPrediction:
    rows: pd.DataFrame
    propensity: np.ndarray


@dataclass
class QBPassPropensityModel(_FeatureModel):
    """Quarterback pass attempts per offensive snap."""

    def fit(self, rows: pd.DataFrame, **sample_kwargs) -> "QBPassPropensityModel":
        import pymc as pm

        out = self._prepare(rows)
        snaps = pd.to_numeric(out.get("offense_snaps"), errors="coerce").fillna(0)
        attempts = pd.to_numeric(out.get("pass_att"), errors="coerce").fillna(0)
        observed = pd.to_numeric(
            out.get("snap_counts_observed", pd.Series(0, index=out.index)),
            errors="coerce",
        ).fillna(0).gt(0)
        valid = out["position"].eq("QB") & observed & snaps.gt(0)
        out = out[valid].reset_index(drop=True)
        n = snaps[valid].round().astype(int).to_numpy()
        y = np.minimum(attempts[valid].round().astype(int).to_numpy(), n)
        if out.empty:
            raise ValueError("QB propensity fitting requires observed quarterback snaps")
        X = self._matrix(out, self._candidates(QB_PROPENSITY_FEATURES), fit=True)
        center = float(logit(np.array([np.clip(y.sum() / n.sum(), 0.05, 0.95)]))[0])
        with pm.Model() as model:
            intercept = pm.Normal("intercept", center, 0.40)
            beta = pm.Normal("beta", 0.0, 0.30, shape=X.shape[1])
            concentration = pm.Gamma("concentration", alpha=4.0, beta=0.08)
            mean = pm.math.sigmoid(intercept + pm.math.sum(X * beta, axis=1))
            pm.BetaBinomial(
                "pass_attempt_obs",
                n=n,
                alpha=mean * concentration,
                beta=(1.0 - mean) * concentration,
                observed=y,
            )
            sample_kwargs.setdefault("target_accept", 0.92)
            self.idata = sample_model(model, **sample_kwargs)
        return self

    def predict_samples(self, rows: pd.DataFrame, *, seed: int = 0) -> QBPassPropensityPrediction:
        if self.idata is None:
            raise RuntimeError("fit the QB propensity model before predicting")
        out = self._prepare(rows)
        X = self._matrix(out, self._candidates(QB_PROPENSITY_FEATURES))
        post = self.idata.posterior
        eta = _stack(post, "intercept")[None, :] + X @ _stack(post, "beta")
        mean = 1.0 / (1.0 + np.exp(-np.clip(eta, -20.0, 20.0)))
        concentration = _stack(post, "concentration")[None, :]
        rng = np.random.default_rng(seed)
        propensity = rng.beta(
            np.clip(mean * concentration, 1e-4, None),
            np.clip((1.0 - mean) * concentration, 1e-4, None),
        )
        propensity[~out["position"].eq("QB").to_numpy()] = 0.0
        return QBPassPropensityPrediction(out, propensity)


@dataclass
class CarryEligibilityPrediction:
    rows: pd.DataFrame
    probability: np.ndarray
    eligible: np.ndarray


@dataclass
class TargetRolePrediction:
    rows: pd.DataFrame
    probability: np.ndarray
    eligible: np.ndarray


@dataclass
class SeasonTargetRoleModel(_FeatureModel):
    """Hurdle for earning at least one target per team game."""

    def fit(self, rows: pd.DataFrame, **sample_kwargs) -> "SeasonTargetRoleModel":
        import pymc as pm

        out = self._prepare(rows)
        receiver = out["position"].isin(("RB", "WR", "TE"))
        out = out[receiver].reset_index(drop=True)
        targets = pd.to_numeric(
            out.get("targets", pd.Series(0, index=out.index)), errors="coerce"
        ).fillna(0)
        team_games = pd.to_numeric(
            out.get("team_games", pd.Series(17, index=out.index)),
            errors="coerce",
        ).fillna(17).clip(lower=1)
        y = targets.div(team_games).ge(1.0).astype(int)
        X = self._matrix(out, self._candidates(TARGET_ROLE_FEATURES), fit=True)
        receiver_positions = ("RB", "WR", "TE")
        position_lookup = {
            position: index for index, position in enumerate(receiver_positions)
        }
        position_index = out["position"].map(position_lookup).to_numpy(dtype=int)
        center = float(logit(np.array([np.clip(y.mean(), 0.03, 0.97)]))[0])
        with pm.Model() as model:
            intercept = pm.Normal("intercept", center, 0.60)
            position_effect = _position_effect(
                pm, "position_effect", 0.55, len(receiver_positions)
            )
            beta = pm.Normal("beta", 0.0, 0.40, shape=X.shape[1])
            eta = intercept + position_effect[position_index]
            eta = eta + pm.math.sum(X * beta, axis=1)
            pm.Bernoulli(
                "target_role_obs", p=pm.math.sigmoid(eta), observed=y.to_numpy()
            )
            sample_kwargs.setdefault("target_accept", 0.92)
            self.idata = sample_model(model, **sample_kwargs)
        return self

    def predict_samples(self, rows: pd.DataFrame, *, seed: int = 0) -> TargetRolePrediction:
        if self.idata is None:
            raise RuntimeError("fit the target-role model before predicting")
        out = self._prepare(rows)
        X = self._matrix(out, self._candidates(TARGET_ROLE_FEATURES))
        receiver_positions = ("RB", "WR", "TE")
        position_lookup = {
            position: index for index, position in enumerate(receiver_positions)
        }
        safe_position_index = (
            out["position"].map(position_lookup).fillna(0).to_numpy(dtype=int)
        )
        post = self.idata.posterior
        eta = _stack(post, "intercept")[None, :]
        eta = eta + _stack(post, "position_effect")[safe_position_index, :]
        eta = eta + X @ _stack(post, "beta")
        probability = 1.0 / (1.0 + np.exp(-np.clip(eta, -20.0, 20.0)))
        receiver = out["position"].isin(("RB", "WR", "TE")).to_numpy()
        probability[~receiver] = 0.0
        eligible = np.random.default_rng(seed).binomial(1, probability).astype(float)
        return TargetRolePrediction(out, probability, eligible)


@dataclass
class SeasonCarryEligibilityModel(_FeatureModel):
    """Hurdle for any rushing attempt, especially sparse WR/TE carries."""

    extra_features: tuple[str, ...] = CARRY_ELIGIBILITY_EFFICIENCY_FEATURES

    def fit(self, rows: pd.DataFrame, **sample_kwargs) -> "SeasonCarryEligibilityModel":
        import pymc as pm

        out = self._prepare(rows)
        y = pd.to_numeric(out.get("rush_att"), errors="coerce").fillna(0).gt(0).astype(int)
        X = self._matrix(out, self._candidates(CARRY_ELIGIBILITY_FEATURES), fit=True)
        position_index = pd.Categorical(out["position"], categories=self.positions).codes
        center = float(logit(np.array([np.clip(y.mean(), 0.03, 0.97)]))[0])
        with pm.Model() as model:
            intercept = pm.Normal("intercept", center, 0.60)
            position_effect = _position_effect(
                pm, "position_effect", 0.65, len(self.positions)
            )
            beta = pm.Normal("beta", 0.0, 0.40, shape=X.shape[1])
            eta = intercept + position_effect[position_index]
            eta = eta + pm.math.sum(X * beta, axis=1)
            pm.Bernoulli("carry_obs", p=pm.math.sigmoid(eta), observed=y.to_numpy())
            sample_kwargs.setdefault("target_accept", 0.92)
            self.idata = sample_model(model, **sample_kwargs)
        return self

    def predict_samples(self, rows: pd.DataFrame, *, seed: int = 0) -> CarryEligibilityPrediction:
        if self.idata is None:
            raise RuntimeError("fit the carry-eligibility model before predicting")
        out = self._prepare(rows)
        X = self._matrix(out, self._candidates(CARRY_ELIGIBILITY_FEATURES))
        position_index = pd.Categorical(out["position"], categories=self.positions).codes
        post = self.idata.posterior
        eta = _stack(post, "intercept")[None, :]
        eta = eta + _stack(post, "position_effect")[position_index, :]
        eta = eta + X @ _stack(post, "beta")
        probability = 1.0 / (1.0 + np.exp(-np.clip(eta, -20.0, 20.0)))
        eligible = np.random.default_rng(seed).binomial(1, probability).astype(float)
        return CarryEligibilityPrediction(out, probability, eligible)
