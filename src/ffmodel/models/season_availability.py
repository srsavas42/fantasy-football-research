"""Preseason availability and quarterback workload models.

All models consume only information available before the projected season.
Availability uses a Beta-Binomial likelihood for games active. Quarterback
workload uses a roster-softmax Multinomial over offensive snaps, so every draw
is a continuous within-team share rather than a starter/backup classification.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ffmodel.features.volume import MODEL_POSITIONS
from ffmodel.models.base import logit, sample_model

GROUP_KEYS = ["season", "team"]
PLAYER_KEYS = GROUP_KEYS + ["player_key"]

AVAILABILITY_FEATURES = (
    "prior_availability",
    "age",
    "experience",
    "team_change",
    "cold_start",
    "roster_active",
    "roster_reserve",
    "depth_rank",
    "qb_listed_starter",
)

STARTER_FEATURES = (
    "prior_availability",
    "age",
    "experience",
    "team_change",
    "cold_start",
    "roster_active",
    "roster_reserve",
    "qb_depth_rank",
    "qb_listed_starter",
)

QB_WORKLOAD_FEATURES = (
    "age",
    "experience",
    "team_change",
    "cold_start",
    "roster_active",
    "qb_depth_rank",
    "qb_listed_starter",
)


def _stack(posterior, name: str) -> np.ndarray:
    return posterior[name].stack(sample=("chain", "draw")).to_numpy()


def _feature_defaults(rows: pd.DataFrame) -> pd.DataFrame:
    out = rows.copy()
    defaults = {
        "prior_availability": np.nan,
        "age": np.nan,
        "experience": np.nan,
        "team_change": 0.0,
        "cold_start": 1.0,
        "roster_active": 1.0,
        "roster_reserve": 0.0,
        "depth_rank": np.nan,
        "qb_depth_rank": np.nan,
        "qb_listed_starter": 0.0,
        "prior_pass_role": np.nan,
        "prior_qb_snap_share": np.nan,
        "draft_pass_prior": 0.0,
    }
    for name, value in defaults.items():
        if name not in out:
            out[name] = value
    return out


@dataclass
class AvailabilityPrediction:
    rows: pd.DataFrame
    probability: np.ndarray
    games_active: np.ndarray
    availability: np.ndarray


@dataclass
class SeasonAvailabilityModel:
    """Beta-Binomial model for a player's active games over a season."""

    positions: list[str] = field(default_factory=lambda: list(MODEL_POSITIONS))
    feature_names: list[str] = field(default_factory=list)
    feature_fill: dict[str, float] = field(default_factory=dict)
    feature_mean: dict[str, float] = field(default_factory=dict)
    feature_scale: dict[str, float] = field(default_factory=dict)
    idata: object = None

    def _prepare(self, rows: pd.DataFrame) -> pd.DataFrame:
        required = set(PLAYER_KEYS + ["position"])
        missing = required - set(rows.columns)
        if missing:
            raise ValueError(
                f"availability rows are missing columns: {sorted(missing)}"
            )
        out = _feature_defaults(rows)
        out["position"] = out["position"].astype(str).str.upper()
        out = out[out["position"].isin(MODEL_POSITIONS)].copy()
        return out.sort_values(PLAYER_KEYS).reset_index(drop=True)

    def _matrix(self, rows: pd.DataFrame, *, fit: bool = False) -> np.ndarray:
        if fit:
            self.feature_names = [
                name
                for name in AVAILABILITY_FEATURES
                if name in rows
                and pd.to_numeric(rows[name], errors="coerce").notna().any()
                and pd.to_numeric(rows[name], errors="coerce").fillna(0).std(ddof=0)
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
        return np.column_stack(columns) if columns else np.zeros((len(rows), 0))

    def fit(self, rows: pd.DataFrame, **sample_kwargs) -> "SeasonAvailabilityModel":
        import pymc as pm

        out = self._prepare(rows)
        if not {"games", "team_games"} <= set(out.columns):
            raise ValueError("availability fitting requires games and team_games")
        games_total = pd.to_numeric(out["team_games"], errors="coerce").fillna(0)
        games_active = pd.to_numeric(out["games"], errors="coerce").fillna(0)
        valid = games_total.gt(0)
        out = out[valid].reset_index(drop=True)
        n = games_total[valid].round().astype(int).to_numpy()
        y = np.minimum(
            games_active[valid].round().astype(int).to_numpy(), n
        )
        X = self._matrix(out, fit=True)
        position_index = pd.Categorical(
            out["position"], categories=self.positions
        ).codes
        center = float(logit(np.array([np.clip(y.sum() / n.sum(), 0.05, 0.98)]))[0])
        with pm.Model() as model:
            intercept = pm.Normal("intercept", center, 0.50)
            position_effect = pm.Normal(
                "position_effect", 0.0, 0.35, shape=len(self.positions)
            )
            beta = pm.Normal("beta", 0.0, 0.35, shape=len(self.feature_names))
            concentration = pm.Gamma("concentration", alpha=3.0, beta=0.12)
            eta = intercept + position_effect[position_index]
            eta = eta + pm.math.sum(X * beta, axis=1)
            probability = pm.math.sigmoid(eta)
            pm.BetaBinomial(
                "games_obs",
                n=n,
                alpha=probability * concentration,
                beta=(1.0 - probability) * concentration,
                observed=y,
            )
            sample_kwargs.setdefault("target_accept", 0.92)
            self.idata = sample_model(model, **sample_kwargs)
        return self

    def predict_samples(
        self, rows: pd.DataFrame, *, team_games=None, seed: int = 0
    ) -> AvailabilityPrediction:
        if self.idata is None:
            raise RuntimeError("fit the availability model before predicting")
        out = self._prepare(rows)
        X = self._matrix(out)
        position_index = pd.Categorical(
            out["position"], categories=self.positions
        ).codes
        post = self.idata.posterior
        eta = _stack(post, "intercept")[None, :]
        eta = eta + _stack(post, "position_effect")[position_index, :]
        eta = eta + X @ _stack(post, "beta")
        mean = 1.0 / (1.0 + np.exp(-np.clip(eta, -20.0, 20.0)))
        concentration = _stack(post, "concentration")[None, :]
        rng = np.random.default_rng(seed)
        probability = rng.beta(
            np.clip(mean * concentration, 1e-4, None),
            np.clip((1.0 - mean) * concentration, 1e-4, None),
        )
        if team_games is None:
            team_games = pd.to_numeric(
                out.get("team_games", pd.Series(17, index=out.index)),
                errors="coerce",
            ).fillna(17).round().astype(int).to_numpy()
        elif np.isscalar(team_games):
            team_games = np.full(len(out), int(team_games), dtype=int)
        else:
            team_games = np.asarray(team_games, dtype=int)
        if team_games.shape != (len(out),) or (team_games <= 0).any():
            raise ValueError("team_games must be positive for every player")
        games_active = rng.binomial(team_games[:, None], probability)
        availability = (games_active + 0.5) / (team_games[:, None] + 1.0)
        return AvailabilityPrediction(
            rows=out,
            probability=probability,
            games_active=games_active,
            availability=availability,
        )


@dataclass
class StarterPrediction:
    rows: pd.DataFrame
    probability: np.ndarray


@dataclass
class QBWorkloadPrediction:
    rows: pd.DataFrame
    group_keys: pd.DataFrame
    shares: np.ndarray


@dataclass
class QBWorkloadShareModel:
    """Continuous season QB offensive-snap share within each team roster."""

    feature_names: list[str] = field(default_factory=list)
    feature_fill: dict[str, float] = field(default_factory=dict)
    feature_mean: dict[str, float] = field(default_factory=dict)
    feature_scale: dict[str, float] = field(default_factory=dict)
    role_innovation_scale: float = 0.60
    idata: object = None

    def _prepare_all(self, rows: pd.DataFrame) -> pd.DataFrame:
        required = set(PLAYER_KEYS + ["position"])
        missing = required - set(rows.columns)
        if missing:
            raise ValueError(f"QB workload rows are missing columns: {sorted(missing)}")
        out = _feature_defaults(rows)
        out["position"] = out["position"].astype(str).str.upper()
        out = out[out["position"].isin(MODEL_POSITIONS)].copy()
        return out.sort_values(PLAYER_KEYS).reset_index(drop=True)

    def _matrix(self, rows: pd.DataFrame, *, fit: bool = False) -> np.ndarray:
        if fit:
            self.feature_names = [
                name
                for name in QB_WORKLOAD_FEATURES
                if name in rows
                and pd.to_numeric(rows[name], errors="coerce").notna().any()
                and pd.to_numeric(rows[name], errors="coerce").fillna(0).std(ddof=0)
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
        return np.column_stack(columns) if columns else np.zeros((len(rows), 0))

    @staticmethod
    def _role_prior(rows: pd.DataFrame) -> np.ndarray:
        snaps = pd.to_numeric(rows["prior_qb_snap_share"], errors="coerce")
        passing = pd.to_numeric(rows["prior_pass_role"], errors="coerce")
        draft = pd.to_numeric(rows["draft_pass_prior"], errors="coerce")
        prior = snaps.where(snaps > 0)
        prior = prior.combine_first(passing.where(passing > 0))
        prior = prior.combine_first(draft.where(draft > 0))
        return np.clip(prior.fillna(0.02).to_numpy(dtype=float), 1e-5, 1.0)

    @staticmethod
    def _observed_counts(quarterbacks: pd.DataFrame) -> np.ndarray:
        snaps = pd.to_numeric(
            quarterbacks.get("offense_snaps", pd.Series(np.nan, index=quarterbacks.index)),
            errors="coerce",
        )
        observed = pd.to_numeric(
            quarterbacks.get(
                "snap_counts_observed", pd.Series(0, index=quarterbacks.index)
            ),
            errors="coerce",
        ).fillna(0).gt(0)
        passing = pd.to_numeric(
            quarterbacks.get("pass_att", pd.Series(0, index=quarterbacks.index)),
            errors="coerce",
        ).fillna(0)
        counts = snaps.where(observed, passing).fillna(0)
        return counts.round().clip(lower=0).to_numpy(dtype=int)

    def _estimate_role_innovation(
        self,
        quarterbacks: pd.DataFrame,
        counts: np.ndarray,
        role: np.ndarray,
    ) -> float:
        availability = pd.to_numeric(
            quarterbacks.get(
                "observed_availability", pd.Series(1.0, index=quarterbacks.index)
            ),
            errors="coerce",
        ).fillna(1.0).clip(0.03, 1.0).to_numpy(dtype=float)
        residuals = []
        for _, group in quarterbacks.groupby(GROUP_KEYS, sort=False, dropna=False):
            indices = group.index.to_numpy(dtype=int)
            if counts[indices].sum() <= 0:
                continue
            expected = role[indices] * availability[indices]
            expected = expected / expected.sum()
            observed = (counts[indices] + 0.5) / (
                counts[indices].sum() + 0.5 * len(indices)
            )
            residual = np.log(observed) - np.log(np.clip(expected, 1e-6, 1.0))
            residuals.extend((residual - residual.mean()).tolist())
        if not residuals:
            return 0.60
        return float(np.clip(np.sqrt(np.mean(np.square(residuals))), 0.10, 2.0))

    def _design(self, rows: pd.DataFrame, *, fit: bool = False):
        all_rows = self._prepare_all(rows)
        quarterbacks = all_rows[all_rows["position"].eq("QB")].copy()
        quarterbacks["_full_index"] = quarterbacks.index
        quarterbacks = quarterbacks.reset_index(drop=True)
        if quarterbacks.empty:
            raise ValueError("QB workload model requires at least one quarterback")
        X = self._matrix(quarterbacks, fit=fit)
        role = self._role_prior(quarterbacks)
        counts_flat = self._observed_counts(quarterbacks)
        if fit:
            self.role_innovation_scale = self._estimate_role_innovation(
                quarterbacks, counts_flat, role
            )
        groups = list(quarterbacks.groupby(GROUP_KEYS, sort=True, dropna=False))
        group_lookup = {tuple(key): index for index, (key, _) in enumerate(groups)}
        all_rows["_group_idx"] = [
            group_lookup.get((season, team), -1)
            for season, team in zip(all_rows["season"], all_rows["team"])
        ]
        slots = max(len(group) for _, group in groups)
        counts = np.zeros((len(groups), slots), dtype=int)
        mask = np.zeros((len(groups), slots), dtype=float)
        role_offset = np.zeros((len(groups), slots), dtype=float)
        availability_offset = np.zeros((len(groups), slots), dtype=float)
        matrix = np.zeros((len(groups), slots, X.shape[1]), dtype=float)
        row_index = np.full((len(groups), slots), -1, dtype=int)
        full_index = np.full((len(groups), slots), -1, dtype=int)
        group_rows = []
        availability = pd.to_numeric(
            quarterbacks.get(
                "observed_availability" if fit else "prior_availability",
                pd.Series(np.nan, index=quarterbacks.index),
            ),
            errors="coerce",
        ).fillna(0.75).clip(0.03, 1.0).to_numpy(dtype=float)
        for group_i, (key, group) in enumerate(groups):
            indices = group.index.to_numpy(dtype=int)
            size = len(indices)
            mask[group_i, :size] = 1.0
            counts[group_i, :size] = counts_flat[indices]
            role_offset[group_i, :size] = np.log(role[indices])
            availability_offset[group_i, :size] = np.log(availability[indices])
            matrix[group_i, :size] = X[indices]
            row_index[group_i, :size] = indices
            full_index[group_i, :size] = quarterbacks.loc[
                indices, "_full_index"
            ].to_numpy(dtype=int)
            group_rows.append(dict(zip(GROUP_KEYS, key)))
        valid = counts.sum(axis=1) > 0 if fit else np.ones(len(groups), dtype=bool)
        return {
            "rows": all_rows,
            "group_keys": pd.DataFrame(group_rows)[valid].reset_index(drop=True),
            "counts": counts[valid],
            "mask": mask[valid],
            "role_offset": role_offset[valid],
            "availability_offset": availability_offset[valid],
            "X": matrix[valid],
            "row_index": row_index[valid],
            "full_index": full_index[valid],
        }

    def fit(self, rows: pd.DataFrame, **sample_kwargs) -> "QBWorkloadShareModel":
        import pymc as pm

        design = self._design(rows, fit=True)
        with pm.Model() as model:
            beta = pm.Normal("beta", 0.0, 0.50, shape=len(self.feature_names))
            eta = design["role_offset"] + design["availability_offset"]
            eta = eta + pm.math.sum(design["X"] * beta, axis=2)
            eta = pm.math.switch(design["mask"] > 0, eta, -20.0)
            probability = pm.math.softmax(eta, axis=1)
            pm.Multinomial(
                "workload_obs",
                n=design["counts"].sum(axis=1),
                p=probability,
                observed=design["counts"],
            )
            sample_kwargs.setdefault("target_accept", 0.92)
            self.idata = sample_model(model, **sample_kwargs)
        return self

    def predict_share_samples(
        self,
        rows: pd.DataFrame,
        *,
        availability_samples: np.ndarray | None = None,
        seed: int = 0,
    ) -> QBWorkloadPrediction:
        if self.idata is None:
            raise RuntimeError("fit the QB workload model before predicting")
        design = self._design(rows)
        beta = _stack(self.idata.posterior, "beta")
        draws = beta.shape[-1]
        eta = design["role_offset"][..., None]
        if availability_samples is None:
            eta = eta + design["availability_offset"][..., None]
        else:
            availability_samples = np.asarray(availability_samples, dtype=float)
            if availability_samples.shape != (len(design["rows"]), draws):
                raise ValueError(
                    "availability samples must align to QB workload roster rows"
                )
            availability = np.full((*design["mask"].shape, draws), 1.0)
            for group_i in range(len(design["group_keys"])):
                active = design["mask"][group_i].astype(bool)
                indices = design["full_index"][group_i, active]
                availability[group_i, active] = availability_samples[indices]
            eta = eta + np.log(np.clip(availability, 0.03, 1.0))
        eta = eta + np.einsum("gkf,fs->gks", design["X"], beta)
        rng = np.random.default_rng(seed)
        eta = eta + rng.normal(size=eta.shape) * self.role_innovation_scale
        eta = np.where(design["mask"][..., None] > 0, eta, -20.0)
        eta -= eta.max(axis=1, keepdims=True)
        probability = np.exp(eta) * design["mask"][..., None]
        probability /= probability.sum(axis=1, keepdims=True)
        shares = np.zeros((len(design["rows"]), draws), dtype=float)
        for group_i in range(len(design["group_keys"])):
            active = design["mask"][group_i].astype(bool)
            indices = design["full_index"][group_i, active]
            shares[indices] = probability[group_i, active]
        return QBWorkloadPrediction(
            rows=design["rows"],
            group_keys=design["group_keys"],
            shares=shares,
        )


@dataclass
class QBStarterModel:
    """Categorical preseason QB1 probability within each team roster."""

    feature_names: list[str] = field(default_factory=list)
    feature_fill: dict[str, float] = field(default_factory=dict)
    feature_mean: dict[str, float] = field(default_factory=dict)
    feature_scale: dict[str, float] = field(default_factory=dict)
    idata: object = None

    def _prepare_all(self, rows: pd.DataFrame) -> pd.DataFrame:
        required = set(PLAYER_KEYS + ["position"])
        missing = required - set(rows.columns)
        if missing:
            raise ValueError(f"starter rows are missing columns: {sorted(missing)}")
        out = _feature_defaults(rows)
        out["position"] = out["position"].astype(str).str.upper()
        out = out[out["position"].isin(MODEL_POSITIONS)].copy()
        return out.sort_values(PLAYER_KEYS).reset_index(drop=True)

    def _matrix(self, rows: pd.DataFrame, *, fit: bool = False) -> np.ndarray:
        if fit:
            self.feature_names = [
                name
                for name in STARTER_FEATURES
                if name in rows
                and pd.to_numeric(rows[name], errors="coerce").notna().any()
                and pd.to_numeric(rows[name], errors="coerce").fillna(0).std(ddof=0)
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
        return np.column_stack(columns) if columns else np.zeros((len(rows), 0))

    @staticmethod
    def _role_prior(rows: pd.DataFrame) -> np.ndarray:
        prior = pd.to_numeric(rows["prior_pass_role"], errors="coerce")
        draft = pd.to_numeric(rows["draft_pass_prior"], errors="coerce")
        prior = prior.where(prior > 0).combine_first(draft.where(draft > 0))
        return np.clip(prior.fillna(0.02).to_numpy(dtype=float), 1e-4, 1.0)

    def _design(self, quarterbacks: pd.DataFrame, *, fit: bool = False):
        quarterbacks = quarterbacks.reset_index(drop=True)
        X = self._matrix(quarterbacks, fit=fit)
        groups = list(quarterbacks.groupby(GROUP_KEYS, sort=True, dropna=False))
        if not groups:
            raise ValueError("starter model requires at least one quarterback")
        slots = max(len(group) for _, group in groups)
        counts = np.zeros((len(groups), slots), dtype=int)
        mask = np.zeros((len(groups), slots), dtype=float)
        offset = np.zeros((len(groups), slots), dtype=float)
        matrix = np.zeros((len(groups), slots, X.shape[1]), dtype=float)
        row_index = np.full((len(groups), slots), -1, dtype=int)
        group_rows = []
        role = self._role_prior(quarterbacks)
        for group_i, (key, group) in enumerate(groups):
            indices = group.index.to_numpy(dtype=int)
            size = len(indices)
            mask[group_i, :size] = 1.0
            offset[group_i, :size] = np.log(role[indices])
            matrix[group_i, :size] = X[indices]
            row_index[group_i, :size] = indices
            if fit:
                observed = pd.to_numeric(
                    quarterbacks.loc[indices, "primary_qb"], errors="coerce"
                ).fillna(0).to_numpy(dtype=int)
                if observed.sum() != 1:
                    observed = np.zeros(size, dtype=int)
                    observed[np.argmax(role[indices])] = 1
                counts[group_i, :size] = observed
            group_rows.append(dict(zip(GROUP_KEYS, key)))
        return {
            "X": matrix,
            "mask": mask,
            "offset": offset,
            "counts": counts,
            "row_index": row_index,
            "group_keys": pd.DataFrame(group_rows),
        }

    def fit(self, rows: pd.DataFrame, **sample_kwargs) -> "QBStarterModel":
        import pymc as pm

        all_rows = self._prepare_all(rows)
        quarterbacks = all_rows[all_rows["position"].eq("QB")].reset_index(drop=True)
        if "primary_qb" not in quarterbacks:
            raise ValueError("starter fitting requires primary_qb labels")
        design = self._design(quarterbacks, fit=True)
        with pm.Model() as model:
            beta = pm.Normal("beta", 0.0, 0.60, shape=len(self.feature_names))
            eta = design["offset"] + pm.math.sum(design["X"] * beta, axis=2)
            eta = pm.math.switch(design["mask"] > 0, eta, -20.0)
            probability = pm.math.softmax(eta, axis=1)
            pm.Multinomial(
                "starter_obs", n=1, p=probability, observed=design["counts"]
            )
            sample_kwargs.setdefault("target_accept", 0.92)
            self.idata = sample_model(model, **sample_kwargs)
        return self

    def predict_samples(self, rows: pd.DataFrame) -> StarterPrediction:
        if self.idata is None:
            raise RuntimeError("fit the starter model before predicting")
        all_rows = self._prepare_all(rows)
        quarterbacks = all_rows[all_rows["position"].eq("QB")].copy()
        quarterbacks["_full_index"] = quarterbacks.index
        quarterbacks = quarterbacks.reset_index(drop=True)
        design = self._design(quarterbacks)
        beta = _stack(self.idata.posterior, "beta")
        eta = design["offset"][..., None]
        eta = eta + np.einsum("gkf,fs->gks", design["X"], beta)
        eta = np.where(design["mask"][..., None] > 0, eta, -20.0)
        eta -= eta.max(axis=1, keepdims=True)
        group_probability = np.exp(eta) * design["mask"][..., None]
        group_probability /= group_probability.sum(axis=1, keepdims=True)
        probability = np.zeros((len(all_rows), beta.shape[-1]), dtype=float)
        for group_i in range(len(design["group_keys"])):
            active = design["mask"][group_i].astype(bool)
            qb_indices = design["row_index"][group_i, active]
            full_indices = quarterbacks.loc[qb_indices, "_full_index"].to_numpy(dtype=int)
            probability[full_indices] = group_probability[group_i, active]
        return StarterPrediction(rows=all_rows, probability=probability)
