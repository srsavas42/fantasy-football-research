"""Exposure-aware season-average efficiency models.

The models predict latent per-opportunity efficiency from lagged, partially
pooled efficiency; preseason context; and leak-free volume projections. The
ridge stack remains the fast walk-forward point benchmark. The posterior stack
uses Beta-Binomial likelihoods for rates and exposure-scaled Student-t
likelihoods for yardage, giving the scoring simulator calibrated season-level
draws rather than treating a point estimate as certain.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from ffmodel.models.base import (
    load_idata,
    logit,
    sample_model,
    sampling_quality,
    save_idata,
)
from ffmodel.models.volume_team import _sum_to_zero_basis


@dataclass(frozen=True)
class EfficiencyModelSpec:
    target: str
    prior_feature: str
    exposure: str
    prior_exposure: str
    volume_feature: str
    positions: tuple[str, ...]
    transform: str = "identity"
    lower: float = 0.0
    upper: float = 1.0
    min_exposure: int = 1
    advanced_features: tuple[str, ...] = ()
    likelihood: str = "continuous"
    numerator: str | None = None
    prior_concentration: float = 100.0


RECEIVING_POSITIONS = ("RB", "WR", "TE")
RUSHING_POSITIONS = ("QB", "RB", "WR", "TE")

EFFICIENCY_MODEL_SPECS = (
    EfficiencyModelSpec(
        "pass_completion_rate",
        "prior_pass_completion_rate",
        "pass_att",
        "prior_pass_att",
        "oof_pass_attempts_per_team_game",
        ("QB",),
        "logit",
        0.0,
        1.0,
        50,
        (
            "prior_pass_epa_per_attempt",
            "prior_pass_air_yards_per_attempt",
            "prior_pass_yac_per_completion",
            "prior_pass_first_down_rate",
        ),
        likelihood="beta_binomial",
        numerator="eff_pass_cmp",
        prior_concentration=150.0,
    ),
    EfficiencyModelSpec(
        "pass_yards_per_attempt",
        "prior_pass_yards_per_attempt",
        "pass_att",
        "prior_pass_att",
        "oof_pass_attempts_per_team_game",
        ("QB",),
        "identity",
        0.0,
        15.0,
        50,
        (
            "prior_pass_epa_per_attempt",
            "prior_pass_air_yards_per_attempt",
            "prior_pass_yac_per_completion",
            "prior_pass_first_down_rate",
        ),
        numerator="eff_pass_yds",
    ),
    EfficiencyModelSpec(
        "pass_td_rate",
        "prior_pass_td_rate",
        "pass_att",
        "prior_pass_att",
        "oof_pass_attempts_per_team_game",
        ("QB",),
        "logit",
        0.0,
        1.0,
        50,
        ("prior_pass_epa_per_attempt", "prior_pass_first_down_rate"),
        likelihood="beta_binomial",
        numerator="eff_pass_td",
        prior_concentration=200.0,
    ),
    EfficiencyModelSpec(
        "pass_int_rate",
        "prior_pass_int_rate",
        "pass_att",
        "prior_pass_att",
        "oof_pass_attempts_per_team_game",
        ("QB",),
        "logit",
        0.0,
        1.0,
        50,
        ("prior_pass_epa_per_attempt", "prior_pass_completion_rate"),
        likelihood="beta_binomial",
        numerator="eff_pass_int",
        prior_concentration=200.0,
    ),
    EfficiencyModelSpec(
        "rec_catch_rate",
        "prior_rec_catch_rate",
        "targets",
        "prior_targets",
        "oof_targets_per_team_game",
        RECEIVING_POSITIONS,
        "logit",
        0.0,
        1.0,
        20,
        (
            "prior_rec_epa_per_target",
            "prior_rec_air_yards_per_target",
            "prior_rec_yac_per_reception",
            "prior_rec_first_down_rate",
            "prior_targets_per_pass_play",
        ),
        likelihood="beta_binomial",
        numerator="eff_receptions",
        prior_concentration=60.0,
    ),
    EfficiencyModelSpec(
        "rec_yards_per_target",
        "prior_rec_yards_per_target",
        "targets",
        "prior_targets",
        "oof_targets_per_team_game",
        RECEIVING_POSITIONS,
        "identity",
        0.0,
        20.0,
        20,
        (
            "prior_rec_epa_per_target",
            "prior_rec_air_yards_per_target",
            "prior_rec_yac_per_reception",
            "prior_rec_first_down_rate",
            "prior_targets_per_pass_play",
        ),
        numerator="eff_rec_yds",
    ),
    EfficiencyModelSpec(
        "rec_td_rate",
        "prior_rec_td_rate",
        "targets",
        "prior_targets",
        "oof_targets_per_team_game",
        RECEIVING_POSITIONS,
        "logit",
        0.0,
        1.0,
        20,
        (
            "prior_rec_epa_per_target",
            "prior_rec_first_down_rate",
            "prior_targets_per_pass_play",
        ),
        likelihood="beta_binomial",
        numerator="eff_rec_td",
        prior_concentration=120.0,
    ),
    EfficiencyModelSpec(
        "rush_yards_per_carry",
        "prior_rush_yards_per_carry",
        "rush_att",
        "prior_rush_att",
        "oof_carries_per_team_game",
        RUSHING_POSITIONS,
        "identity",
        -2.0,
        15.0,
        20,
        ("prior_rush_epa_per_carry", "prior_rush_first_down_rate"),
        numerator="eff_rush_yds",
    ),
    EfficiencyModelSpec(
        "rush_td_rate",
        "prior_rush_td_rate",
        "rush_att",
        "prior_rush_att",
        "oof_carries_per_team_game",
        RUSHING_POSITIONS,
        "logit",
        0.0,
        1.0,
        20,
        ("prior_rush_epa_per_carry", "prior_rush_first_down_rate"),
        likelihood="beta_binomial",
        numerator="eff_rush_td",
        prior_concentration=120.0,
    ),
    EfficiencyModelSpec(
        "fumble_lost_rate",
        "prior_fumble_lost_rate",
        "fumble_opportunities",
        "prior_fumble_opportunities",
        "oof_fumble_opportunities_per_team_game",
        RUSHING_POSITIONS,
        "logit",
        0.0,
        1.0,
        20,
        (),
        likelihood="beta_binomial",
        numerator="eff_fumbles_lost",
        prior_concentration=250.0,
    ),
)

EFFICIENCY_MODEL_BY_TARGET = {spec.target: spec for spec in EFFICIENCY_MODEL_SPECS}

# Each response draws from its own seed, offset from the caller's. Deriving that
# offset from the response's position in ``self.models`` made it depend on
# insertion order, and insertion order is not stable across a save/load round
# trip: ``fit`` inserts in spec order while ``load`` reads a JSON object written
# with ``sort_keys=True`` and inserts alphabetically. A reloaded artifact
# therefore handed every response a different seed and produced a different
# realization from the pipeline it was saved from — 302 PPR points at the
# maximum, distributionally identical and reproducibly wrong. Key the offset to
# the response itself instead, which no ordering can disturb.
EFFICIENCY_SEED_OFFSET = {
    spec.target: index for index, spec in enumerate(EFFICIENCY_MODEL_SPECS)
}

BASE_EFFICIENCY_FEATURES = (
    "prior_availability",
    "prior_snap_share",
    "age",
    "experience",
    "team_change",
    "cold_start",
)

# Production mean gate after the 2022-2024 posterior screen. The posterior
# challenger improved receiving yards/target by 1.22% with two fold wins. Every
# other flexible mean failed either pooled accuracy or the two-of-three
# stability rule, so its accepted v1 point forecast remains the conditional
# mean while the likelihood adds calibrated future-season uncertainty.
POSTERIOR_MEAN_MODE = {
    "pass_completion_rate": "ridge",
    "pass_yards_per_attempt": "ridge",
    "pass_td_rate": "ridge",
    "pass_int_rate": "ridge",
    "rec_catch_rate": "prior",
    "rec_yards_per_target": "posterior",
    "rec_td_rate": "prior",
    "rush_yards_per_carry": "ridge",
    "rush_td_rate": "prior",
    "fumble_lost_rate": "prior",
}


@dataclass
class ExposureWeightedEfficiencyModel:
    """Weighted ridge regression for one transformed efficiency response."""

    spec: EfficiencyModelSpec
    alpha: float = 20.0
    use_volume: bool = True
    use_advanced: bool = True
    feature_names: list[str] = field(default_factory=list)
    fill: dict[str, float] = field(default_factory=dict)
    mean: dict[str, float] = field(default_factory=dict)
    scale: dict[str, float] = field(default_factory=dict)
    coefficients: np.ndarray | None = None
    training_rows: int = 0

    def fit(self, rows: pd.DataFrame) -> "ExposureWeightedEfficiencyModel":
        d = self._eligible(rows, require_target=True)
        if d.empty:
            raise ValueError(f"no eligible training rows for {self.spec.target}")
        self.feature_names = [self.spec.prior_feature, self.spec.prior_exposure]
        self.feature_names.extend(
            feature for feature in BASE_EFFICIENCY_FEATURES if feature in d
        )
        if self.use_volume and self.spec.volume_feature in d:
            self.feature_names.append(self.spec.volume_feature)
        if self.use_advanced:
            self.feature_names.extend(
                feature
                for feature in self.spec.advanced_features
                if feature in d and feature != self.spec.prior_feature
            )
        self.feature_names = list(dict.fromkeys(self.feature_names))

        X = self._matrix(d, fit=True)
        y = self._forward(d[self.spec.target].to_numpy(dtype=float))
        exposure = pd.to_numeric(d[self.spec.exposure], errors="coerce").to_numpy(
            dtype=float
        )
        median = max(float(np.median(exposure)), 1.0)
        weights = np.clip(exposure / median, 0.25, 5.0)
        root_weight = np.sqrt(weights)
        Xw = X * root_weight[:, None]
        yw = y * root_weight
        penalty = np.eye(X.shape[1]) * self.alpha
        penalty[0, 0] = 0.0
        self.coefficients = np.linalg.solve(Xw.T @ Xw + penalty, Xw.T @ yw)
        self.training_rows = len(d)
        return self

    def predict(self, rows: pd.DataFrame) -> np.ndarray:
        if self.coefficients is None:
            raise RuntimeError("fit the efficiency model before predicting")
        d = rows.copy().reset_index(drop=True)
        prediction = self._inverse(self._matrix(d) @ self.coefficients)
        valid_position = d["position"].astype(str).isin(self.spec.positions).to_numpy()
        return np.where(valid_position, prediction, np.nan)

    def predict_volume_conditioned_samples(
        self,
        rows: pd.DataFrame,
        volume_feature_samples: np.ndarray,
    ) -> np.ndarray:
        """Evaluate a ridge mean at aligned simulated volume draws.

        Training uses an out-of-fold mean volume feature. At scoring time the
        same fitted relationship can be evaluated at each coherent volume draw,
        creating a directed volume-to-efficiency dependency without changing
        the fitted marginal model when no volume feature is present.
        """
        if self.coefficients is None:
            raise RuntimeError("fit the efficiency model before predicting")
        samples = np.asarray(volume_feature_samples, dtype=float)
        d = rows.copy().reset_index(drop=True)
        if samples.ndim != 2 or samples.shape[0] != len(d):
            raise ValueError(
                f"{self.spec.target} volume-feature samples must have shape "
                "(rows, draws)"
            )
        if self.spec.volume_feature not in self.feature_names:
            return np.repeat(self.predict(d)[:, None], samples.shape[1], axis=1)

        matrix = self._matrix(d)
        feature_index = self.feature_names.index(self.spec.volume_feature)
        matrix_column = 1 + 2 * feature_index
        values = np.log1p(np.clip(samples, 0.0, None))
        values = np.where(
            np.isfinite(values), values, self.fill[self.spec.volume_feature]
        )
        scaled = (values - self.mean[self.spec.volume_feature]) / self.scale[
            self.spec.volume_feature
        ]
        linear = matrix @ self.coefficients
        linear = linear[:, None] + (
            scaled - matrix[:, matrix_column, None]
        ) * self.coefficients[matrix_column]
        prediction = self._inverse(linear)
        valid_position = d["position"].astype(str).isin(self.spec.positions).to_numpy()
        return np.where(valid_position[:, None], prediction, np.nan)

    def _eligible(self, rows: pd.DataFrame, *, require_target: bool) -> pd.DataFrame:
        d = rows.copy().reset_index(drop=True)
        mask = d["position"].astype(str).isin(self.spec.positions)
        exposure = pd.to_numeric(d.get(self.spec.exposure), errors="coerce")
        mask &= exposure.ge(self.spec.min_exposure)
        if require_target:
            target = pd.to_numeric(d.get(self.spec.target), errors="coerce")
            mask &= target.notna() & np.isfinite(target)
        return d[mask].reset_index(drop=True)

    def _matrix(self, rows: pd.DataFrame, *, fit: bool = False) -> np.ndarray:
        columns = [np.ones(len(rows), dtype=float)]
        for name in self.feature_names:
            values = pd.to_numeric(
                rows.get(name, pd.Series(np.nan, index=rows.index)), errors="coerce"
            )
            if name in {self.spec.prior_exposure, self.spec.volume_feature}:
                values = np.log1p(values.clip(lower=0))
            if fit:
                fill = float(values.median()) if values.notna().any() else 0.0
                filled = values.fillna(fill)
                scale = float(filled.std(ddof=0))
                self.fill[name] = fill
                self.mean[name] = float(filled.mean())
                self.scale[name] = scale if scale > 1e-8 else 1.0
            missing = values.isna().to_numpy(dtype=float)
            filled = values.fillna(self.fill[name]).to_numpy(dtype=float)
            columns.append((filled - self.mean[name]) / self.scale[name])
            columns.append(missing)
        for position in ("QB", "RB", "WR"):
            columns.append(
                rows["position"].astype(str).eq(position).to_numpy(dtype=float)
            )
        return np.column_stack(columns)

    def _forward(self, values: np.ndarray) -> np.ndarray:
        if self.spec.transform == "logit":
            clipped = np.clip(values, 1e-4, 1 - 1e-4)
            return np.log(clipped / (1 - clipped))
        return values

    def _inverse(self, values: np.ndarray) -> np.ndarray:
        if self.spec.transform == "logit":
            values = 1.0 / (1.0 + np.exp(-np.clip(values, -20, 20)))
        return np.clip(values, self.spec.lower, self.spec.upper)


@dataclass
class SeasonAverageEfficiencyPipeline:
    """Fit and apply the complete set of player efficiency regressions."""

    alpha: float = 20.0
    use_volume: bool = True
    use_advanced: bool = True
    models: dict[str, ExposureWeightedEfficiencyModel] = field(default_factory=dict)

    def fit(self, rows: pd.DataFrame) -> "SeasonAverageEfficiencyPipeline":
        self.models = {}
        for spec in EFFICIENCY_MODEL_SPECS:
            model = ExposureWeightedEfficiencyModel(
                spec,
                alpha=self.alpha,
                use_volume=self.use_volume,
                use_advanced=self.use_advanced,
            )
            try:
                model.fit(rows)
            except ValueError:
                continue
            self.models[spec.target] = model
        if not self.models:
            raise ValueError("no efficiency response had eligible training rows")
        return self

    def predict(self, rows: pd.DataFrame) -> pd.DataFrame:
        out = rows.copy().reset_index(drop=True)
        for target, model in self.models.items():
            out[f"pred_{target}"] = model.predict(out)
        return out


def _posterior_stack(posterior, name: str, indices: np.ndarray) -> np.ndarray:
    values = posterior[name].stack(sample=("chain", "draw")).to_numpy()
    return np.take(values, indices, axis=-1)


def _sample_indices(available: int, draws: int, seed: int) -> np.ndarray:
    """Choose reproducible aligned posterior draws.

    Deterministic thinning retains the full chain span when enough samples are
    available. Prediction requests larger than a fitted posterior are handled
    by reproducible resampling, which is preferable to silently changing the
    volume pipeline's draw count.
    """
    if available <= 0 or draws <= 0:
        raise ValueError("posterior predictions require at least one draw")
    if draws <= available:
        return np.linspace(0, available - 1, draws, dtype=int)
    return np.random.default_rng(seed).choice(available, size=draws, replace=True)


@dataclass
class EfficiencyRatePrediction:
    """Posterior location and future-season rate draws for one response."""

    rows: pd.DataFrame
    mean: np.ndarray
    rate: np.ndarray


@dataclass
class SeasonAverageEfficiencyPrediction:
    """Aligned posterior efficiency draws for every fitted scoring response."""

    player_rows: pd.DataFrame
    means: dict[str, np.ndarray]
    rates: dict[str, np.ndarray]

    @property
    def draws(self) -> int:
        if not self.rates:
            return 0
        return next(iter(self.rates.values())).shape[1]


@dataclass
class PosteriorSeasonEfficiencyModel:
    """One exposure-aware Bayesian season-efficiency response.

    Proportion responses use a Beta-Binomial likelihood. The regression models
    the population mean while the concentration captures persistent
    player-season heterogeneity. Yardage responses use a bounded mean and a
    Student-t scale with separate season and per-opportunity components.
    """

    spec: EfficiencyModelSpec
    use_volume: bool = True
    use_advanced: bool = True
    mean_mode: str | None = None
    ridge_alpha: float = 500.0
    # Backward-compatible override for early efficiency-v2 checkpoints.
    prior_only: bool | None = None
    feature_names: list[str] = field(default_factory=list)
    missing_feature_names: list[str] = field(default_factory=list)
    feature_fill: dict[str, float] = field(default_factory=dict)
    feature_mean: dict[str, float] = field(default_factory=dict)
    feature_scale: dict[str, float] = field(default_factory=dict)
    feature_projection: np.ndarray | None = None
    prior_fill: float = 0.0
    prior_position_fill: dict[str, float] = field(default_factory=dict)
    prior_center: float = 0.0
    ridge_model: ExposureWeightedEfficiencyModel | None = None
    training_rows: int = 0
    idata: object = None

    def _mean_mode(self) -> str:
        if self.mean_mode is not None:
            mode = self.mean_mode
        elif self.prior_only is not None:
            mode = "prior" if self.prior_only else "posterior"
        else:
            mode = POSTERIOR_MEAN_MODE[self.spec.target]
        if mode not in {"prior", "ridge", "posterior"}:
            raise ValueError(f"unsupported posterior mean mode: {mode}")
        return mode

    def _uses_prior_only(self) -> bool:
        return self._mean_mode() == "prior"

    def _candidates(self) -> tuple[str, ...]:
        names = [self.spec.prior_exposure]
        if self._mean_mode() == "posterior":
            names.extend(BASE_EFFICIENCY_FEATURES)
            if self.use_volume:
                names.append(self.spec.volume_feature)
            if self.use_advanced:
                names.extend(self.spec.advanced_features)
        return tuple(dict.fromkeys(name for name in names if name))

    def _prior_signal(self, rows: pd.DataFrame, *, fit: bool = False) -> np.ndarray:
        values = pd.to_numeric(
            rows.get(
                self.spec.prior_feature, pd.Series(np.nan, index=rows.index)
            ),
            errors="coerce",
        )
        if fit:
            self.prior_fill = (
                float(values.median()) if values.notna().any() else float("nan")
            )
            if not np.isfinite(self.prior_fill):
                self.prior_fill = (self.spec.lower + self.spec.upper) / 2.0
            self.prior_position_fill = {}
            positions = rows["position"].astype(str).str.upper()
            for position in self.spec.positions:
                position_values = values[positions.eq(position)].dropna()
                self.prior_position_fill[position] = (
                    float(position_values.median())
                    if len(position_values)
                    else self.prior_fill
                )
        position_fill = rows["position"].astype(str).str.upper().map(
            self.prior_position_fill
        )
        filled = values.fillna(position_fill).fillna(self.prior_fill).to_numpy(
            dtype=float
        )
        span = self.spec.upper - self.spec.lower
        unit = np.clip((filled - self.spec.lower) / span, 1e-4, 1 - 1e-4)
        linked = logit(unit)
        if fit:
            self.prior_center = float(linked.mean())
        return linked - self.prior_center

    def _prior_mean(self, rows: pd.DataFrame) -> np.ndarray:
        signal = self._prior_signal(rows)
        unit = 1.0 / (
            1.0 + np.exp(-np.clip(signal + self.prior_center, -20.0, 20.0))
        )
        return self.spec.lower + (self.spec.upper - self.spec.lower) * unit

    def _fixed_mean(self, rows: pd.DataFrame) -> np.ndarray:
        mode = self._mean_mode()
        if mode == "prior":
            return self._prior_mean(rows)
        if mode == "ridge":
            if self.ridge_model is None:
                raise RuntimeError(
                    f"ridge mean for {self.spec.target} has not been fitted"
                )
            return self.ridge_model.predict(rows)
        raise RuntimeError("posterior-regression means are not fixed")

    def _eligible(self, rows: pd.DataFrame) -> pd.DataFrame:
        out = rows.copy().reset_index(drop=True)
        if "position" not in out:
            raise ValueError("efficiency rows are missing position")
        exposure = pd.to_numeric(
            out.get(self.spec.exposure, pd.Series(np.nan, index=out.index)),
            errors="coerce",
        )
        target = pd.to_numeric(
            out.get(self.spec.target, pd.Series(np.nan, index=out.index)),
            errors="coerce",
        )
        valid = (
            out["position"].astype(str).str.upper().isin(self.spec.positions)
            & exposure.ge(self.spec.min_exposure)
            & target.notna()
            & np.isfinite(target)
        )
        if self.spec.likelihood == "beta_binomial":
            if self.spec.numerator and self.spec.numerator in out:
                numerator = pd.to_numeric(out[self.spec.numerator], errors="coerce")
            else:
                numerator = target * exposure
            valid &= numerator.notna() & np.isfinite(numerator)
        replacement = pd.to_numeric(
            out.get("is_replacement_player", pd.Series(0, index=out.index)),
            errors="coerce",
        ).fillna(0)
        valid &= replacement.ne(1)
        return out[valid].reset_index(drop=True)

    def _feature_values(self, rows: pd.DataFrame, name: str) -> pd.Series:
        values = pd.to_numeric(
            rows.get(name, pd.Series(np.nan, index=rows.index)), errors="coerce"
        )
        if name in {self.spec.prior_exposure, self.spec.volume_feature}:
            values = np.log1p(values.clip(lower=0))
        return values

    def _raw_matrix(self, rows: pd.DataFrame, *, fit: bool = False) -> np.ndarray:
        if fit:
            self.feature_names = []
            self.missing_feature_names = []
            self.feature_fill = {}
            self.feature_mean = {}
            self.feature_scale = {}
            for name in self._candidates():
                if name not in rows:
                    continue
                values = self._feature_values(rows, name)
                if not values.notna().any():
                    continue
                fill = float(values.median())
                filled = values.fillna(fill)
                scale = float(filled.std(ddof=0))
                if scale <= 1e-8:
                    continue
                self.feature_names.append(name)
                self.feature_fill[name] = fill
                self.feature_mean[name] = float(filled.mean())
                self.feature_scale[name] = scale
                if values.isna().any() and values.notna().any():
                    self.missing_feature_names.append(name)

        columns: list[np.ndarray] = []
        for name in self.feature_names:
            values = self._feature_values(rows, name)
            filled = values.fillna(self.feature_fill[name]).to_numpy(dtype=float)
            columns.append(
                (filled - self.feature_mean[name]) / self.feature_scale[name]
            )
            if name in self.missing_feature_names:
                columns.append(values.isna().to_numpy(dtype=float))
        return np.column_stack(columns) if columns else np.zeros((len(rows), 0))

    def _matrix(self, rows: pd.DataFrame, *, fit: bool = False) -> np.ndarray:
        matrix = self._raw_matrix(rows, fit=fit)
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

    def _volume_feature_column(self) -> int | None:
        """Return the raw-design column for the fitted volume covariate."""
        if self.spec.volume_feature not in self.feature_names:
            return None
        column = 0
        for name in self.feature_names:
            if name == self.spec.volume_feature:
                return column
            column += 1 + int(name in self.missing_feature_names)
        return None

    def _scaled_volume_feature_samples(
        self, samples: np.ndarray, rows: int
    ) -> np.ndarray:
        values = np.asarray(samples, dtype=float)
        if values.ndim != 2 or values.shape[0] != rows:
            raise ValueError(
                f"{self.spec.target} volume-feature samples must have shape "
                "(rows, draws)"
            )
        name = self.spec.volume_feature
        if name not in self.feature_fill:
            raise RuntimeError(f"{self.spec.target} has no fitted volume feature")
        values = np.log1p(np.clip(values, 0.0, None))
        values = np.where(np.isfinite(values), values, self.feature_fill[name])
        return (values - self.feature_mean[name]) / self.feature_scale[name]

    def _position_index(self, rows: pd.DataFrame) -> np.ndarray:
        lookup = {position: i for i, position in enumerate(self.spec.positions)}
        return (
            rows["position"]
            .astype(str)
            .str.upper()
            .map(lookup)
            .fillna(0)
            .to_numpy(dtype=int)
        )

    def fit(self, rows: pd.DataFrame, **sample_kwargs) -> "PosteriorSeasonEfficiencyModel":
        import pymc as pm

        out = self._eligible(rows)
        if out.empty:
            raise ValueError(f"no eligible training rows for {self.spec.target}")
        exposure = (
            pd.to_numeric(out[self.spec.exposure], errors="coerce")
            .round()
            .astype(int)
            .to_numpy()
        )
        response = pd.to_numeric(out[self.spec.target], errors="coerce").to_numpy(
            dtype=float
        )
        X = self._matrix(out, fit=True)
        prior_signal = self._prior_signal(out, fit=True)
        mode = self._mean_mode()
        if mode == "ridge":
            self.ridge_model = ExposureWeightedEfficiencyModel(
                self.spec,
                alpha=self.ridge_alpha,
                use_volume=self.use_volume,
                use_advanced=self.use_advanced,
            ).fit(rows)
        position_index = self._position_index(out)
        weighted_center = float(np.average(response, weights=exposure))
        span = float(self.spec.upper - self.spec.lower)

        if self.spec.likelihood == "beta_binomial":
            if self.spec.numerator and self.spec.numerator in out:
                success = (
                    pd.to_numeric(out[self.spec.numerator], errors="coerce")
                    .round()
                    .astype(int)
                    .to_numpy()
                )
            else:
                success = np.rint(response * exposure).astype(int)
            success = np.clip(success, 0, exposure)
            center = float(logit(np.array([np.clip(weighted_center, 1e-4, 1 - 1e-4)]))[0])
        elif self.spec.likelihood == "continuous":
            unit_center = np.clip(
                (weighted_center - self.spec.lower) / span, 1e-4, 1 - 1e-4
            )
            center = float(logit(np.array([unit_center]))[0])
        else:
            raise ValueError(f"unsupported efficiency likelihood: {self.spec.likelihood}")

        with pm.Model() as model:
            if mode != "posterior":
                # Preserve the point forecast that survived the walk-forward
                # mean gate and estimate only the dispersion needed downstream.
                mean = np.clip(
                    self._fixed_mean(out), self.spec.lower, self.spec.upper
                )
                if self.spec.likelihood == "beta_binomial":
                    mean = np.clip(mean, 1e-5, 1.0 - 1e-5)
                    concentration = pm.LogNormal(
                        "concentration",
                        mu=np.log(self.spec.prior_concentration),
                        sigma=0.75,
                    )
                    pm.BetaBinomial(
                        "efficiency_obs",
                        n=exposure,
                        alpha=mean * concentration,
                        beta=(1.0 - mean) * concentration,
                        observed=success,
                    )
                else:
                    season_sigma = pm.HalfNormal(
                        "season_sigma", sigma=span * 0.10
                    )
                    opportunity_sigma = pm.HalfNormal(
                        "opportunity_sigma", sigma=span * 0.75
                    )
                    scale = pm.math.sqrt(
                        season_sigma**2 + opportunity_sigma**2 / exposure
                    )
                    pm.StudentT(
                        "efficiency_obs",
                        nu=5.0,
                        mu=mean,
                        sigma=scale,
                        observed=response,
                    )
            else:
                intercept = pm.Normal("intercept", center, 0.75)
                prior_persistence = pm.Normal(
                    "prior_persistence", mu=0.80, sigma=0.25
                )
                beta = pm.Normal("beta", 0.0, 0.30, shape=X.shape[1])
                eta = (
                    intercept
                    + prior_persistence * prior_signal
                    + pm.math.dot(X, beta)
                )
                if len(self.spec.positions) > 1:
                    raw = pm.Normal(
                        "position_effect_raw",
                        0.0,
                        0.35,
                        shape=len(self.spec.positions) - 1,
                    )
                    position_effect = pm.Deterministic(
                        "position_effect",
                        pm.math.dot(
                            _sum_to_zero_basis(len(self.spec.positions)), raw
                        ),
                    )
                    eta = eta + position_effect[position_index]

                if self.spec.likelihood == "beta_binomial":
                    mean = pm.math.sigmoid(eta)
                    concentration = pm.LogNormal(
                        "concentration",
                        mu=np.log(self.spec.prior_concentration),
                        sigma=0.75,
                    )
                    pm.BetaBinomial(
                        "efficiency_obs",
                        n=exposure,
                        alpha=mean * concentration,
                        beta=(1.0 - mean) * concentration,
                        observed=success,
                    )
                else:
                    mean = self.spec.lower + span * pm.math.sigmoid(eta)
                    season_sigma = pm.HalfNormal(
                        "season_sigma", sigma=span * 0.10
                    )
                    opportunity_sigma = pm.HalfNormal(
                        "opportunity_sigma", sigma=span * 0.75
                    )
                    scale = pm.math.sqrt(
                        season_sigma**2 + opportunity_sigma**2 / exposure
                    )
                    pm.StudentT(
                        "efficiency_obs",
                        nu=5.0,
                        mu=mean,
                        sigma=scale,
                        observed=response,
                    )
            sample_kwargs.setdefault("target_accept", 0.92)
            self.idata = sample_model(model, **sample_kwargs)
        self.training_rows = len(out)
        return self

    def _prediction_exposure(
        self,
        rows: pd.DataFrame,
        draws: int,
        exposure_samples: np.ndarray | None,
    ) -> np.ndarray:
        if exposure_samples is not None:
            exposure = np.asarray(exposure_samples, dtype=float)
            if exposure.ndim == 1:
                exposure = np.repeat(exposure[:, None], draws, axis=1)
            if exposure.shape != (len(rows), draws):
                raise ValueError(
                    f"{self.spec.target} exposure samples must align to rows and draws"
                )
            return np.clip(exposure, 1.0, None)

        prior = pd.to_numeric(
            rows.get(self.spec.prior_exposure, pd.Series(np.nan, index=rows.index)),
            errors="coerce",
        )
        if self.spec.volume_feature in rows:
            projected = pd.to_numeric(rows[self.spec.volume_feature], errors="coerce")
            games = pd.to_numeric(
                rows.get("team_games", pd.Series(17, index=rows.index)),
                errors="coerce",
            ).fillna(17)
            prior = projected.mul(games).combine_first(prior)
        fallback = float(prior[prior.gt(0)].median()) if prior.gt(0).any() else 1.0
        prior = prior.fillna(fallback).clip(lower=1.0).to_numpy(dtype=float)
        return np.repeat(prior[:, None], draws, axis=1)

    def predict_samples(
        self,
        rows: pd.DataFrame,
        *,
        draws: int | None = None,
        exposure_samples: np.ndarray | None = None,
        volume_feature_samples: np.ndarray | None = None,
        seed: int = 0,
    ) -> EfficiencyRatePrediction:
        if self.idata is None:
            raise RuntimeError("fit the posterior efficiency model before predicting")
        out = rows.copy().reset_index(drop=True)
        if "position" not in out:
            raise ValueError("efficiency rows are missing position")
        raw_X = self._raw_matrix(out)
        X = (
            raw_X
            if self.feature_projection is None
            else raw_X @ np.asarray(self.feature_projection, dtype=float)
        )
        posterior = self.idata.posterior
        available = int(posterior.sizes["chain"] * posterior.sizes["draw"])
        draws = available if draws is None else int(draws)
        indices = _sample_indices(available, draws, seed)
        if self._mean_mode() != "posterior":
            if (
                self._mean_mode() == "ridge"
                and volume_feature_samples is not None
                and self.ridge_model is not None
            ):
                mean = self.ridge_model.predict_volume_conditioned_samples(
                    out, volume_feature_samples
                )
                if mean.shape[1] != draws:
                    raise ValueError(
                        f"{self.spec.target} volume-feature draws must align "
                        "to requested posterior draws"
                    )
            else:
                mean = np.repeat(self._fixed_mean(out)[:, None], draws, axis=1)
        else:
            eta = _posterior_stack(posterior, "intercept", indices)[None, :]
            eta = eta + self._prior_signal(out)[:, None] * _posterior_stack(
                posterior, "prior_persistence", indices
            )[None, :]
            if X.shape[1]:
                beta = _posterior_stack(posterior, "beta", indices)
                eta = eta + X @ beta
                raw_column = self._volume_feature_column()
                if volume_feature_samples is not None and raw_column is not None:
                    dynamic = self._scaled_volume_feature_samples(
                        volume_feature_samples, len(out)
                    )
                    if dynamic.shape[1] != draws:
                        raise ValueError(
                            f"{self.spec.target} volume-feature draws must align "
                            "to requested posterior draws"
                        )
                    effective_beta = (
                        beta
                        if self.feature_projection is None
                        else np.asarray(self.feature_projection, dtype=float) @ beta
                    )
                    eta = eta + (
                        dynamic - raw_X[:, raw_column, None]
                    ) * effective_beta[raw_column, None, :]
            if len(self.spec.positions) > 1:
                position_index = self._position_index(out)
                eta = eta + _posterior_stack(
                    posterior, "position_effect", indices
                )[position_index, :]
            unit_mean = 1.0 / (
                1.0 + np.exp(-np.clip(eta, -20.0, 20.0))
            )
            mean = (
                unit_mean
                if self.spec.likelihood == "beta_binomial"
                else self.spec.lower
                + (self.spec.upper - self.spec.lower) * unit_mean
            )
        rng = np.random.default_rng(seed)
        if self.spec.likelihood == "beta_binomial":
            concentration = _posterior_stack(
                posterior, "concentration", indices
            )[None, :]
            rate = rng.beta(
                np.clip(mean * concentration, 1e-6, None),
                np.clip((1.0 - mean) * concentration, 1e-6, None),
            )
        else:
            exposure = self._prediction_exposure(out, draws, exposure_samples)
            season_sigma = _posterior_stack(
                posterior, "season_sigma", indices
            )[None, :]
            opportunity_sigma = _posterior_stack(
                posterior, "opportunity_sigma", indices
            )[None, :]
            scale = np.sqrt(
                season_sigma**2 + opportunity_sigma**2 / exposure
            )
            rate = mean + rng.standard_t(5.0, size=mean.shape) * scale
            rate = np.clip(rate, self.spec.lower, self.spec.upper)

        supported = (
            out["position"].astype(str).str.upper().isin(self.spec.positions).to_numpy()
        )
        mean = np.where(supported[:, None], mean, np.nan)
        rate = np.where(supported[:, None], rate, np.nan)
        return EfficiencyRatePrediction(out, mean, rate)

    def predict_observed_samples(
        self,
        rows: pd.DataFrame,
        *,
        draws: int | None = None,
        seed: int = 0,
    ) -> np.ndarray:
        """Posterior predictive response draws at each row's actual exposure."""
        exposure = pd.to_numeric(
            rows.get(self.spec.exposure, pd.Series(np.nan, index=rows.index)),
            errors="coerce",
        ).fillna(0).round().astype(int).to_numpy()
        requested_draws = draws
        prediction = self.predict_samples(
            rows,
            draws=requested_draws,
            exposure_samples=exposure,
            seed=seed,
        )
        if self.spec.likelihood != "beta_binomial":
            return prediction.rate
        rng = np.random.default_rng(seed + 10_000)
        success = rng.binomial(exposure[:, None], np.nan_to_num(prediction.rate))
        return np.divide(
            success,
            exposure[:, None],
            out=np.full_like(prediction.rate, np.nan, dtype=float),
            where=exposure[:, None] > 0,
        )

    def predict(self, rows: pd.DataFrame) -> np.ndarray:
        return np.nanmean(self.predict_samples(rows).mean, axis=1)


@dataclass
class SeasonAveragePosteriorEfficiencyPipeline:
    """Fit, persist, and apply every posterior season-efficiency model."""

    use_volume: bool = True
    use_advanced: bool = True
    ridge_alpha: float = 500.0
    # Each response is fitted only on rows clearing its ``min_exposure`` but is
    # then scored on every row, and the gap is large: on the nflverse frame 57%
    # of quarterback rows, 58% of receiving rows and 82% of rushing rows fall
    # below their floor. So the fitted mean describes high-usage players and is
    # extrapolated onto everyone else. Both likelihoods already downweight a
    # thin row correctly — a Beta-Binomial with n=3 carries almost no
    # information, and the Student-t scale grows as sqrt(1/n) — so the hard
    # floor is doing by exclusion what the likelihood does by weighting, and it
    # buys that at the cost of selecting on usage. Lowering this admits the
    # excluded population; ``None`` keeps each spec's own floor.
    #
    # Promoted 2026-08-02 at 5, chosen on an inner fold rather than on the
    # holdouts it is scored against — see scripts/select_exposure_floor.py. The
    # nested procedure improves efficiency MAE on all three outer holdouts
    # (-0.437%, -0.373%, -0.314%; pooled -0.374% +/- 0.036%) and CRPS by
    # 0.68-1.04%. On total fantasy points it is worth 0.31-0.40% MAE across all
    # three scoring systems, with every coverage move inside half a point.
    #
    # The inner folds picked 5, 5 and 10. On the 2022 inner fold 5 and 10 differ
    # by 0.001 percentage points, so the choice between them is noise; what is
    # consistent across all three is that lowering the floor beats each spec's
    # own. 5 is the majority pick and is within 0.18pp of the per-fold argmin
    # everywhere. Taking the per-fold winner instead would be fitting the noise
    # this procedure exists to avoid.
    exposure_floor: int | None = 5
    models: dict[str, PosteriorSeasonEfficiencyModel] = field(default_factory=dict)
    fit_seconds: dict[str, float] = field(default_factory=dict)

    def fit(
        self,
        rows: pd.DataFrame,
        *,
        targets: tuple[str, ...] | list[str] | None = None,
        **sample_kwargs,
    ) -> "SeasonAveragePosteriorEfficiencyPipeline":
        selected = (
            set(EFFICIENCY_MODEL_BY_TARGET)
            if targets is None
            else set(map(str, targets))
        )
        unknown = selected - set(EFFICIENCY_MODEL_BY_TARGET)
        if unknown:
            raise ValueError(f"unknown efficiency targets: {sorted(unknown)}")
        self.models = {}
        self.fit_seconds = {}
        base_seed = int(sample_kwargs.pop("seed", 42))
        for index, spec in enumerate(EFFICIENCY_MODEL_SPECS):
            if spec.target not in selected:
                continue
            if self.exposure_floor is not None:
                spec = replace(
                    spec, min_exposure=min(spec.min_exposure, int(self.exposure_floor))
                )
            model = PosteriorSeasonEfficiencyModel(
                spec,
                use_volume=self.use_volume,
                use_advanced=self.use_advanced,
                ridge_alpha=self.ridge_alpha,
            )
            started = perf_counter()
            try:
                model.fit(rows, seed=base_seed + index, **sample_kwargs)
            except ValueError as error:
                if "no eligible training rows" not in str(error):
                    raise
                continue
            self.fit_seconds[spec.target] = perf_counter() - started
            self.models[spec.target] = model
        if not self.models:
            raise ValueError("no efficiency response had eligible training rows")
        return self

    def predict_samples(
        self,
        rows: pd.DataFrame,
        *,
        draws: int | None = None,
        exposure_samples: dict[str, np.ndarray] | None = None,
        volume_feature_samples: dict[str, np.ndarray] | None = None,
        seed: int = 0,
    ) -> SeasonAverageEfficiencyPrediction:
        if not self.models:
            raise RuntimeError("fit the posterior efficiency pipeline before predicting")
        if draws is None:
            draws = min(
                int(model.idata.posterior.sizes["chain"])
                * int(model.idata.posterior.sizes["draw"])
                for model in self.models.values()
            )
        out = rows.copy().reset_index(drop=True)
        means: dict[str, np.ndarray] = {}
        rates: dict[str, np.ndarray] = {}
        exposure_samples = exposure_samples or {}
        volume_feature_samples = volume_feature_samples or {}
        for target, model in self.models.items():
            exposure = exposure_samples.get(
                target, exposure_samples.get(model.spec.exposure)
            )
            volume_features = volume_feature_samples.get(model.spec.volume_feature)
            prediction = model.predict_samples(
                out,
                draws=draws,
                exposure_samples=exposure,
                volume_feature_samples=volume_features,
                seed=seed + EFFICIENCY_SEED_OFFSET[target],
            )
            means[target] = prediction.mean
            rates[target] = prediction.rate
        return SeasonAverageEfficiencyPrediction(out, means, rates)

    def predict(self, rows: pd.DataFrame) -> pd.DataFrame:
        out = rows.copy().reset_index(drop=True)
        for target, model in self.models.items():
            out[f"pred_{target}"] = model.predict(out)
        return out

    def diagnostics(self, *, min_bulk_ess: float = 100.0) -> dict[str, object]:
        results = {}
        for target, model in self.models.items():
            if model._mean_mode() != "posterior":
                variables = (
                    ["concentration"]
                    if model.spec.likelihood == "beta_binomial"
                    else ["season_sigma", "opportunity_sigma"]
                )
            else:
                variables = ["intercept", "prior_persistence", "beta"]
                if len(model.spec.positions) > 1:
                    variables.append("position_effect")
                if model.spec.likelihood == "beta_binomial":
                    variables.append("concentration")
                else:
                    variables.extend(("season_sigma", "opportunity_sigma"))
            results[target] = sampling_quality(
                model.idata, variables, min_bulk_ess=min_bulk_ess
            )
        return results

    @staticmethod
    def _model_state(model: PosteriorSeasonEfficiencyModel) -> dict[str, object]:
        ridge_state = None
        if model.ridge_model is not None:
            ridge = model.ridge_model
            ridge_state = {
                "alpha": ridge.alpha,
                "use_volume": ridge.use_volume,
                "use_advanced": ridge.use_advanced,
                "feature_names": ridge.feature_names,
                "fill": ridge.fill,
                "mean": ridge.mean,
                "scale": ridge.scale,
                "coefficients": (
                    None
                    if ridge.coefficients is None
                    else np.asarray(ridge.coefficients).tolist()
                ),
                "training_rows": ridge.training_rows,
            }
        return {
            "feature_names": model.feature_names,
            "missing_feature_names": model.missing_feature_names,
            "feature_fill": model.feature_fill,
            "feature_mean": model.feature_mean,
            "feature_scale": model.feature_scale,
            "feature_projection": (
                None
                if model.feature_projection is None
                else np.asarray(model.feature_projection).tolist()
            ),
            "prior_only": model.prior_only,
            "mean_mode": model._mean_mode(),
            "ridge_alpha": model.ridge_alpha,
            "ridge_model": ridge_state,
            "prior_fill": model.prior_fill,
            "prior_position_fill": model.prior_position_fill,
            "prior_center": model.prior_center,
            "training_rows": model.training_rows,
        }

    @staticmethod
    def _restore_model_state(
        model: PosteriorSeasonEfficiencyModel, state: dict[str, object]
    ) -> None:
        model.feature_names = list(state["feature_names"])
        model.missing_feature_names = list(state.get("missing_feature_names", ()))
        model.prior_only = state.get("prior_only")
        model.mean_mode = state.get("mean_mode")
        model.ridge_alpha = float(state.get("ridge_alpha", 500.0))
        model.prior_fill = float(state.get("prior_fill", 0.0))
        model.prior_position_fill = {
            key: float(value)
            for key, value in state.get("prior_position_fill", {}).items()
        }
        model.prior_center = float(state.get("prior_center", 0.0))
        ridge_state = state.get("ridge_model")
        if ridge_state is not None:
            ridge = ExposureWeightedEfficiencyModel(
                model.spec,
                alpha=float(ridge_state["alpha"]),
                use_volume=bool(ridge_state["use_volume"]),
                use_advanced=bool(ridge_state["use_advanced"]),
            )
            ridge.feature_names = list(ridge_state["feature_names"])
            for attribute in ("fill", "mean", "scale"):
                setattr(
                    ridge,
                    attribute,
                    {
                        key: float(value)
                        for key, value in ridge_state[attribute].items()
                    },
                )
            coefficients = ridge_state.get("coefficients")
            ridge.coefficients = (
                None
                if coefficients is None
                else np.asarray(coefficients, dtype=float)
            )
            ridge.training_rows = int(ridge_state.get("training_rows", 0))
            model.ridge_model = ridge
        for attribute in ("feature_fill", "feature_mean", "feature_scale"):
            setattr(
                model,
                attribute,
                {key: float(value) for key, value in state[attribute].items()},
            )
        projection = state.get("feature_projection")
        model.feature_projection = (
            None if projection is None else np.asarray(projection, dtype=float)
        )
        model.training_rows = int(state.get("training_rows", 0))

    def save(self, directory: str | Path) -> Path:
        if not self.models or any(model.idata is None for model in self.models.values()):
            raise RuntimeError("fit posterior efficiency models before saving")
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        metadata = {
            "architecture_version": 1,
            "use_volume": self.use_volume,
            "use_advanced": self.use_advanced,
            "ridge_alpha": self.ridge_alpha,
            # Records which rows the responses were fitted on. It has no effect
            # at prediction time, which is exactly why it has to be written
            # down: without it a reloaded pipeline reports whatever the current
            # default happens to be, and a refit from that configuration would
            # silently train on a different sample.
            "exposure_floor": self.exposure_floor,
            "fit_seconds": self.fit_seconds,
            "models": {},
        }
        for target, model in self.models.items():
            save_idata(model.idata, directory / f"{target}.nc")
            metadata["models"][target] = self._model_state(model)
        (directory / "metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
        )
        return directory

    @classmethod
    def load(cls, directory: str | Path) -> "SeasonAveragePosteriorEfficiencyPipeline":
        directory = Path(directory)
        metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        pipeline = cls(
            use_volume=bool(metadata["use_volume"]),
            use_advanced=bool(metadata["use_advanced"]),
            ridge_alpha=float(metadata.get("ridge_alpha", 500.0)),
            exposure_floor=(
                None
                if metadata.get("exposure_floor") is None
                else int(metadata["exposure_floor"])
            ),
            fit_seconds={
                key: float(value)
                for key, value in metadata.get("fit_seconds", {}).items()
            },
        )
        # Restore in spec order rather than the alphabetical order the metadata
        # is written in, so a reloaded pipeline iterates exactly as a fitted one
        # does. The per-response seed no longer depends on this, but anything
        # else that walks ``self.models`` still gets the same order either way.
        saved = metadata["models"]
        ordered = sorted(saved, key=lambda t: EFFICIENCY_SEED_OFFSET.get(t, len(saved)))
        for target in ordered:
            state = saved[target]
            if target not in EFFICIENCY_MODEL_BY_TARGET:
                raise ValueError(f"saved efficiency target is unsupported: {target}")
            model = PosteriorSeasonEfficiencyModel(
                EFFICIENCY_MODEL_BY_TARGET[target],
                use_volume=pipeline.use_volume,
                use_advanced=pipeline.use_advanced,
                mean_mode=state.get("mean_mode"),
                ridge_alpha=float(state.get("ridge_alpha", 500.0)),
                prior_only=state.get("prior_only"),
            )
            cls._restore_model_state(model, state)
            model.idata = load_idata(directory / f"{target}.nc")
            pipeline.models[target] = model
        return pipeline
