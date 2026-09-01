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
        (
            "prior_rush_epa_per_carry",
            "prior_rush_first_down_rate",
            # The situation the carries came from, not just how they went.
            # -1.12% MAE and -1.03% CRPS on 2023/2024/2025, three folds of
            # three, and -5.60% on the quartile of backs who actually run in
            # short yardage against +0.46% on the quartile who do not.
            "prior_rush_short_yardage_share",
        ),
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

# The receiving responses, which are the ones a quarterback plausibly moves.
# Rushing is left out on purpose: the mechanism by which a passer changes yards
# per carry is indirect at best, and testing where there is no story to tell is
# how a feature earns a fold win by chance.
TEAMMATE_QUALITY_TARGETS = ("rec_catch_rate", "rec_yards_per_target", "rec_td_rate")

# Reserve status on the one efficiency response that was measured to want it.
#
# Deliberately not the three receiving responses. rec_yards_per_target is what
# the walk-forward scored: MAE -0.37% and CRPS -0.26% overall and -2.48% and
# -1.85% on the reserve population, three holdouts of three. rec_catch_rate
# admits no covariates at all -- its mean mode returns an empty design, so every
# arm on it produced byte-identical numbers -- and rec_td_rate was never scored.
# Extending an accepted result to responses it was not measured on is how a
# promotion quietly becomes three unvalidated arms.
RESERVE_EFFICIENCY_TARGETS = ("rec_yards_per_target",)
RESERVE_EFFICIENCY_FEATURES = ("roster_reserve",)

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
    "rec_catch_rate": "posterior",
    "rec_yards_per_target": "posterior",
    "rec_td_rate": "prior",
    "rush_yards_per_carry": "ridge",
    "rush_td_rate": "posterior",
    "fumble_lost_rate": "prior",
}

# The ``prior`` responses, with the one thing the mean gate never offered them.
#
# ``prior`` mode is an identity map: ``_prior_signal`` links the lagged feature
# and subtracts a centre, ``_prior_mean`` re-adds that centre and inverts the
# link, so the conditional mean handed to the simulator *is*
# ``prior_<response>`` exactly. Its implied persistence coefficient on the
# shrunk feature is 1.000, and the whole of the layer's regression to the mean
# is therefore the pseudo-count ``K`` in ``EFFICIENCY_SPECS``.
#
# Measured by ``scripts/measure_efficiency_reversion.py`` over nflverse
# 1999-2025, walk-forward, scored in touchdowns at realized next-season
# exposure so the volume layer is held out of it:
#
#   response          K     effective persistence   slope the data wants   held-out MAE
#   rec_td_rate       120           0.413                   0.591          -3.15%, 12/15
#   rush_td_rate      120           0.572                   0.441          -5.42%, 12/18
#   rec_catch_rate     40           0.671                   0.795          -2.38%, 13/15
#
# The error is a location bias that changes sign across the distribution, which
# is why a pooled metric could not see it: by quintile of the lagged feature,
# the top fifth is projected +5.1 PPR points high at receiver and +8.0 at
# running back, the bottom fifth the same distance low, with the sign flipping
# at the median on 15 and 18 folds.
#
# The efficiency-v2 gate is not evidence against this. Its challenger was the
# full posterior regression -- fitted persistence *plus* the base-feature block,
# the advanced-efficiency block and the projected-volume covariate, all admitted
# at once on roughly 2,000 rows -- and it lost 0/3 folds on receiving touchdown
# rate. There was no mode between "use the feature raw" and "regress on
# everything". This is that mode: an intercept, sum-to-zero position offsets,
# and a slope. Position offsets are not optional -- the shrinkage pools toward a
# season-and-position mean, so a single shared intercept would pull three
# positions with genuinely different touchdown rates toward one grand mean.
#
# It is a strict generalisation of ``prior``: slope 1, intercept at the prior
# centre and zero position offsets reproduce today's behaviour exactly, so the
# arm cannot lose by more than sampling noise.
#
# ``fumble_lost_rate`` stays on ``prior``. It is 0.05% of points variance and
# was not measured; there is no reason to widen the surface for it.
#
# Validated by ``scripts/validate_persistence_mean.py`` -- both arms the real
# model on the 2015-2025 frame, holding out 2022/2023/2024, 600 draws, four
# chains, zero divergences and max R-hat 1.01 throughout:
#
#   response          fitted slope    MAE            CRPS
#   rec_catch_rate    0.537-0.545     -1.00%  3/3    -1.45%  3/3
#   rush_td_rate      0.327-0.361     -2.48%  2/3    -0.26%  1/3
#   rec_td_rate       0.310-0.342     -1.10%  2/3    +2.00%  1/3
#
# The finding holds everywhere: the fitted slope excludes 1.000 on every fold of
# every response, so the layer was asserting a persistence the data does not
# support. The *remedy* only half works, and the reason is visible in the
# likelihood. A Beta-Binomial has one location and one dispersion. With the mean
# pinned to an identity map that is wrong at the tails, the only free parameter
# left to explain the residual is the concentration, so it is fitted small and
# the predictive comes out wide. Fit the mean, the residual shrinks, the
# concentration is fitted larger, and the predictive narrows -- correct for a
# model whose only spread is Beta-Binomial noise, wrong here, because much of
# the real season-to-season spread in touchdown rate is player-season
# heterogeneity that no covariate in this arm explains. Sharpening the location
# leaves nothing holding that variance open, and CRPS charges for it.
#
# CRPS was first reported against the *latent* rate draws, and that is the wrong
# target. The response is an observed season rate and carries binomial sampling
# noise at the player's exposure; the latent draws do not. Scoring one against
# the other penalises whichever arm has the tighter latent distribution -- and
# fitting the mean is exactly what tightens it, so the better arm took the
# larger penalty. Re-scored against the posterior predictive at realized
# exposure, on the same fits:
#
#   response          CRPS vs latent draws    CRPS vs posterior predictive
#   rec_catch_rate    -1.45%  3/3             -2.08%  3/3
#   rush_td_rate      -0.26%  1/3             -2.31%  3/3
#   rec_td_rate       +2.00%  1/3             -0.31%  1/3
#
# Nothing regresses on the correct target. Two responses improve materially and
# unanimously; ``rec_td_rate`` is immaterial either way but no longer negative,
# and it clears the efficiency-v2 mean gate on MAE (-1.10%, two folds of three).
# All three are promoted.
#
# 80% coverage of the predictive, against a nominal 0.80: catch rate 0.870 ->
# 0.872, rushing TD 0.918 -> 0.918, receiving TD 0.899 -> 0.908. The layer
# over-covers, in the base arm as much as the challenger, and this change does
# not move it. That reproduces the efficiency-v2 table (0.899 and 0.917) closely
# enough to be a useful check on the harness.
# Two of the three came back off persistence. Measured by
# ``scripts/validate_catch_rate_covariates.py`` on holdouts 2023/2024/2025, 600
# draws and four chains, zero divergences, scored against the posterior
# predictive at realized exposure on each spec's eligible population:
#
#   response          MAE            CRPS           cov80
#   rec_catch_rate    -1.53%  3/3    -1.10%  3/3    0.839 -> 0.844
#   rush_td_rate      -1.13%  3/3    -1.24%  3/3    0.900 -> 0.906
#
# This does not overturn the reasoning above; it narrows it. The claim that the
# layer was asserting a persistence the data does not support still stands, and
# ``persistence`` mode is still better than ``prior``. What the note above got
# wrong was the scope of the efficiency-v2 evidence: the full posterior
# regression lost 0/3 folds on *receiving touchdown rate*, and that one result
# was generalized to all three responses. It does not transfer. rec_td_rate is
# the response with no covariate that correlates with it at all -- a situational
# screen over target depth, air-yards spread, screen share and end-zone share
# found nothing beyond the prior -- so an empty design costs it nothing, and it
# stays here. The other two have covariates their own specs already name and
# were discarding: aDOT for catch rate, prior EPA and first-down rate for
# rushing touchdowns.
#
# Against the obvious story: the catch-rate gain is *not* concentrated on deep
# receivers. The top aDOT quartile improves 3/3 folds but by -1.09% MAE, about
# what the whole population gets, so this is a response wanting a fitted
# covariate design rather than deep threats having been mispredicted in
# particular.
#
# ``fumble_lost_rate`` stays on ``prior`` for the reason given above, unmeasured
# and immaterial.
PERSISTENCE_MEAN_MODE = {
    "rec_td_rate": "persistence",
}



# Width of the prior on the Beta-Binomial overdispersion, in log space.
#
# It was 0.75, and at that width rec_td_rate ran the concentration to 1.6e19 on
# the 2020 fold: 178 divergences, bulk ESS 7 of 4000 draws, R-hat 1.54. The
# chains did not find that value -- it is about fifty prior standard deviations
# out -- they broke, most likely on the precision of the Beta-Binomial
# log-density at very large alpha and beta.
#
# Truncating the parameter was tried first and worked, but a bounded variable's
# transform costs a scattering of single divergences that the acceptance gate
# treats as blockers, and raising target_accept to 0.95 did not clear them.
# Tightening the prior instead keeps the log transform, fixes the runaway, and
# samples better than truncation everywhere tested.
#
# It is a modelling change, so what it moves was measured against how much each
# response is worth. The shifts are largest where they matter least, which is
# the prior behaving correctly rather than a cost:
#
#   response              share of points variance   concentration shift
#   rec_catch_rate                  8.39%                    +0.2%
#   pass_td_rate                    7.03%                    -3.1%
#   rec_td_rate                     2.60%                    -7.3%
#   rush_td_rate                    2.33%                    -5.1%
#   pass_int_rate                   0.39%                   -20.9%
#   fumble_lost_rate                0.05%                   -38.0%
#
# Rare-event responses have diffuse concentration posteriors because the data
# cannot pin them down, so a tighter prior pulls them in. High-usage responses
# are pinned by the data and barely move.
CONCENTRATION_PRIOR_SIGMA = 0.40


def _concentration(pm, prior: float):
    """Overdispersion, on the log scale so the sampler sees no boundary."""
    return pm.LogNormal(
        "concentration", mu=np.log(prior), sigma=CONCENTRATION_PRIOR_SIGMA
    )


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
    # Covariates on trial, appended to the response's own design. Opt-in and
    # empty by default, so a model fitted without them is byte-identical.
    extra_features: tuple[str, ...] = ()
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
        if mode not in {"prior", "ridge", "posterior", "persistence"}:
            raise ValueError(f"unsupported posterior mean mode: {mode}")
        return mode

    def _uses_prior_only(self) -> bool:
        return self._mean_mode() == "prior"

    def _fits_the_mean(self) -> bool:
        """Whether the likelihood estimates the location as well as the spread.

        ``prior`` and ``ridge`` hand the likelihood a fixed mean computed
        outside it. ``posterior`` and ``persistence`` both build a linear
        predictor the sampler fits; they differ only in whether the covariate
        block is admitted, and ``_candidates`` is what decides that.
        """
        return self._mean_mode() in {"posterior", "persistence"}

    def _candidates(self) -> tuple[str, ...]:
        # ``persistence`` is the shrunk prior with a fitted intercept, position
        # offsets and a slope, and nothing else -- an empty design is the
        # point of it, not an oversight. Its own exposure already entered
        # through the ``den / (den + K)`` weight in the feature.
        if self._mean_mode() == "persistence":
            return ()
        names = [self.spec.prior_exposure]
        if self._mean_mode() == "posterior":
            names.extend(BASE_EFFICIENCY_FEATURES)
            if self.use_volume:
                names.append(self.spec.volume_feature)
            if self.use_advanced:
                names.extend(self.spec.advanced_features)
            names.extend(self.extra_features)
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
            # A numerator larger than its own exposure is not a rate, and it is
            # not rare enough to ignore: 71 rows of 7,937 on the 2015-2025
            # frame, in every season of the shipping window.
            #
            # The cause is a scope mismatch rather than bad source data.
            # ``player_preseason_rows`` merges the efficiency labels on
            # ``(season, player_key)`` while the frame is keyed by
            # ``(season, team, player_key)``, so the numerator (``eff_*``) is
            # the player's season total across every team he played for while
            # the exposure stays team-scoped to his Week-1 roster snapshot. A
            # mid-season move therefore pairs one team's targets with the whole
            # season's receptions -- Shaun Draughn in 2015 arrives as 27
            # receptions on 3 targets.
            #
            # ``fit`` clips success to exposure, so before this filter those
            # rows trained as a rate of exactly 1.000: a fabricated label, not a
            # dropped one. Excluding them is the conservative half of the fix
            # and is obviously right on its own. The other half is larger and
            # deliberately not done here: roughly 3.9% of rows (7.8% of
            # team-changers) carry the same mismatch without tripping the
            # inequality, and correcting those means changing the exposure the
            # whole layer trains on, which needs its own gate.
            valid &= numerator.le(exposure)
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
            if not self._fits_the_mean():
                # Preserve the point forecast that survived the walk-forward
                # mean gate and estimate only the dispersion needed downstream.
                mean = np.clip(
                    self._fixed_mean(out), self.spec.lower, self.spec.upper
                )
                if self.spec.likelihood == "beta_binomial":
                    mean = np.clip(mean, 1e-5, 1.0 - 1e-5)
                    concentration = _concentration(
                        pm, self.spec.prior_concentration
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
                eta = intercept + prior_persistence * prior_signal
                # ``persistence`` mode has no covariates by construction, and a
                # zero-length Normal is not a random variable the sampler can
                # be asked for. Guarding here rather than materialising an
                # empty ``beta`` also keeps it out of the posterior, so
                # ``diagnostics`` does not look for a variable that is not
                # there.
                if X.shape[1]:
                    beta = pm.Normal("beta", 0.0, 0.30, shape=X.shape[1])
                    eta = eta + pm.math.dot(X, beta)
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
                    concentration = _concentration(
                        pm, self.spec.prior_concentration
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
        if not self._fits_the_mean():
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
    # Cross-positional teammate quality on the receiving responses. Off
    # until gated: every efficiency spec has run with an empty feature list
    # since the pipeline was written, so this is the first thing any of them
    # learns about a player's teammates. See docs/teammate-quality-2026-08.md.
    teammate_quality_features: bool = False
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
    # Give the ``prior``-mode responses a fitted intercept, position offsets and
    # a persistence slope instead of asserting a slope of 1.000. See
    # ``PERSISTENCE_MEAN_MODE`` for what it replaces and what it is worth. Kept
    # as its own flag so the single-season identity-map arm stays reproducible.
    fitted_persistence_means: bool = True
    # Reserve status on rec_yards_per_target. Promoted 2026-09-01: the flag is
    # worth MAE -0.37% and CRPS -0.26% overall and -2.48% and -1.85% on the
    # reserve population, winning three holdouts of three in every population
    # scored. See scripts/validate_injury_efficiency.py and
    # reports/injury_efficiency.json.
    #
    # It is the only injury covariate that has cleared a gate outside the
    # availability layer. Four encodings were rejected in the role layer, and
    # injury recurrence adds nothing here either (-0.06%, two folds of three),
    # so this is the pooled reserve flag on one response, not injury information
    # in general.
    reserve_efficiency_features: bool = True
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
        if (
            self.teammate_quality_features
            and "teammate_qb_quality_signal" not in rows
        ):
            # ``_matrix`` keeps only features present in the frame, so an absent
            # one is dropped without a word and the model fits as though the
            # flag were off. That is how this feature's first walk-forward came
            # back identical to its baseline to five decimal places on every
            # metric: the cached frames predated the column. A flag that asks
            # for a feature and silently gets none has to fail instead.
            raise ValueError(
                "teammate_quality_features is on but teammate_qb_quality_signal "
                "is not in the rows; rebuild the frames with "
                "add_teammate_quality_features rather than fitting a model that "
                "quietly ignores the flag"
            )
        if self.reserve_efficiency_features and "roster_reserve" not in rows:
            # Same failure the teammate-quality flag hit: ``_matrix`` keeps only
            # the features present in the frame, so a missing one is dropped
            # without a word and the model fits as though the flag were off.
            raise ValueError(
                "reserve_efficiency_features is on but roster_reserve is not in "
                "the rows; rebuild the frames with "
                "scripts/build_projection_cache.py rather than fitting a model "
                "that quietly ignores the flag"
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
            if self.teammate_quality_features and spec.target in TEAMMATE_QUALITY_TARGETS:
                spec = replace(
                    spec,
                    advanced_features=(
                        *spec.advanced_features,
                        "teammate_qb_quality_signal",
                    ),
                )
            mean_mode = (
                PERSISTENCE_MEAN_MODE.get(spec.target)
                if self.fitted_persistence_means
                else None
            )
            extra_features = (
                RESERVE_EFFICIENCY_FEATURES
                if self.reserve_efficiency_features
                and spec.target in RESERVE_EFFICIENCY_TARGETS
                else ()
            )
            model = PosteriorSeasonEfficiencyModel(
                spec,
                mean_mode=mean_mode,
                use_volume=self.use_volume,
                use_advanced=self.use_advanced,
                ridge_alpha=self.ridge_alpha,
                extra_features=extra_features,
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
            if not model._fits_the_mean():
                variables = (
                    ["concentration"]
                    if model.spec.likelihood == "beta_binomial"
                    else ["season_sigma", "opportunity_sigma"]
                )
            else:
                variables = ["intercept", "prior_persistence"]
                if model.feature_names:
                    variables.append("beta")
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
            # ``feature_names`` is restored directly, so prediction is correct
            # without this. It is written for the same reason ``exposure_floor``
            # is: a refit from a reloaded pipeline would otherwise rebuild the
            # design from whatever the current default is, and silently drop a
            # covariate the artifact was fitted with.
            "extra_features": list(model.extra_features),
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
                extra_features=tuple(state.get("extra_features", ())),
            )
            cls._restore_model_state(model, state)
            model.idata = load_idata(directory / f"{target}.nc")
            pipeline.models[target] = model
        return pipeline
