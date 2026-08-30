"""Preseason availability and quarterback workload models.

All models consume only information available before the projected season.
Availability uses a Bernoulli/Beta-Binomial hurdle for appearing and games
active conditional on appearing. Quarterback workload uses a roster-softmax
Multinomial over offensive snaps, so every draw is a continuous within-team
share rather than a starter/backup classification.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ffmodel.features.volume import MODEL_POSITIONS
from ffmodel.models.base import (
    calibrate_innovation_scale,
    logit,
    mean_preserving_shares,
    sample_model,
    simplex_shares,
)
from ffmodel.models.volume_team import _sum_to_zero_basis

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
    "is_replacement_player",
)

# A player's own availability history, beyond the single season the layer has
# always read.
#
# ``features/season_pathways.py`` has built this for every row since the
# pathway work: a career exponentially-weighted mean at alpha 0.50, grouped by
# ``player_key`` so it follows a trade, over inputs that are already lagged.
# ``build_season_average_data`` attaches it to every frame. Nothing read it.
#
# One lagged season is a poor estimate of a durability trait -- pooled
# year-over-year availability correlation is 0.365 -- and averaging three is a
# better one. Measured by ``scripts/measure_history_depth.py`` on a 2018-2024
# build: held-out availability MAE 0.25836 -> 0.25271, **-2.19% on five folds of
# five**, against the package's 0.25% materiality floor.
#
# The trend term is deliberately not here. It exists (``prior_availability_trend``)
# but it needs two *consecutive* prior seasons and is missing on roughly half the
# rows, and a half-missing feature in a design that median-fills is a different
# and larger change than adding a mean.
AVAILABILITY_HISTORY_FEATURES = ("prior_availability_3yr",)

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
    "is_replacement_qb",
)


def _logit_scalar(p: float, eps: float = 1e-3) -> float:
    """Logit of a proportion, kept away from the asymptotes for use as a prior."""
    p = float(np.clip(p, eps, 1.0 - eps))
    return float(np.log(p / (1.0 - p)))


def _stack(posterior, name: str) -> np.ndarray:
    return posterior[name].stack(sample=("chain", "draw")).to_numpy()


def _apply_slopes(beta: np.ndarray, X: np.ndarray, position_index) -> np.ndarray:
    """Linear predictor for a shared slope vector or a per-position one.

    Read off the posterior's own shape rather than a flag, so an artifact
    fitted one way cannot be served the other. ``(features, draws)`` is the
    shared vector; ``(positions, features, draws)`` is the hierarchy.
    """
    beta = np.asarray(beta, dtype=float)
    if beta.ndim == 2:
        return X @ beta
    if beta.ndim != 3:
        raise ValueError(
            f"availability slopes have unexpected shape {beta.shape}; expected "
            "(features, draws) or (positions, features, draws)"
        )
    # One matrix product per position rather than materialising a
    # (rows, features, draws) array, which for a full frame is gigabytes.
    out = np.zeros((X.shape[0], beta.shape[-1]), dtype=float)
    for position in np.unique(position_index):
        rows = position_index == position
        out[rows] = X[rows] @ beta[position]
    return out


def _position_effect(pm, name: str, scale: float, size: int):
    raw = pm.Normal(f"{name}_raw", 0.0, scale, shape=size - 1)
    return pm.Deterministic(name, pm.math.dot(_sum_to_zero_basis(size), raw))


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
        "is_replacement_player": 0.0,
        "is_replacement_qb": 0.0,
    }
    for name, value in defaults.items():
        if name not in out:
            out[name] = value
    return out


def _eligible_games(rows: pd.DataFrame, team_games: np.ndarray) -> np.ndarray:
    """Team games less any games a known suspension already removes.

    A ban whose length was announced before the season is not a hazard to be
    sampled -- the games are gone with certainty -- so it is subtracted from the
    exposure the hurdle draws against rather than fed to the regression as a
    covariate. The denominator is deliberately *not* changed with it: the share
    layers divide by ``team_games`` to get per-team-game role intensities, and a
    suspended player is the same player per game he plays. Shrinking both would
    hold his season totals flat, which is the opposite of the intended effect.

    Defaults to zero, so a frame without the column reproduces the pre-existing
    behaviour exactly.
    """
    suspended = pd.to_numeric(
        rows.get("suspended_games", pd.Series(0.0, index=rows.index)),
        errors="coerce",
    ).fillna(0.0).to_numpy(dtype=float)
    if (suspended < 0).any():
        raise ValueError("suspended_games must be nonnegative")
    return np.clip(team_games - np.rint(suspended).astype(int), 0, None)


@dataclass
class AvailabilityPrediction:
    rows: pd.DataFrame
    probability: np.ndarray
    games_active: np.ndarray
    availability: np.ndarray


@dataclass
class SeasonAvailabilityModel:
    """Hurdle model for playing at all and games active conditional on playing."""

    positions: list[str] = field(default_factory=lambda: list(MODEL_POSITIONS))
    # Which exposure the model is projecting.
    #
    # ``games`` is roster-active weeks. For a drafted player that is within
    # about a game of the truth, but for an undrafted one it measures
    # employment rather than participation: over 2022-2025 an undrafted
    # quarterback is on the roster 11.31 weeks and takes an offensive snap in
    # 4.36. Fitting one regression to a label that means two different things
    # for two halves of the population is part of why the layer could not tell
    # the halves apart.
    #
    # ``snap_games`` counts weeks with at least one offensive snap. The obvious
    # third option, ``stat_activity_games``, is worse than either: it counts a
    # game only if the player recorded a stat, so a blocking tight end
    # registers nothing -- undrafted tight ends average 6.23 stat-line games
    # against 10.87 with a snap.
    #
    # Changing this alone is incoherent. ``SeasonSnapShareModel`` divides the
    # observed snap share by this same exposure to get a conditional rate, so
    # the two must be set together; ``SeasonAverageVolumePipeline`` owns that
    # pairing and no caller should set one without the other.
    games_column: str = "games"
    # Let each position have its own slope vector, drawn around a shared mean.
    #
    # The layer carries position-specific *intercepts* and one shared slope
    # vector. Receivers are 37.9% of all training rows and 60.7% undrafted, so
    # the shared slope is fitted largely on fringe receivers -- which is the
    # standing explanation for why they keep about half their drafted/undrafted
    # shrinkage after the board was added.
    #
    # Additive position dummies are the obvious encoding and are the wrong one
    # here. That is how the role-layer interaction arm was built and it lost on
    # every holdout, because the dummies are collinear with the main effects and
    # a shared feature prior cannot hold the opposing coefficients that
    # requires. A hierarchy has no such problem: each position gets its own
    # vector, non-centred, with a half-normal on how far positions may drift
    # from the common mean. If they do not differ the scale shrinks toward zero
    # and this collapses back to the current model, so it is a generalisation
    # rather than a different model.
    #
    # It has to apply to every slope, not just the market ones. ``_matrix``
    # projects the design onto its SVD basis, so after projection a column is a
    # rotation of all the features and there is no "ADP coefficient" to single
    # out.
    position_varying_slopes: bool = False
    # How far a position's slopes may drift from the shared mean. Deliberately
    # tighter than the slope priors themselves (0.40 and 0.35): the claim being
    # entertained is that positions differ somewhat, not that they are unrelated.
    position_slope_scale: float = 0.15
    # New covariates remain opt-in until a multi-fold posterior validation
    # clears the model-promotion gate. See ``validate_injury_availability.py``.
    extra_features: tuple[str, ...] = ()
    position_specific_concentration: bool = False
    feature_names: list[str] = field(default_factory=list)
    feature_fill: dict[str, float] = field(default_factory=dict)
    feature_mean: dict[str, float] = field(default_factory=dict)
    feature_scale: dict[str, float] = field(default_factory=dict)
    feature_projection: np.ndarray | None = None
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
            candidates = tuple(
                dict.fromkeys((*AVAILABILITY_FEATURES, *self.extra_features))
            )
            self.feature_names = [
                name
                for name in candidates
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

    def _slopes(self, pm, name: str, prior: float, features: int):
        """One shared slope vector, or one per position drawn around a shared mean.

        Non-centred on purpose. A centred hierarchy with few positions and a
        small between-position scale is the classic funnel, and this sampler is
        already run at target_accept 0.92 because the layer is not easy.
        """
        if not self.position_varying_slopes:
            return pm.Normal(f"{name}_beta", 0.0, prior, shape=features)
        mean = pm.Normal(f"{name}_beta_mu", 0.0, prior, shape=features)
        scale = pm.HalfNormal(f"{name}_beta_sd", self.position_slope_scale)
        offset = pm.Normal(
            f"{name}_beta_z", 0.0, 1.0, shape=(len(self.positions), features)
        )
        return pm.Deterministic(f"{name}_beta", mean + scale * offset)

    def _linear(self, pm, matrix, beta, position_index):
        if not self.position_varying_slopes:
            return pm.math.sum(matrix * beta, axis=1)
        return pm.math.sum(matrix * beta[position_index], axis=1)

    def fit(self, rows: pd.DataFrame, **sample_kwargs) -> "SeasonAvailabilityModel":
        import pymc as pm

        out = self._prepare(rows)
        if not {self.games_column, "team_games"} <= set(out.columns):
            raise ValueError(
                f"availability fitting requires {self.games_column} and "
                "team_games"
            )
        games_total = pd.to_numeric(out["team_games"], errors="coerce").fillna(0)
        games_active = pd.to_numeric(out[self.games_column], errors="coerce").fillna(0)
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
        played = y > 0
        any_center = float(
            logit(np.array([np.clip(played.mean(), 0.05, 0.98)]))[0]
        )
        conditional = played & (n > 1)
        remaining_n = n[conditional] - 1
        remaining_y = y[conditional] - 1
        conditional_rate = np.clip(
            remaining_y.sum() / max(remaining_n.sum(), 1), 0.05, 0.98
        )
        rate_center = float(logit(np.array([conditional_rate]))[0])
        with pm.Model() as model:
            any_intercept = pm.Normal("any_intercept", any_center, 0.60)
            any_position_effect = _position_effect(
                pm, "any_position_effect", 0.40, len(self.positions)
            )
            any_beta = self._slopes(pm, "any", 0.40, X.shape[1])
            any_eta = any_intercept + any_position_effect[position_index]
            any_eta = any_eta + self._linear(pm, X, any_beta, position_index)
            pm.Bernoulli(
                "played_obs", p=pm.math.sigmoid(any_eta), observed=played.astype(int)
            )

            rate_intercept = pm.Normal("rate_intercept", rate_center, 0.50)
            rate_position_effect = _position_effect(
                pm, "rate_position_effect", 0.35, len(self.positions)
            )
            rate_beta = self._slopes(pm, "rate", 0.35, X.shape[1])
            concentration_shape = (
                len(self.positions) if self.position_specific_concentration else None
            )
            rate_concentration = pm.Gamma(
                "rate_concentration",
                alpha=3.0,
                beta=0.12,
                shape=concentration_shape,
            )
            rate_eta = (
                rate_intercept + rate_position_effect[position_index[conditional]]
            )
            rate_eta = rate_eta + self._linear(
                pm, X[conditional], rate_beta, position_index[conditional]
            )
            rate_probability = pm.math.sigmoid(rate_eta)
            conditional_concentration = (
                rate_concentration[position_index[conditional]]
                if self.position_specific_concentration
                else rate_concentration
            )
            pm.BetaBinomial(
                "conditional_games_obs",
                n=remaining_n,
                alpha=rate_probability * conditional_concentration,
                beta=(1.0 - rate_probability) * conditional_concentration,
                observed=remaining_y,
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
        # Pipelines saved before the hurdle-model upgrade contain the original
        # single-stage Beta-Binomial posterior. Keep those artifacts loadable
        # so accepted historical comparisons remain exactly reproducible.
        if "any_intercept" not in post:
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
            eligible = _eligible_games(out, team_games)
            games_active = rng.binomial(eligible[:, None], probability)
            availability = games_active / team_games[:, None]
            return AvailabilityPrediction(
                rows=out,
                probability=probability,
                games_active=games_active,
                availability=availability,
            )
        any_eta = _stack(post, "any_intercept")[None, :]
        any_eta = any_eta + _stack(post, "any_position_effect")[position_index, :]
        any_eta = any_eta + _apply_slopes(
            _stack(post, "any_beta"), X, position_index
        )
        any_probability = 1.0 / (
            1.0 + np.exp(-np.clip(any_eta, -20.0, 20.0))
        )
        rate_eta = _stack(post, "rate_intercept")[None, :]
        rate_eta = rate_eta + _stack(post, "rate_position_effect")[position_index, :]
        rate_eta = rate_eta + _apply_slopes(
            _stack(post, "rate_beta"), X, position_index
        )
        rate_mean = 1.0 / (
            1.0 + np.exp(-np.clip(rate_eta, -20.0, 20.0))
        )
        concentration = _stack(post, "rate_concentration")
        if concentration.ndim == 1:
            concentration = concentration[None, :]
        else:
            concentration = concentration[position_index, :]
        rng = np.random.default_rng(seed)
        conditional_probability = rng.beta(
            np.clip(rate_mean * concentration, 1e-4, None),
            np.clip((1.0 - rate_mean) * concentration, 1e-4, None),
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
        eligible = _eligible_games(out, team_games)
        # A ban covering the whole season leaves no game for the hurdle to
        # clear, so the "plays at all" draw is forced to zero rather than left
        # to grant him the guaranteed first appearance the hurdle assumes.
        played = rng.binomial(1, any_probability) * (eligible > 0)[:, None]
        remaining_games = rng.binomial(
            np.maximum(eligible - 1, 0)[:, None], conditional_probability
        )
        games_active = played * (1 + remaining_games)
        expected_games = any_probability * (
            1.0
            + np.maximum(eligible - 1, 0)[:, None] * conditional_probability
        ) * (eligible > 0)[:, None]
        probability = expected_games / team_games[:, None]
        availability = games_active / team_games[:, None]
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
    """Season QB passing share within each team roster, gated by a hurdle.

    The room is close to winner-take-all, so a backup's realized share is a
    mixture: nothing in most seasons, a substantial share when the starter goes
    down. A softmax alone cannot represent that, because it never emits an exact
    zero. ``hurdle_min_attempts`` sets how much workload counts as taking a
    meaningful share when labelling the gate during fitting.

    The two halves are a factorisation of one distribution, not two independent
    models: the softmax is fit over the gated-in room and is therefore the share
    *conditional* on clearing the hurdle, and multiplying it by the gate at
    prediction time recovers the marginal. Fitting the softmax unconditionally
    and then gating it would count the zeros twice and pull expected share away
    from backups toward starters.

    ``hurdle_min_attempts`` is compared against ``_observed_counts``, which is
    offensive snaps where snap counts exist and pass attempts otherwise, so the
    threshold is not a pure attempt count. Both are small-workload bars, but
    putting the gate on one basis is worth revisiting.
    """

    feature_names: list[str] = field(default_factory=list)
    feature_fill: dict[str, float] = field(default_factory=dict)
    feature_mean: dict[str, float] = field(default_factory=dict)
    feature_scale: dict[str, float] = field(default_factory=dict)
    extra_features: tuple[str, ...] = ()
    role_innovation_scale: float = 0.60
    # The innovation is meant to widen the room, not to reallocate it. Softmax
    # renormalization makes it do both unless corrected: see
    # ``mean_preserving_shares``. Quarterback rooms are the most concentrated
    # simplex in the pipeline, so this is where the leakage is largest.
    #
    # Rejected by the walk-forward 2026-08-02 and left off: pass-attempt CRPS
    # +4.30% and workload-share CRPS +5.42%, against a 0.5% allowance on both.
    # The reason is a second defect in the same three lines —
    # ``role_innovation_scale`` is measured as *realized* log-share dispersion
    # and then applied on the input side of the softmax, which compresses it by
    # 0.70-0.93x depending on room shape. Correcting the mean while the scale is
    # wrong makes the distribution worse. See docs/role-innovation-2026-08.md.
    mean_preserving_innovation: bool = False
    # ``role_innovation_scale`` is measured as *realized* log-share dispersion
    # and then applied on the input side of the softmax, which compresses it by
    # about 18% for a three-deep room. See ``calibrate_innovation_scale``.
    #
    # Promoted 2026-08-03 with a documented exception to the acceptance gate.
    # Quarterback rooms average 2.37 gated-in passers, so this layer realized
    # only 76% of the churn it measured, and its 80% intervals covered 0.647,
    # 0.619 and 0.726 of outcomes. Calibration moves those to 0.824, 0.774 and
    # 0.881, improving on every fold, with CRPS flat (+0.60% workload, +0.27%
    # pass). It costs 1.02% workload MAE and 0.91% pass MAE, over the 0.5%
    # protected allowance, and the owner accepted that trade explicitly: this
    # package exists to publish distributions, and nine points of coverage on
    # its worst-calibrated layer is worth a percent of point accuracy.
    calibrated_innovation: bool = True
    innovation_calibration_seed: int = 0
    hurdle_min_attempts: int = 25
    # Availability reaches this layer twice — as the softmax's exposure offset
    # and again through the gate. Drawing the two independently lets one draw
    # pair "available all season" with a closed gate; this makes the gate a
    # function of the same availability value the offset uses, so the pair moves
    # together within a draw.
    #
    # Promoted 2026-08-02 on the 1000/1000/4 walk-forward: CRPS improves 4.06%
    # on pass attempts and 5.16% on workload share, winning all three holdouts
    # on both, with coverage up at 80% and 95% on both and CRPS wins on all
    # three scoring systems downstream. It carries one documented exception to
    # the acceptance rule — workload-share MAE wins one holdout of three rather
    # than two, with both losses on 2023 — accepted deliberately because this is
    # a distributional model and every distributional metric is unanimous. See
    # docs/volume-fix-validation-2026-08.md.
    couple_gate_to_availability: bool = True
    hurdle_availability_mean: float = 0.0
    hurdle_availability_scale: float = 1.0
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
                for name in tuple(dict.fromkeys((*QB_WORKLOAD_FEATURES, *self.extra_features)))
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
        # ``snap_counts_observed`` answers "does this *team-season* have snap
        # coverage", not "were this player's snaps measured" — ``_merge_snap_usage``
        # sets it from the team's presence in the snap feed and zero-fills the
        # players the feed omits. Reading it as a per-player flag makes every
        # omitted passer look like a measured zero, so the pass-attempt fallback
        # never fires and the whole room can collapse to an all-zero response.
        # Require the player's own positive snap count before preferring snaps.
        observed = pd.to_numeric(
            quarterbacks.get(
                "snap_counts_observed", pd.Series(0, index=quarterbacks.index)
            ),
            errors="coerce",
        ).fillna(0).gt(0) & snaps.fillna(0).gt(0)
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
        support: np.ndarray | None = None,
    ) -> float:
        """RMS log-share error of the *conditional* room, on training rows only.

        ``support`` marks the quarterbacks the softmax is fit over. Passers who
        did not clear the hurdle are excluded, because their near-zero share is
        represented by the gate rather than by this dispersion term — leaving
        them in would charge the same zero-inflation twice.
        """
        availability = pd.to_numeric(
            quarterbacks.get(
                "observed_availability", pd.Series(1.0, index=quarterbacks.index)
            ),
            errors="coerce",
        ).fillna(1.0).clip(0.03, 1.0).to_numpy(dtype=float)
        if support is None:
            support = np.ones(len(quarterbacks), dtype=bool)
        residuals = []
        for _, group in quarterbacks.groupby(GROUP_KEYS, sort=False, dropna=False):
            indices = group.index.to_numpy(dtype=int)
            indices = indices[support[indices]]
            if len(indices) < 2 or counts[indices].sum() <= 0:
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

    def _calibrated_innovation(
        self,
        quarterbacks: pd.DataFrame,
        role: np.ndarray,
        support: np.ndarray,
        target: float,
    ) -> float:
        """Input scale whose realized log-share spread matches ``target``.

        Quarterback rooms are the smallest simplex the pipeline allocates over,
        which is where softmax renormalization eats the most dispersion — about
        18% for a three-deep room against 8% for a seven-deep target room. This
        is therefore the layer where the uncalibrated scale hurts most, and it
        matches the coverage: workload share sits 6 to 18 points under its 80%
        nominal on every holdout.
        """
        availability = pd.to_numeric(
            quarterbacks.get(
                "observed_availability", pd.Series(1.0, index=quarterbacks.index)
            ),
            errors="coerce",
        ).fillna(1.0).clip(0.03, 1.0).to_numpy(dtype=float)
        rooms = []
        for _, group in quarterbacks.groupby(GROUP_KEYS, sort=False, dropna=False):
            indices = group.index.to_numpy(dtype=int)
            indices = indices[support[indices]]
            if len(indices) > 1:
                weights = role[indices] * availability[indices]
                rooms.append(weights / weights.sum())
        if not rooms:
            return target
        width = max(len(room) for room in rooms)
        allocation = np.zeros((len(rooms), width), dtype=float)
        mask = np.zeros((len(rooms), width), dtype=bool)
        for row, room in enumerate(rooms):
            allocation[row, : len(room)] = room
            mask[row, : len(room)] = True
        return calibrate_innovation_scale(
            allocation, mask, target, seed=self.innovation_calibration_seed
        )

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
            support = counts_flat >= self.hurdle_min_attempts
            target = self._estimate_role_innovation(
                quarterbacks, counts_flat, role, support=support
            )
            self.role_innovation_scale = (
                self._calibrated_innovation(quarterbacks, role, support, target)
                if self.calibrated_innovation
                else target
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

    def _conditional_design(self, design: dict) -> dict:
        """Split one room design into its gated-in share fit and its hurdle fit.

        The softmax is the share *conditional on* clearing the hurdle, so it is
        fit over the gated-in room only. Fitting it over every quarterback would
        make its mean the marginal share; gating that same mean at prediction
        time and renormalising would then shift it, scoring a backup who takes a
        large share in the seasons he plays as if he took a small share every
        season. Rooms where nobody cleared the gate carry no conditional
        information and drop out of the share fit, but still label the hurdle.
        """
        active = design["mask"] > 0
        played = (design["counts"] >= self.hurdle_min_attempts) & active
        graded = played.any(axis=1)
        return {
            "support": played[graded],
            "counts": np.where(played, design["counts"], 0)[graded],
            "role_offset": design["role_offset"][graded],
            "availability_offset": design["availability_offset"][graded],
            "X": design["X"][graded],
            "hurdle_X": design["X"][active],
            "hurdle_y": played[active].astype(float),
            # Log availability, the same quantity the softmax uses as its
            # exposure offset, so an enabled coupling ties the two together.
            "hurdle_availability": design["availability_offset"][active],
        }

    def fit(self, rows: pd.DataFrame, **sample_kwargs) -> "QBWorkloadShareModel":
        import pymc as pm

        design = self._design(rows, fit=True)
        if design["counts"].size == 0:
            raise ValueError(
                "QB workload fitting found no room with a positive workload "
                "response; check that offense_snaps or pass_att are populated"
            )
        conditional = self._conditional_design(design)
        share_counts = conditional["counts"]
        hurdle_X = conditional["hurdle_X"]
        hurdle_y = conditional["hurdle_y"]
        with pm.Model() as model:
            beta = pm.Normal("beta", 0.0, 0.50, shape=len(self.feature_names))
            eta = conditional["role_offset"] + conditional["availability_offset"]
            eta = eta + pm.math.sum(conditional["X"] * beta, axis=2)
            eta = pm.math.switch(conditional["support"], eta, -20.0)
            probability = pm.math.softmax(eta, axis=1)
            pm.Multinomial(
                "workload_obs",
                n=share_counts.sum(axis=1),
                p=probability,
                observed=share_counts,
            )
            # Whether a quarterback takes a meaningful share at all. The softmax
            # above cannot emit an exact zero, so without this every backup gets
            # a unimodal distribution centred between the two outcomes that
            # actually occur: nothing in most seasons, a large share when the
            # starter goes down.
            hurdle_intercept = pm.Normal(
                "hurdle_intercept", float(_logit_scalar(hurdle_y.mean())), 1.0
            )
            hurdle_beta = pm.Normal(
                "hurdle_beta", 0.0, 0.75, shape=len(self.feature_names)
            )
            hurdle_eta = hurdle_intercept + pm.math.dot(hurdle_X, hurdle_beta)
            if self.couple_gate_to_availability:
                scaled = self._scaled_hurdle_availability(
                    conditional["hurdle_availability"], fit=True
                )
                hurdle_availability_beta = pm.Normal(
                    "hurdle_availability_beta", 0.0, 1.0
                )
                hurdle_eta = hurdle_eta + hurdle_availability_beta * scaled
            pm.Bernoulli("played_obs", logit_p=hurdle_eta, observed=hurdle_y)
            sample_kwargs.setdefault("target_accept", 0.92)
            self.idata = sample_model(model, **sample_kwargs)
        return self

    def _scaled_hurdle_availability(
        self, values: np.ndarray, *, fit: bool = False
    ) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        if fit:
            self.hurdle_availability_mean = float(values.mean())
            scale = float(values.std())
            self.hurdle_availability_scale = scale if scale > 1e-8 else 1.0
        return (values - self.hurdle_availability_mean) / self.hurdle_availability_scale

    def _hurdle_gate(self, design, rng, log_availability=None) -> np.ndarray | None:
        """Per-draw Bernoulli gate over the room, or None for legacy posteriors.

        Gating is what turns the conditional softmax fitted above back into a
        marginal share, so the two must stay paired: a posterior without
        ``hurdle_beta`` was fitted unconditionally and must not be gated.

        ``log_availability`` carries the per-draw exposure the softmax is using.
        When the coupling is enabled the gate reads it, so within one draw a
        passer who is available is also more likely to clear the hurdle instead
        of the two being sampled independently.
        """
        posterior = self.idata.posterior
        if "hurdle_beta" not in posterior:
            return None
        intercept = _stack(posterior, "hurdle_intercept")
        hurdle_beta = _stack(posterior, "hurdle_beta")
        eta = intercept[None, None, :] + np.einsum(
            "gkf,fs->gks", design["X"], hurdle_beta
        )
        if "hurdle_availability_beta" in posterior:
            if log_availability is None:
                log_availability = np.repeat(
                    design["availability_offset"][..., None], eta.shape[-1], axis=2
                )
            eta = eta + self._scaled_hurdle_availability(log_availability) * _stack(
                posterior, "hurdle_availability_beta"
            )
        probability = 1.0 / (1.0 + np.exp(-eta))
        gate = rng.random(probability.shape) < probability
        gate &= design["mask"][..., None] > 0
        # A room with nobody throwing is not a possible season, so the most
        # likely passer is retained wherever every gate closed on a draw.
        empty = ~gate.any(axis=1)
        if empty.any():
            fallback = np.where(design["mask"][..., None] > 0, probability, -np.inf)
            best = fallback.argmax(axis=1)
            groups, samples = np.nonzero(empty)
            gate[groups, best[groups, samples], samples] = True
        return gate

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
        log_availability = None
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
            log_availability = np.log(np.clip(availability, 0.03, 1.0))
            eta = eta + log_availability
        eta = eta + np.einsum("gkf,fs->gks", design["X"], beta)
        rng = np.random.default_rng(seed)
        innovation = rng.normal(size=eta.shape) * self.role_innovation_scale
        gate = self._hurdle_gate(design, rng, log_availability)
        live = np.broadcast_to(design["mask"][..., None] > 0, eta.shape)
        if gate is not None:
            live = live & gate
        live = np.ascontiguousarray(live)
        if self.mean_preserving_innovation:
            probability = mean_preserving_shares(eta, eta + innovation, live)
        else:
            probability = simplex_shares(eta + innovation, live)
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
