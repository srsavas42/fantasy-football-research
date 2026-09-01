"""Season-wide per-game volume models.

This module estimates stable team-season rates and roster-level player roles.
It intentionally omits matchup covariates: weekly observations are used to
construct season counts, while the latent prediction target is average volume
per game over a full season.

The player likelihood is roster coherent. Quarterback pass attempts combine a
continuous QB offensive-snap workload simplex with attempts per snap. Targets
and carries combine projected snap exposure with lagged per-snap propensity;
carries add a sparse any-carry hurdle. Position-level replacement buckets
preserve point-in-time roster accounting for later signings and call-ups.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from ffmodel.features.market import ADP_FEATURES, ADP_INTERACTION_FEATURES
from ffmodel.features.season_injury import INJURY_AVAILABILITY_FEATURES
from ffmodel.features.season_average import (
    POSTSEASON_FEATURES,
    SeasonAverageData,
    TEAM_KEYS,
)
from ffmodel.features.volume import MODEL_POSITIONS
from ffmodel.models.base import (
    calibrate_innovation_scale,
    load_idata,
    logit,
    mean_preserving_shares,
    sample_model,
    sampling_quality,
    save_idata,
    simplex_shares,
)
from ffmodel.models.season_availability import (
    RESERVE_KIND_FEATURES,
    AVAILABILITY_HISTORY_FEATURES,
    AvailabilityPrediction,
    QBWorkloadShareModel,
    SeasonAvailabilityModel,
)
from ffmodel.models.season_opportunity import (
    QBPassPropensityModel,
    SeasonCarryEligibilityModel,
    SeasonSnapShareModel,
    SeasonTargetRoleModel,
)
from ffmodel.models.season_regime import (
    REGIME_LIKELIHOOD_FEATURES,
    SeasonRegimeModel,
    SeasonRegimePrediction,
    add_regime_probabilities,
    add_walk_forward_regime_probabilities,
)
from ffmodel.models.season_regime_coupling import SeasonRegimeRoleCoupling
from ffmodel.models.volume_team import _sum_to_zero_basis

GROUP_KEYS = ["season", "team"]

STREAMS = {
    "pass": {
        "count": "pass_att",
        "role": "prior_pass_role",
        "per_snap_role": "prior_qb_attempts_per_snap",
        "draft": "draft_pass_prior",
        "fallback": {"QB": 0.800, "RB": 0.0005, "WR": 0.0005, "TE": 0.0002},
    },
    "target": {
        "count": "targets",
        "role": "prior_target_role",
        "per_snap_role": "prior_target_per_snap",
        "draft": "draft_target_prior",
        "fallback": {"QB": 0.0001, "RB": 0.120, "WR": 0.180, "TE": 0.150},
    },
    "carry": {
        "count": "rush_att",
        "role": "prior_carry_role",
        "per_snap_role": "prior_carry_per_snap",
        "draft": "draft_carry_prior",
        "fallback": {"QB": 0.080, "RB": 0.450, "WR": 0.010, "TE": 0.003},
    },
}

DEFAULT_SACK_RATE = 0.065

BASE_ADJUSTMENT_FEATURES = (
    "prior_availability",
    "prior_snap_share",
    "age",
    "experience",
    "team_change",
    "cold_start",
)

# Efficiency is stream-specific: receiving performance can inform future target
# allocation without being allowed to distort QB or carry allocation, and vice
# versa. All values are pooled observations from Y-1.
VOLUME_EFFICIENCY_FEATURES = {
    "pass": (
        "prior_pass_yards_per_attempt",
        "prior_pass_epa_per_attempt",
        "prior_pass_completion_rate",
        "prior_pass_first_down_rate",
        "prior_pass_td_rate",
        "prior_pass_quality_rank",
        "prior_pass_quality_signal",
    ),
    "target": (
        "prior_rec_yards_per_target",
        "prior_rec_epa_per_target",
        "prior_rec_air_yards_per_target",
        "prior_rec_first_down_rate",
        "prior_rec_quality_rank",
        "prior_rec_quality_signal",
    ),
    "carry": (
        "prior_rush_yards_per_carry",
        "prior_rush_epa_per_carry",
        "prior_rush_first_down_rate",
        "prior_rush_td_rate",
        "prior_rush_quality_rank",
        "prior_rush_quality_signal",
    ),
}

# The direct-share challenger benefits from QB and rushing efficiency. The
# production volume-v2 path uses the target/carry share models below but routes
# QB volume through separate workload and attempts-per-snap layers.
DIRECT_SHARE_EFFICIENCY_FEATURES = {
    "pass": ("prior_pass_quality_signal", "prior_pass_td_rate"),
    "target": (),
    "carry": ("prior_rush_epa_per_carry",),
}

# Production gate: neither the QB-layer proxy nor the posterior-controlled
# carry comparison transferred the direct-share gains. Keep every efficiency
# input challenger-only until it improves the accepted volume-v2 architecture.
ACCEPTED_VOLUME_EFFICIENCY_FEATURES = {
    "pass": (),
    "target": (),
    "carry": (),
}

PARTICIPATION_VOLUME_FEATURES = {
    "pass": (),
    "target": ("prior_targets_per_pass_play",),
    "carry": (),
}

# Backwards-compatible union used by callers that inspect the available
# preseason adjustment contract. Models select the relevant stream subset via
# ``volume_adjustment_features``.
ADJUSTMENT_FEATURES = BASE_ADJUSTMENT_FEATURES + tuple(
    dict.fromkeys(
        feature
        for features in VOLUME_EFFICIENCY_FEATURES.values()
        for feature in features
    )
)


def volume_adjustment_features(
    stream: str, *, include_experimental: bool = False
) -> tuple[str, ...]:
    if stream not in STREAMS:
        raise ValueError(f"stream must be one of {sorted(STREAMS)}")
    efficiency = (
        VOLUME_EFFICIENCY_FEATURES[stream]
        if include_experimental
        else DIRECT_SHARE_EFFICIENCY_FEATURES[stream]
    )
    participation = PARTICIPATION_VOLUME_FEATURES[stream] if include_experimental else ()
    return BASE_ADJUSTMENT_FEATURES + efficiency + participation

# The Bayesian posterior retains only effects that were stable and materially
# identified in the initial full-feature walk-forward fit. Richer nonlinear
# interactions remain the job of the optional XGBoost challenger.
BAYESIAN_FEATURES = (
    "prior_availability",
    "prior_snap_share",
)

TARGET_BAYESIAN_FEATURES = ("prior_availability",)

BACKUP_QB_EXPOSURE = 0.08

# Below this many rows on either side of the split, the cold-start dispersion
# ratio is estimated from too little data to act on, and the model falls back to
# a single innovation scale.
MIN_COLD_ROLE_ROWS = 200


def _stack(posterior, name: str) -> np.ndarray:
    return posterior[name].stack(sample=("chain", "draw")).to_numpy()


def _codes(values: pd.Series, categories: list[str]) -> np.ndarray:
    lookup = {value: index for index, value in enumerate(categories)}
    return values.astype(str).map(lookup).fillna(-1).to_numpy(dtype=int)


@dataclass
class TeamSeasonAverageModel:
    """Joint team opportunity/pass/sack/target model at team-season grain."""

    teams: list[str] = field(default_factory=list)
    season_mean: float = 2020.0
    play_prior_center: float = 4.1
    pass_prior_center: float = 0.3
    sack_prior_center: float = -2.67
    target_prior_center: float = 2.6
    models_sacks: bool = False
    # plays_obs is NegativeBinomial, a Poisson-Gamma mixture that already carries
    # overdispersion. A per-row log-normal term on the same mean is a second
    # dispersion source on one likelihood, and the two are not separately
    # identified: the variance posterior collapses to 0.008 with r-hat 1.02 and
    # drags the intercept's effective sample size down with it. The pass and
    # target likelihoods are Binomial, which has no free dispersion parameter, so
    # their transition terms are doing real work and are unconditional.
    models_play_transition: bool = False
    # Each stream's intercept and its 32 team effects are additively confounded:
    # a shift shared across all teams is indistinguishable from a shift in the
    # intercept, and ``Normal(0, small)`` identifies that only softly, through
    # the prior. The pair mixes slowly as a result. Constraining the effects to
    # sum to zero removes the direction entirely, which is what
    # ``_position_effect`` already does for position effects elsewhere.
    sum_to_zero_team_effects: bool = True
    # The market's preseason opinion on each team, as a within-season z-score of
    # its win total. This layer has never had a non-play-by-play input, and a
    # win total is the one market signal that is about *teams* rather than
    # players, so it is the natural place to ask whether the book knows
    # something the history does not.
    #
    # Standardized within season on purpose -- see add_team_win_totals. The raw
    # line drifts with the schedule expansion and would partly duplicate the
    # era term already in every stream.
    #
    # Off until measured.
    market_features: bool = False
    market_feature_scale: float = 0.15
    idata: object = None

    def _market(self, rows: pd.DataFrame) -> np.ndarray:
        """The market covariate, or zeros when the arm is off.

        Absent columns raise rather than defaulting to zero: a silent zero is a
        model that fits the baseline while reporting itself as the candidate,
        which has happened twice on this branch with other features.
        """
        if not self.market_features:
            return np.zeros(len(rows), dtype=float)
        if "market_win_total" not in rows:
            raise ValueError(
                "market_features is on but market_win_total is absent from the "
                "team rows; rebuild the cache rather than fitting a model that "
                "would silently drop it and report the baseline as a null"
            )
        values = pd.to_numeric(rows["market_win_total"], errors="coerce")
        if values.isna().any():
            raise ValueError(
                "market_win_total has missing values; a hole in a team-level "
                "covariate becomes a missingness pattern the model reads as "
                "information about those teams"
            )
        return values.to_numpy(dtype=float)

    def _team_effect(self, pm, name: str, scale: float):
        size = len(self.teams)
        if not self.sum_to_zero_team_effects or size < 2:
            return pm.Normal(name, 0.0, scale, shape=size)
        raw = pm.Normal(f"{name}_raw", 0.0, scale, shape=size - 1)
        return pm.Deterministic(name, pm.math.dot(_sum_to_zero_basis(size), raw))

    def _design(self, rows: pd.DataFrame, *, fit: bool = False):
        required = {
            "season",
            "team",
            "prior_opportunity_plays_per_game",
            "prior_pass_rate",
            "prior_target_rate",
        }
        missing = required - set(rows.columns)
        if missing:
            raise ValueError(f"team-season rows are missing columns: {sorted(missing)}")
        d = rows.copy().sort_values(TEAM_KEYS).reset_index(drop=True)
        if fit:
            self.teams = sorted(d["team"].astype(str).unique())
            self.season_mean = float(d["season"].mean())
            self.play_prior_center = float(
                np.log(
                    np.clip(
                        d["prior_opportunity_plays_per_game"].mean(), 1.0, None
                    )
                )
            )
            self.pass_prior_center = float(
                logit(np.array([d["prior_pass_rate"].mean()]))[0]
            )
            self.target_prior_center = float(
                logit(np.array([d["prior_target_rate"].mean()]))[0]
            )
            prior_sack = pd.to_numeric(
                d.get("prior_sack_rate", pd.Series(np.nan, index=d.index)),
                errors="coerce",
            )
            if prior_sack.notna().any():
                self.sack_prior_center = float(
                    logit(np.array([prior_sack.dropna().mean()]))[0]
                )
        team_idx = _codes(d["team"], self.teams)
        era = (d["season"].to_numpy(dtype=float) - self.season_mean) / 5.0
        play_prior = np.log(
            np.clip(
                d["prior_opportunity_plays_per_game"].to_numpy(dtype=float),
                1.0,
                None,
            )
        ) - self.play_prior_center
        pass_prior = logit(d["prior_pass_rate"].to_numpy(dtype=float)) - self.pass_prior_center
        target_prior = logit(d["prior_target_rate"].to_numpy(dtype=float)) - self.target_prior_center
        prior_sack = pd.to_numeric(
            d.get("prior_sack_rate", pd.Series(np.nan, index=d.index)),
            errors="coerce",
        ).fillna(1.0 / (1.0 + np.exp(-self.sack_prior_center)))
        sack_prior = logit(prior_sack.to_numpy(dtype=float)) - self.sack_prior_center
        market = self._market(d)
        return (
            d, team_idx, era, play_prior, pass_prior, sack_prior, target_prior, market
        )

    def fit(self, rows: pd.DataFrame, **sample_kwargs) -> "TeamSeasonAverageModel":
        """Fit season totals with prior-season rates as strictly lagged inputs."""
        import pymc as pm

        (
            d,
            team_idx,
            era,
            play_prior,
            pass_prior,
            sack_prior,
            target_prior,
            market,
        ) = self._design(rows, fit=True)
        games = d["games"].to_numpy(dtype=int)
        opportunity_plays = d["opportunity_plays"].to_numpy(dtype=int)
        passes = d["pass_attempts"].to_numpy(dtype=int)
        sacks = pd.to_numeric(
            d.get("sacks", pd.Series(0, index=d.index)), errors="coerce"
        ).fillna(0).to_numpy(dtype=int)
        sacks_observed = d.get(
            "sacks_observed", pd.Series(False, index=d.index)
        ).astype(bool).to_numpy()
        dropbacks = passes + sacks
        target_n = pd.to_numeric(
            d.get("valid_target_pass_attempts", d["pass_attempts"]),
            errors="coerce",
        ).fillna(0).to_numpy(dtype=int)
        targets = pd.to_numeric(
            d.get("valid_targets", d["targets"]), errors="coerce"
        ).fillna(0).to_numpy(dtype=int)
        valid_target = (target_n > 0) & (targets <= target_n)
        valid_sack = sacks_observed & (dropbacks > 0) & (sacks <= dropbacks)
        self.models_sacks = bool(valid_sack.any())

        play_center = float(
            np.log(np.clip((opportunity_plays / games).mean(), 1.0, None))
        )
        pass_center = float(
            logit(np.array([passes.sum() / opportunity_plays.sum()]))[0]
        )
        if self.models_sacks:
            self.sack_prior_center = float(
                logit(
                    np.array(
                        [sacks[valid_sack].sum() / dropbacks[valid_sack].sum()]
                    )
                )[0]
            )
            sack_prior = logit(
                pd.to_numeric(
                    d.get("prior_sack_rate", pd.Series(np.nan, index=d.index)),
                    errors="coerce",
                )
                .fillna(1.0 / (1.0 + np.exp(-self.sack_prior_center)))
                .to_numpy(dtype=float)
            ) - self.sack_prior_center
        target_center = float(
            logit(
                np.array(
                    [targets[valid_target].sum() / target_n[valid_target].sum()]
                )
            )[0]
        )

        with pm.Model() as model:
            play_intercept = pm.Normal("play_intercept", play_center, 0.20)
            play_persistence = pm.Normal("play_persistence", 0.75, 0.25)
            play_era = pm.Normal("play_era", 0.0, 0.12)
            play_team = self._team_effect(pm, "play_team", 0.06)
            play_eta = (
                play_intercept
                + play_persistence * play_prior
                + play_era * era
                + play_team[team_idx]
            )
            if self.market_features:
                play_market = pm.Normal(
                    "play_market", 0.0, self.market_feature_scale
                )
                play_eta = play_eta + play_market * market
            if self.models_play_transition:
                play_transition_sd = pm.HalfNormal("play_transition_sd", 0.12)
                play_transition_z = pm.Normal(
                    "play_transition_z", 0.0, 1.0, shape=len(d)
                )
                play_eta = play_eta + play_transition_z * play_transition_sd
            play_mu_pg = pm.math.exp(play_eta)
            play_alpha_pg = pm.Gamma("play_alpha_pg", alpha=3.0, beta=0.15)
            pm.NegativeBinomial(
                "plays_obs",
                mu=play_mu_pg * games,
                alpha=play_alpha_pg * games,
                observed=opportunity_plays,
            )

            pass_intercept = pm.Normal("pass_intercept", pass_center, 0.30)
            pass_persistence = pm.Normal("pass_persistence", 0.75, 0.25)
            pass_era = pm.Normal("pass_era", 0.0, 0.15)
            pass_team = self._team_effect(pm, "pass_team", 0.12)
            pass_transition_sd = pm.HalfNormal("pass_transition_sd", 0.25)
            pass_transition_z = pm.Normal(
                "pass_transition_z", 0.0, 1.0, shape=len(d)
            )
            pass_eta = (
                pass_intercept
                + pass_persistence * pass_prior
                + pass_era * era
                + pass_team[team_idx]
                + pass_transition_z * pass_transition_sd
            )
            if self.market_features:
                pass_market = pm.Normal(
                    "pass_market", 0.0, self.market_feature_scale
                )
                pass_eta = pass_eta + pass_market * market
            pm.Binomial(
                "passes_obs",
                n=opportunity_plays,
                p=pm.math.sigmoid(pass_eta),
                observed=passes,
            )

            if self.models_sacks:
                sack_intercept = pm.Normal(
                    "sack_intercept", self.sack_prior_center, 0.30
                )
                sack_persistence = pm.Normal("sack_persistence", 0.50, 0.30)
                sack_era = pm.Normal("sack_era", 0.0, 0.15)
                sack_team = self._team_effect(pm, "sack_team", 0.10)
                sack_eta = (
                    sack_intercept
                    + sack_persistence * sack_prior
                    + sack_era * era
                    + sack_team[team_idx]
                )
                if self.market_features:
                    sack_market = pm.Normal(
                        "sack_market", 0.0, self.market_feature_scale
                    )
                    sack_eta = sack_eta + sack_market * market
                pm.Binomial(
                    "sacks_obs",
                    n=dropbacks[valid_sack],
                    p=pm.math.sigmoid(sack_eta[valid_sack]),
                    observed=sacks[valid_sack],
                )

            target_intercept = pm.Normal("target_intercept", target_center, 0.30)
            target_persistence = pm.Normal("target_persistence", 0.65, 0.30)
            target_era = pm.Normal("target_era", 0.0, 0.15)
            target_team = self._team_effect(pm, "target_team", 0.10)
            target_transition_sd = pm.HalfNormal("target_transition_sd", 0.25)
            target_transition_z = pm.Normal(
                "target_transition_z", 0.0, 1.0, shape=len(d)
            )
            target_eta = (
                target_intercept
                + target_persistence * target_prior
                + target_era * era
                + target_team[team_idx]
                + target_transition_z * target_transition_sd
            )
            if self.market_features:
                target_market = pm.Normal(
                    "target_market", 0.0, self.market_feature_scale
                )
                target_eta = target_eta + target_market * market
            pm.Binomial(
                "targets_obs",
                n=target_n[valid_target],
                p=pm.math.sigmoid(target_eta[valid_target]),
                observed=targets[valid_target],
            )
            sample_kwargs.setdefault("target_accept", 0.92)
            self.idata = sample_model(model, **sample_kwargs)
        return self

    def _known_effect(self, posterior, name: str, idx: np.ndarray) -> np.ndarray:
        effect = _stack(posterior, name)
        out = np.zeros((len(idx), effect.shape[-1]), dtype=float)
        known = idx >= 0
        if known.any():
            out[known] = effect[idx[known], :]
        return out

    def predict_average_samples(
        self, rows: pd.DataFrame, *, games=None, seed: int = 0
    ) -> dict[str, object]:
        """Posterior season-average team counts per game."""
        if self.idata is None:
            raise RuntimeError("fit the team-season model before predicting")
        (
            d,
            team_idx,
            era,
            play_prior,
            pass_prior,
            sack_prior,
            target_prior,
            market,
        ) = self._design(rows)
        if games is None:
            games = d.get("games", pd.Series(17, index=d.index)).to_numpy(dtype=int)
        elif np.isscalar(games):
            games = np.full(len(d), int(games), dtype=int)
        else:
            games = np.asarray(games, dtype=int)
        if games.shape != (len(d),) or (games <= 0).any():
            raise ValueError("games must be positive with one value per team-season")

        post = self.idata.posterior
        rng = np.random.default_rng(seed)
        play_eta = _stack(post, "play_intercept")[None, :]
        play_eta = play_eta + (
            play_prior[:, None] * _stack(post, "play_persistence")[None, :]
        )
        play_eta = play_eta + era[:, None] * _stack(post, "play_era")[None, :]
        play_eta = play_eta + self._known_effect(post, "play_team", team_idx)
        # Keyed on the posterior actually containing the coefficient, not on the
        # flag: an artifact fitted without the market term and served with the
        # flag on would otherwise look for a variable that is not there.
        if "play_market" in post:
            play_eta = play_eta + market[:, None] * _stack(
                post, "play_market"
            )[None, :]
        if "play_transition_sd" in post:
            play_eta = play_eta + rng.normal(size=play_eta.shape) * _stack(
                post, "play_transition_sd"
            )[None, :]
        pass_eta = _stack(post, "pass_intercept")[None, :]
        pass_eta = pass_eta + (
            pass_prior[:, None] * _stack(post, "pass_persistence")[None, :]
        )
        pass_eta = pass_eta + era[:, None] * _stack(post, "pass_era")[None, :]
        pass_eta = pass_eta + self._known_effect(post, "pass_team", team_idx)
        # Keyed on the posterior actually containing the coefficient, not on the
        # flag: an artifact fitted without the market term and served with the
        # flag on would otherwise look for a variable that is not there.
        if "pass_market" in post:
            pass_eta = pass_eta + market[:, None] * _stack(
                post, "pass_market"
            )[None, :]
        pass_eta = pass_eta + rng.normal(size=pass_eta.shape) * _stack(
            post, "pass_transition_sd"
        )[None, :]
        sack_eta = np.full_like(pass_eta, self.sack_prior_center)
        if self.models_sacks and "sack_intercept" in post:
            sack_eta = _stack(post, "sack_intercept")[None, :]
            sack_eta = sack_eta + (
                sack_prior[:, None] * _stack(post, "sack_persistence")[None, :]
            )
            sack_eta = sack_eta + era[:, None] * _stack(post, "sack_era")[None, :]
            sack_eta = sack_eta + self._known_effect(post, "sack_team", team_idx)
            # Keyed on the posterior actually containing the coefficient, not on the
            # flag: an artifact fitted without the market term and served with the
            # flag on would otherwise look for a variable that is not there.
            if "sack_market" in post:
                sack_eta = sack_eta + market[:, None] * _stack(
                    post, "sack_market"
                )[None, :]
        target_eta = _stack(post, "target_intercept")[None, :]
        target_eta = target_eta + (
            target_prior[:, None] * _stack(post, "target_persistence")[None, :]
        )
        target_eta = target_eta + era[:, None] * _stack(post, "target_era")[None, :]
        target_eta = target_eta + self._known_effect(post, "target_team", team_idx)
        # Keyed on the posterior actually containing the coefficient, not on the
        # flag: an artifact fitted without the market term and served with the
        # flag on would otherwise look for a variable that is not there.
        if "target_market" in post:
            target_eta = target_eta + market[:, None] * _stack(
                post, "target_market"
            )[None, :]
        target_eta = target_eta + rng.normal(size=target_eta.shape) * _stack(
            post, "target_transition_sd"
        )[None, :]

        mu = np.exp(np.clip(play_eta, -10.0, 10.0)) * games[:, None]
        alpha = _stack(post, "play_alpha_pg")[None, :] * games[:, None]
        probability = alpha / (alpha + mu)
        opportunity_plays = rng.negative_binomial(alpha, probability)
        pass_rate = 1.0 / (1.0 + np.exp(-np.clip(pass_eta, -20.0, 20.0)))
        passes = rng.binomial(opportunity_plays, pass_rate)
        sack_rate = 1.0 / (1.0 + np.exp(-np.clip(sack_eta, -20.0, 20.0)))
        # Conditional on A non-sack attempts, the number of sacks before those
        # attempts is Negative-Binomial with success probability 1-sack_rate.
        sacks = rng.negative_binomial(
            np.maximum(passes, 1), np.clip(1.0 - sack_rate, 1e-5, 1.0)
        )
        sacks = np.where(passes > 0, sacks, 0)
        dropbacks = passes + sacks
        rush_attempts = opportunity_plays - passes
        plays = opportunity_plays + sacks
        target_rate = 1.0 / (1.0 + np.exp(-np.clip(target_eta, -20.0, 20.0)))
        targets = rng.binomial(passes, target_rate)
        no_target_attempts = passes - targets
        divisor = games[:, None]
        return {
            "rows": d,
            "games": games,
            "plays": plays,
            "opportunity_plays": opportunity_plays,
            "pass_attempts": passes,
            "sacks": sacks,
            "dropbacks": dropbacks,
            "targets": targets,
            "no_target_attempts": no_target_attempts,
            "rush_attempts": rush_attempts,
            "plays_per_game": plays / divisor,
            "opportunity_plays_per_game": opportunity_plays / divisor,
            "pass_attempts_per_game": passes / divisor,
            "sacks_per_game": sacks / divisor,
            "dropbacks_per_game": dropbacks / divisor,
            "targets_per_game": targets / divisor,
            "no_target_attempts_per_game": no_target_attempts / divisor,
            "rush_attempts_per_game": rush_attempts / divisor,
        }


@dataclass
class RosterSharePrediction:
    rows: pd.DataFrame
    group_keys: pd.DataFrame
    shares: np.ndarray


@dataclass
class SeasonRosterShareModel:
    """Roster-coherent season pass, target, or carry share model."""

    stream: str = "target"
    extra_efficiency_features: tuple[str, ...] | None = None
    extra_features: tuple[str, ...] = ()
    players: list[str] = field(default_factory=list)
    feature_names: list[str] = field(default_factory=list)
    feature_fill: dict[str, float] = field(default_factory=dict)
    feature_mean: dict[str, float] = field(default_factory=dict)
    feature_scale: dict[str, float] = field(default_factory=dict)
    cold_role_prior: dict[str, float] = field(default_factory=dict)
    availability_prior: dict[str, float] = field(default_factory=dict)
    role_innovation_scale: float = 0.75
    # See ``mean_preserving_shares``: the softmax renormalization turns role
    # churn into a systematic transfer away from whoever leads the room.
    mean_preserving_innovation: bool = False
    # ``role_innovation_scale`` is estimated as *realized* log-share dispersion
    # and then applied on the input side of the softmax, which compresses it.
    # See ``calibrate_innovation_scale``.
    calibrated_innovation: bool = False
    innovation_calibration_seed: int = 0
    per_snap_weight: float | None = None
    innovation_cap: float | None = None
    # A player with no prior role gets the position mean as a point estimate and
    # the same innovation as an established starter. Measured on 2025, 28% of
    # rows with no prior snap share fall outside a 95% interval on total points,
    # 32 of 33 of them above it. ``cold_role_innovation`` gives those rows their
    # own, wider, innovation scale.
    #
    # Promoted 2026-08-04 with mode "measured". In-window over 2022/2023/2024 it
    # takes pooled PPR coverage to nominal at both levels -- 95% from z=+6.62 to
    # +0.17, 80% from +4.33 to +0.56 -- while MAE falls 3.07% and CRPS 1.68%, on
    # all three folds. Confirmed once on 2025: cov95 z +3.99 to -1.10, MAE
    # -3.60%, CRPS -2.03%. See docs/out-of-sample-2025.md.
    cold_role_innovation: bool = True
    # How the cold scale is derived. "relative" keeps the ratio of cold to warm
    # realized dispersion, which preserves the gap between the populations but
    # inherits the cap's compression: the base is capped from 1.94 to 0.25, so a
    # 1.38x ratio lands cold rows at 0.35 against a measured 2.68. "measured"
    # targets the cold population's own dispersion instead, so the cap bounds
    # the typical row without also bounding the row it was never about.
    cold_role_scale_mode: str = "measured"
    cold_role_multiplier: float = 1.0
    # This binds in both modes on real data -- measured mode asks for 2.68 over a
    # base of 0.25 -- so it, not the measurement, sets where cold rows land. It
    # was chosen before any result and has never been selected against folds.
    # Whatever it is worth, it is the least evidenced number in this feature.
    cold_role_multiplier_cap: float = 6.0
    # Allocate week by week instead of multiplying by season-average
    # availability and renormalising once. See ``_per_game_shares`` for what the
    # default approximation costs and who it costs it to. Opt-in: it changes
    # every share a low-availability player takes, so it needs its own
    # walk-forward before it can be the default.
    per_game_allocation: bool = False
    allocation_games: int = 17
    idata: object = None

    def __post_init__(self):
        if self.stream not in STREAMS:
            raise ValueError(f"stream must be one of {sorted(STREAMS)}")
        if self.per_snap_weight is None:
            self.per_snap_weight = 0.75 if self.stream == "target" else 1.0
        if self.innovation_cap is None:
            # Promoted 2026-08-03 at 0.25, chosen on an inner fold. The previous
            # 0.50 was never validated and it binds on every fit -- measured
            # dispersion is 1.43 for targets and 2.00 for carries -- so it was
            # the operative parameter rather than a safety rail. Inner folds pick
            # 0.15, 0.25 and 0.25, with the penalty rising on both sides of that
            # range and uncapped worst on every fold, so the minimum is interior.
            # On the holdouts: target CRPS -1.57%, carry CRPS -1.21%, carry MAE
            # -0.48%, target cov80 -1.81pp toward nominal, all three folds each.
            self.innovation_cap = 0.25 if self.stream in {"target", "carry"} else 2.0

    @property
    def count_col(self) -> str:
        return STREAMS[self.stream]["count"]

    def _prepare(self, rows: pd.DataFrame) -> pd.DataFrame:
        required = set(GROUP_KEYS + ["player_key", "player_name", "position"])
        missing = required - set(rows.columns)
        if missing:
            raise ValueError(f"player-season rows are missing columns: {sorted(missing)}")
        d = rows.copy()
        d["position"] = d["position"].astype(str).str.upper()
        d = d[d["position"].isin(MODEL_POSITIONS)].copy()
        if self.count_col not in d:
            d[self.count_col] = 0
        d[self.count_col] = (
            pd.to_numeric(d[self.count_col], errors="coerce")
            .fillna(0.0)
            .round()
            .clip(lower=0)
            .astype(int)
        )
        return d.sort_values(GROUP_KEYS + ["player_key"]).reset_index(drop=True)

    def _fit_metadata(self, d: pd.DataFrame) -> None:
        self.players = sorted(d["player_key"].astype(str).unique())
        candidates = list(
            TARGET_BAYESIAN_FEATURES if self.stream == "target" else BAYESIAN_FEATURES
        )
        efficiency_features = (
            self.extra_efficiency_features
            if self.extra_efficiency_features is not None
            else ACCEPTED_VOLUME_EFFICIENCY_FEATURES[self.stream]
        )
        candidates.extend(efficiency_features)
        candidates.extend(self.extra_features)
        self.feature_names = [name for name in candidates if name in d]
        for name in self.feature_names:
            values = pd.to_numeric(d[name], errors="coerce")
            fill = float(values.median()) if values.notna().any() else 0.0
            filled = values.fillna(fill)
            scale = float(filled.std(ddof=0))
            self.feature_fill[name] = fill
            self.feature_mean[name] = float(filled.mean())
            self.feature_scale[name] = scale if scale > 1e-8 else 1.0

        # The cold-start prior stands in for ``per_snap_role`` in ``_role_prior``,
        # so it has to be a per-snap rate. Only rows with observed snaps can
        # supply one: dividing a count by an availability fraction instead puts a
        # season snap total (hundreds) and a fraction (at most one) in the same
        # average, which inflates the estimate by orders of magnitude and does so
        # differently per position — so the roster softmax cannot normalise it
        # away, and the clip in ``_role_prior`` then flattens every position onto
        # the same saturated value.
        snaps = pd.to_numeric(
            d.get("offense_snaps", pd.Series(np.nan, index=d.index)),
            errors="coerce",
        ).to_numpy(dtype=float)
        measured = np.isfinite(snaps) & (snaps > 0)
        opportunity_rate = np.divide(
            d[self.count_col].to_numpy(dtype=float),
            snaps,
            out=np.full(len(d), np.nan, dtype=float),
            where=measured,
        )
        is_cold = pd.to_numeric(
            d.get("cold_start", pd.Series(0, index=d.index)), errors="coerce"
        ).fillna(0).to_numpy(dtype=int) == 1
        prior_availability = pd.to_numeric(
            d.get("prior_availability", pd.Series(np.nan, index=d.index)),
            errors="coerce",
        )
        fallback = STREAMS[self.stream]["fallback"]
        for position in MODEL_POSITIONS:
            is_position = d["position"].to_numpy() == position
            values = opportunity_rate[is_position & is_cold]
            values = values[np.isfinite(values)]
            if not len(values):
                values = opportunity_rate[is_position]
                values = values[np.isfinite(values)]
            estimate = float(np.mean(values)) if len(values) else fallback[position]
            # A per-snap rate cannot exceed one opportunity per snap. Clipping
            # here rather than only in ``_role_prior`` keeps a saturated estimate
            # visible in the persisted metadata instead of silently becoming the
            # same number for every position.
            self.cold_role_prior[position] = float(np.clip(estimate, 1e-4, 1.0))
            position_availability = prior_availability[is_position].dropna()
            self.availability_prior[position] = (
                float(position_availability.median())
                if len(position_availability)
                else 0.75
            )
        # The cap bounds how much role churn this model is willing to represent,
        # so it belongs on the measured quantity. Calibration then asks what
        # input scale delivers that much churn through the softmax.
        target = min(self._estimate_role_innovation(d), float(self.innovation_cap))
        allocation = mask = None
        if self.calibrated_innovation:
            allocation, mask = self._innovation_rooms(d)
            self.role_innovation_scale = calibrate_innovation_scale(
                allocation, mask, target, seed=self.innovation_calibration_seed
            )
        else:
            self.role_innovation_scale = target
        self.cold_role_multiplier = self._fit_cold_role_multiplier(
            d, allocation, mask
        )

    def _fit_cold_role_multiplier(self, d, allocation, mask) -> float:
        """The factor cold rows' innovation is scaled by, in the chosen mode.

        Both modes go through the same calibration as the base scale when it is
        on, because a realized dispersion and an input scale are not the same
        quantity on either side of the split.
        """
        if not self.cold_role_innovation:
            return 1.0
        if self.cold_role_scale_mode == "relative":
            return self._estimate_cold_role_multiplier(d)
        if self.cold_role_scale_mode != "measured":
            raise ValueError(
                f"unknown cold_role_scale_mode: {self.cold_role_scale_mode!r}; "
                "choose 'relative' or 'measured'"
            )
        cold_rms, _ = self._cold_and_warm_dispersion(d)
        if not np.isfinite(cold_rms) or self.role_innovation_scale <= 1e-8:
            return 1.0
        # Keyed on the model's own configuration, not on whether the caller
        # happened to pass rooms in. Those come apart: ``_fit_metadata`` only
        # supplies them when calibration is on, so in the pipeline the two
        # agree, but a caller holding this model directly could calibrate the
        # cold scale while the base scale it is divided by was not calibrated.
        # The ratio between a calibrated numerator and an uncalibrated
        # denominator is not a widening factor, and nothing would have said so.
        if self.calibrated_innovation:
            if allocation is None or mask is None:
                allocation, mask = self._innovation_rooms(d)
            cold_scale = calibrate_innovation_scale(
                allocation, mask, cold_rms, seed=self.innovation_calibration_seed
            )
        else:
            cold_scale = cold_rms
        return float(
            np.clip(
                cold_scale / self.role_innovation_scale,
                1.0,
                self.cold_role_multiplier_cap,
            )
        )

    def _innovation_support(self, d: pd.DataFrame) -> np.ndarray:
        """Rows the softmax is actually fitted over.

        For targets, ``_design`` masks quarterbacks out of the likelihood. The
        innovation estimator used to group over every row regardless, so both
        the softmax denominator and the observed-share normalization included
        players the model never allocates to — the scale was estimated under a
        support the model does not use.
        """
        if self.stream == "target":
            return (~d["position"].eq("QB")).to_numpy()
        return np.ones(len(d), dtype=bool)

    def _innovation_rooms(self, d: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Padded deterministic allocations per team-season, and their support."""
        support = self._innovation_support(d)
        groups = [
            group.index.to_numpy(dtype=int)[support[group.index.to_numpy(dtype=int)]]
            for _, group in d.groupby(GROUP_KEYS, sort=False, dropna=False)
        ]
        groups = [indices for indices in groups if len(indices) > 1]
        if not groups:
            return np.ones((1, 1), dtype=float), np.ones((1, 1), dtype=bool)
        role = self._role_prior(d)
        exposure = self._innovation_exposure(d)
        score = np.log(role) + np.log(np.clip(exposure, 0.001, 1.0))
        width = max(len(indices) for indices in groups)
        allocation = np.zeros((len(groups), width), dtype=float)
        mask = np.zeros((len(groups), width), dtype=bool)
        for row, indices in enumerate(groups):
            weights = np.exp(score[indices] - score[indices].max())
            allocation[row, : len(indices)] = weights / weights.sum()
            mask[row, : len(indices)] = True
        return allocation, mask

    def _innovation_exposure(self, d: pd.DataFrame) -> np.ndarray:
        snap_share = pd.to_numeric(
            d.get("snap_share", pd.Series(np.nan, index=d.index)), errors="coerce"
        )
        availability = pd.to_numeric(
            d.get("observed_availability", pd.Series(1.0, index=d.index)),
            errors="coerce",
        ).fillna(1.0)
        return snap_share.where(snap_share.gt(0), availability).fillna(0.03).to_numpy(
            dtype=float
        )

    def _cold_role_rows(self, d: pd.DataFrame) -> np.ndarray:
        """Rows whose role prior is a population fallback, not their own history.

        ``_role_prior`` falls through four sources in order: the player's own
        per-snap rate, last season's share, the draft prior, and the position
        mean. The last two describe someone the model has never seen play. Their
        season outcomes are the most dispersed population in the data — most do
        nothing and a few take over a job — and a single innovation scale fitted
        over everyone cannot represent both them and an established starter.

        A missing role in *this* stream is not enough on its own. A receiver
        with a full season of snaps and no carries also falls through to the
        position mean in the carry room, and the model predicts his zero
        perfectly well; widening him buys nothing. Requiring no prior snap
        exposure either restricts this to the population the coverage split
        actually found — on the carry stream that is 34% of rows rather than
        62%.
        """
        per_snap = pd.to_numeric(
            d.get(
                STREAMS[self.stream].get("per_snap_role"),
                pd.Series(np.nan, index=d.index),
            ),
            errors="coerce",
        )
        role = pd.to_numeric(d.get(STREAMS[self.stream]["role"]), errors="coerce")
        if "prior_snap_share" not in d and self.cold_role_innovation:
            # Without it the mask degrades to "no role in this stream", which is
            # a different and deliberately rejected population: 62% of carry
            # rows rather than 34%, including receivers whose zero carries the
            # model already predicts well. That is a configuration nothing
            # validated, and it would run silently.
            raise ValueError(
                "cold_role_innovation needs prior_snap_share to tell a player "
                "the model has never seen from one it has seen play but never "
                "carry; the column is missing from these rows"
            )
        snaps = pd.to_numeric(
            d.get("prior_snap_share", pd.Series(np.nan, index=d.index)), errors="coerce"
        )
        return (~per_snap.gt(0) & ~role.gt(0) & ~snaps.gt(0)).to_numpy()

    def _role_innovation_residuals(
        self, d: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray]:
        """Within-room-centred log-share residuals, and which rows are cold.

        Restricted to the support the model allocates over. For targets that
        excludes quarterbacks, who ``_design`` masks out of the likelihood; the
        estimator used to leave them in both the softmax denominator and the
        observed-share normalization, so it described a room the model never
        fits.
        """
        role = self._role_prior(d)
        exposure = self._innovation_exposure(d)
        support = self._innovation_support(d)
        cold = self._cold_role_rows(d)
        score = np.log(role) + np.log(np.clip(exposure, 0.001, 1.0))
        residuals: list[np.ndarray] = []
        cold_flags: list[np.ndarray] = []
        for _, group in d.groupby(GROUP_KEYS, sort=False, dropna=False):
            indices = group.index.to_numpy(dtype=int)
            indices = indices[support[indices]]
            if len(indices) < 2:
                continue
            weights = np.exp(score[indices] - score[indices].max())
            expected = weights / weights.sum()
            counts = d.loc[indices, self.count_col].to_numpy(dtype=float)
            observed = (counts + 0.5) / (counts.sum() + 0.5 * len(indices))
            group_residual = np.log(observed) - np.log(expected)
            residuals.append(group_residual - group_residual.mean())
            cold_flags.append(cold[indices])
        if not residuals:
            return np.zeros(0), np.zeros(0, dtype=bool)
        return np.concatenate(residuals), np.concatenate(cold_flags)

    def _estimate_role_innovation(self, d: pd.DataFrame) -> float:
        """Training-only RMS log-share error after removing roster location."""
        residuals, _ = self._role_innovation_residuals(d)
        if not len(residuals):
            return 0.10
        return float(np.clip(np.sqrt(np.mean(residuals**2)), 0.10, 2.0))

    def _cold_and_warm_dispersion(self, d: pd.DataFrame) -> tuple[float, float]:
        """Realized log-share spread either side of the cold split, or NaNs."""
        residuals, cold = self._role_innovation_residuals(d)
        if cold.sum() < MIN_COLD_ROLE_ROWS or (~cold).sum() < MIN_COLD_ROLE_ROWS:
            return float("nan"), float("nan")
        return (
            float(np.sqrt(np.mean(residuals[cold] ** 2))),
            float(np.sqrt(np.mean(residuals[~cold] ** 2))),
        )

    def _estimate_cold_role_multiplier(self, d: pd.DataFrame) -> float:
        """How much wider the cold rows' realized log-share spread is.

        Expressed as a ratio rather than a second absolute scale so that it
        composes with whatever the base scale ends up being — the cap, and the
        calibration that inverts the softmax compression, both act on the base
        and the ratio rides on top of them.
        """
        residuals, cold = self._role_innovation_residuals(d)
        if cold.sum() < MIN_COLD_ROLE_ROWS or (~cold).sum() < MIN_COLD_ROLE_ROWS:
            return 1.0
        warm_rms = float(np.sqrt(np.mean(residuals[~cold] ** 2)))
        cold_rms = float(np.sqrt(np.mean(residuals[cold] ** 2)))
        if warm_rms <= 1e-8:
            return 1.0
        return float(np.clip(cold_rms / warm_rms, 1.0, self.cold_role_multiplier_cap))

    def _role_prior(self, d: pd.DataFrame) -> np.ndarray:
        per_snap = pd.to_numeric(
            d.get(
                STREAMS[self.stream].get("per_snap_role"),
                pd.Series(np.nan, index=d.index),
            ),
            errors="coerce",
        )
        role = pd.to_numeric(d.get(STREAMS[self.stream]["role"]), errors="coerce")
        draft = pd.to_numeric(d.get(STREAMS[self.stream]["draft"]), errors="coerce")
        cold = d["position"].map(self.cold_role_prior).astype(float)
        valid_per_snap = per_snap.gt(0)
        valid_role = role.gt(0)
        prior = per_snap.where(valid_per_snap)
        both = valid_per_snap & valid_role
        prior.loc[both] = np.exp(
            float(self.per_snap_weight) * np.log(per_snap.loc[both])
            + (1.0 - float(self.per_snap_weight)) * np.log(role.loc[both])
        )
        prior = prior.where(prior.notna(), role.where(role > 0))
        prior = prior.where(prior.notna(), draft.where(draft > 0))
        prior = prior.where(prior.notna(), cold)
        return np.clip(prior.to_numpy(dtype=float), 1e-5, 1.0)

    def _matrix(self, d: pd.DataFrame) -> np.ndarray:
        columns = []
        for name in self.feature_names:
            values = pd.to_numeric(
                d.get(name, pd.Series(np.nan, index=d.index)), errors="coerce"
            ).fillna(self.feature_fill[name])
            columns.append(
                (values.to_numpy(dtype=float) - self.feature_mean[name])
                / self.feature_scale[name]
            )
        return np.column_stack(columns) if columns else np.zeros((len(d), 0))

    def _design(
        self,
        rows: pd.DataFrame,
        *,
        fit: bool = False,
        use_observed_availability: bool = False,
        use_observed_snap: bool = False,
        use_observed_starter: bool = False,
    ):
        d = self._prepare(rows)
        if fit:
            totals = d.groupby(GROUP_KEYS)[self.count_col].transform("sum")
            d = d[totals > 0].reset_index(drop=True)
            self._fit_metadata(d)
        if d.empty:
            raise ValueError(f"no roster rows available for {self.stream} allocation")

        role_prior = self._role_prior(d)
        cold_role = self._cold_role_rows(d)
        X = self._matrix(d)
        player_idx = _codes(d["player_key"], self.players)
        if use_observed_snap:
            exposure = pd.to_numeric(
                d.get("snap_share", pd.Series(np.nan, index=d.index)),
                errors="coerce",
            )
            fallback_exposure = pd.to_numeric(
                d.get("observed_availability", pd.Series(1.0, index=d.index)),
                errors="coerce",
            ).fillna(1.0)
            availability = exposure.where(exposure.gt(0), fallback_exposure * 0.03)
        elif use_observed_availability:
            availability = pd.to_numeric(
                d.get("observed_availability", pd.Series(1.0, index=d.index)),
                errors="coerce",
            ).fillna(1.0)
        else:
            source = d.get("projected_snap_share")
            if source is None:
                source = d.get(
                    "projected_availability",
                    d.get("prior_availability", pd.Series(np.nan, index=d.index)),
                )
            availability = pd.to_numeric(source, errors="coerce")
            availability = availability.fillna(d["position"].map(self.availability_prior))
            availability = availability.fillna(0.75)
        availability = np.clip(availability.to_numpy(dtype=float), 0.001, 1.0)
        d["_projected_snap_share"] = availability
        starter_exposure = np.ones(len(d), dtype=float)
        if self.stream == "pass" and use_observed_starter:
            primary = pd.to_numeric(
                d.get("primary_qb", pd.Series(0, index=d.index)), errors="coerce"
            ).fillna(0).to_numpy(dtype=float)
            quarterback = d["position"].eq("QB").to_numpy()
            starter_exposure[quarterback] = (
                BACKUP_QB_EXPOSURE
                + (1.0 - BACKUP_QB_EXPOSURE) * primary[quarterback]
            )

        groups = list(d.groupby(GROUP_KEYS, sort=True, dropna=False))
        max_slots = max(len(group) for _, group in groups)
        n_groups = len(groups)
        counts = np.zeros((n_groups, max_slots), dtype=int)
        mask = np.zeros((n_groups, max_slots), dtype=float)
        role = np.ones((n_groups, max_slots), dtype=float)
        cold = np.zeros((n_groups, max_slots), dtype=bool)
        offset = np.zeros((n_groups, max_slots), dtype=float)
        starter_offset = np.zeros((n_groups, max_slots), dtype=float)
        matrix = np.zeros((n_groups, max_slots, X.shape[1]), dtype=float)
        players = np.zeros((n_groups, max_slots), dtype=int)
        row_idx = np.full((n_groups, max_slots), -1, dtype=int)
        innovation_basis = np.zeros(
            (n_groups, max_slots, max(max_slots - 1, 0)), dtype=float
        )
        group_rows = []
        for group_i, (key, group) in enumerate(groups):
            indices = group.index.to_numpy(dtype=int)
            size = len(indices)
            support = np.ones(size, dtype=float)
            if self.stream == "target":
                support = (~d.loc[indices, "position"].eq("QB")).to_numpy(dtype=float)
            counts[group_i, :size] = (
                d.loc[indices, self.count_col].to_numpy(dtype=int) * support.astype(int)
            )
            mask[group_i, :size] = support
            role[group_i, :size] = role_prior[indices]
            cold[group_i, :size] = cold_role[indices]
            offset[group_i, :size] = np.log(availability[indices])
            starter_offset[group_i, :size] = np.log(starter_exposure[indices])
            matrix[group_i, :size] = X[indices]
            players[group_i, :size] = player_idx[indices]
            row_idx[group_i, :size] = indices
            if size > 1:
                innovation_basis[group_i, :size, : size - 1] = _sum_to_zero_basis(size)
            d.loc[indices, "_group_idx"] = group_i
            group_rows.append(dict(zip(GROUP_KEYS, key)))
        d["_group_idx"] = d["_group_idx"].astype(int)
        return {
            "rows": d,
            "group_keys": pd.DataFrame(group_rows),
            "counts": counts,
            "totals": counts.sum(axis=1),
            "mask": mask,
            "role_prior": role,
            "cold_role": cold,
            "availability_offset": offset,
            "starter_offset": starter_offset,
            "X": matrix,
            "player_idx": players,
            "row_idx": row_idx,
            "innovation_basis": innovation_basis,
        }

    def fit(self, rows: pd.DataFrame, **sample_kwargs) -> "SeasonRosterShareModel":
        """Fit a season-level logistic-normal Multinomial roster allocation."""
        import pymc as pm

        design = self._design(
            rows,
            fit=True,
            use_observed_availability=True,
            use_observed_snap=True,
            use_observed_starter=self.stream == "pass",
        )
        with pm.Model() as model:
            # Once snap exposure is explicit, the lagged availability/snap
            # covariates are only small residual corrections. A tight target
            # prior prevents counting playing time twice.
            beta_scale = 0.05 if self.stream == "target" else 0.35
            beta = pm.Normal("beta", 0.0, beta_scale, shape=len(self.feature_names))
            eta = (
                np.log(design["role_prior"])
                + design["availability_offset"]
                + design["starter_offset"]
            )
            eta = eta + pm.math.sum(design["X"] * beta, axis=2)
            masked_eta = pm.math.switch(design["mask"] > 0, eta, -20.0)
            probability = pm.math.softmax(masked_eta, axis=1)
            pm.Multinomial(
                "obs",
                n=design["totals"],
                p=probability,
                observed=design["counts"],
            )
            sample_kwargs.setdefault("target_accept", 0.93)
            self.idata = sample_model(model, **sample_kwargs)
        return self

    def predict_share_samples(
        self,
        rows: pd.DataFrame,
        *,
        availability_samples: np.ndarray | None = None,
        snap_samples: np.ndarray | None = None,
        eligibility_samples: np.ndarray | None = None,
        starter_probability_samples: np.ndarray | None = None,
        seed: int = 0,
    ) -> RosterSharePrediction:
        """Draw coherent season shares; every team-season sums to one."""
        if self.idata is None:
            raise RuntimeError("fit the roster-share model before predicting")
        design = self._design(rows)
        post = self.idata.posterior
        beta = _stack(post, "beta")
        draws = beta.shape[-1]
        eta = np.log(design["role_prior"])[..., None]
        exposure_samples = snap_samples if snap_samples is not None else availability_samples
        exposure_matrix: np.ndarray | None = None
        if exposure_samples is None:
            eta = eta + design["availability_offset"][..., None]
        else:
            exposure_samples = np.asarray(exposure_samples, dtype=float)
            if exposure_samples.shape != (len(design["rows"]), draws):
                raise ValueError(
                    "exposure samples must align to roster rows and posterior draws"
                )
            exposure_matrix = _roster_sample_matrix(
                design, np.clip(exposure_samples, 1e-5, 1.0), fill=1.0
            )
            if not self.per_game_allocation:
                eta = eta + np.log(exposure_matrix)
        if self.stream == "pass" and starter_probability_samples is not None:
            starter_probability_samples = np.asarray(
                starter_probability_samples, dtype=float
            )
            if starter_probability_samples.shape != (len(design["rows"]), draws):
                raise ValueError(
                    "starter samples must align to roster rows and posterior draws"
                )
            quarterback = design["rows"]["position"].eq("QB").to_numpy()[:, None]
            exposure = np.where(
                quarterback,
                BACKUP_QB_EXPOSURE
                + (1.0 - BACKUP_QB_EXPOSURE) * starter_probability_samples,
                1.0,
            )
            eta = eta + np.log(_roster_sample_matrix(design, exposure, fill=1.0))
        else:
            eta = eta + design["starter_offset"][..., None]
        eta = eta + np.einsum("gkf,fs->gks", design["X"], beta)
        live = np.broadcast_to(design["mask"][..., None] > 0, eta.shape)
        if eligibility_samples is not None:
            eligibility_samples = np.asarray(eligibility_samples, dtype=float)
            if eligibility_samples.shape != (len(design["rows"]), draws):
                raise ValueError(
                    "eligibility samples must align to roster rows and posterior draws"
                )
            eligibility = _roster_sample_matrix(
                design, eligibility_samples, fill=0.0, min_value=0.0
            )
            # If a draw gates out an entire roster, retain the highest-scoring
            # supported player so the team total can still be allocated.
            none_eligible = eligibility.sum(axis=1) <= 0
            if none_eligible.any():
                fallback = np.argmax(np.where(live, eta, -np.inf), axis=1)
                group_draw = np.argwhere(none_eligible)
                eligibility[
                    group_draw[:, 0], fallback[none_eligible], group_draw[:, 1]
                ] = 1.0
            live = live & (eligibility > 0)
        live = np.ascontiguousarray(live)

        rng = np.random.default_rng(seed)
        innovation_z = rng.normal(
            size=(
                len(design["group_keys"]),
                design["innovation_basis"].shape[2],
                draws,
            )
        )
        # The sum-to-zero basis keeps a room's innovation from moving its
        # location. Scaling per row after the basis is applied gives up that
        # property, but the softmax renormalizes regardless, and the alternative
        # -- one scale for a room mixing rookies and established starters -- is
        # what the widths are wrong about.
        scale = np.full(design["mask"].shape, float(self.role_innovation_scale))
        if self.cold_role_innovation and self.cold_role_multiplier != 1.0:
            scale = np.where(
                design["cold_role"], scale * float(self.cold_role_multiplier), scale
            )
        innovation = (
            np.einsum("gkj,gjs->gks", design["innovation_basis"], innovation_z)
            * scale[..., None]
        )
        if self.per_game_allocation and exposure_matrix is not None:
            probability = _per_game_shares(
                eta + innovation,
                live,
                exposure_matrix,
                games=int(self.allocation_games),
                seed=seed + 991,
            )
        elif self.mean_preserving_innovation:
            probability = mean_preserving_shares(eta, eta + innovation, live)
        else:
            probability = simplex_shares(eta + innovation, live)
        shares = np.zeros((len(design["rows"]), draws), dtype=float)
        for group_i in range(len(design["group_keys"])):
            active = design["mask"][group_i].astype(bool)
            indices = design["row_idx"][group_i, active]
            shares[indices] = probability[group_i, active]
        return RosterSharePrediction(
            rows=design["rows"], group_keys=design["group_keys"], shares=shares
        )


def volume_input_problems(data: SeasonAverageData) -> list[str]:
    """Preconditions the season-average fit needs, checked before any sampling.

    The pipeline fits eight components in sequence, and the quarterback layers
    come fourth. A source that cannot support them therefore used to spend three
    posterior fits before failing, and failed from inside whichever model noticed
    first — for the committed CSVs that was a PyTensor ``MemoryError`` on an empty
    softmax, which says nothing about the actual problem. Check the responses up
    front instead, and name the source limitation rather than the symptom.
    """
    problems: list[str] = []
    team_rows = data.team_rows
    player_rows = data.player_rows
    if team_rows is None or team_rows.empty:
        problems.append("team_rows is empty; no team-season volume to fit")
    if player_rows is None or player_rows.empty:
        problems.append("player_rows is empty; no roster to allocate over")
        return problems

    quarterbacks = player_rows[player_rows["position"].astype(str).str.upper().eq("QB")]
    if quarterbacks.empty:
        problems.append("player_rows contain no quarterbacks; QB volume cannot be fit")
        return problems

    snaps = pd.to_numeric(
        quarterbacks.get("offense_snaps", pd.Series(np.nan, index=quarterbacks.index)),
        errors="coerce",
    ).fillna(0.0)
    if not snaps.gt(0).any():
        sources = sorted(
            player_rows.get(
                "roster_snapshot_source", pd.Series(dtype=object)
            ).dropna().unique()
        )
        problems.append(
            "no quarterback has a positive offense_snaps value, which the QB "
            "workload and pass-propensity models both require. The committed "
            "snapcount CSVs contain no quarterback rows, so a legacy-only run "
            "cannot fit these layers — use source='nflverse' or 'auto'"
            + (f" (roster snapshot sources present: {', '.join(sources)})" if sources else "")
        )
    return problems


@dataclass
class SeasonAveragePrediction:
    team: dict[str, object]
    player_rows: pd.DataFrame
    availability_probability: np.ndarray
    games_active: np.ndarray
    availability: np.ndarray
    snap_share: np.ndarray
    qb_workload_share: np.ndarray
    qb_pass_propensity: np.ndarray
    target_role_probability: np.ndarray
    carry_eligibility_probability: np.ndarray
    pass_attempt_share: np.ndarray
    target_share: np.ndarray
    carry_share: np.ndarray
    pass_attempts: np.ndarray
    targets: np.ndarray
    carries: np.ndarray
    pass_attempts_per_team_game: np.ndarray
    targets_per_team_game: np.ndarray
    carries_per_team_game: np.ndarray
    pass_attempts_per_active_game: np.ndarray
    targets_per_active_game: np.ndarray
    carries_per_active_game: np.ndarray
    regime_probability: np.ndarray | None = None
    regime_samples: np.ndarray | None = None


@dataclass
class SeasonAverageVolumePipeline:
    """End-to-end season-average team volume and player role pipeline."""

    team_model: TeamSeasonAverageModel = field(default_factory=TeamSeasonAverageModel)
    target_model: SeasonRosterShareModel = field(
        default_factory=lambda: SeasonRosterShareModel("target")
    )
    carry_model: SeasonRosterShareModel = field(
        default_factory=lambda: SeasonRosterShareModel("carry")
    )
    availability_model: SeasonAvailabilityModel = field(
        default_factory=SeasonAvailabilityModel
    )
    workload_model: QBWorkloadShareModel = field(default_factory=QBWorkloadShareModel)
    snap_model: SeasonSnapShareModel = field(default_factory=SeasonSnapShareModel)
    qb_propensity_model: QBPassPropensityModel = field(
        default_factory=QBPassPropensityModel
    )
    target_role_model: SeasonTargetRoleModel = field(
        default_factory=SeasonTargetRoleModel
    )
    carry_eligibility_model: SeasonCarryEligibilityModel = field(
        default_factory=SeasonCarryEligibilityModel
    )
    # Allocate roster shares week by week rather than multiplying by
    # season-average availability and renormalising once. Set together for all
    # three streams because they share the misspecification and a run with it on
    # for carries and off for passes is neither of the two models. Off by
    # default: it moves every low-availability share, so it needs a walk-forward
    # of its own before it can ship. See ``_per_game_shares``.
    per_game_allocation: bool = False
    # Experimental role-only challenger. The default baseline is unchanged.
    role_regime_coupling: bool = False
    # Upstream likelihood challenger: out-of-fold regime probabilities enter
    # role and availability regressions instead of tilting fitted shares.
    regime_likelihood_features: bool = False
    # Lagged postseason role signal, restricted to the skill-position role
    # models. Promoted 2026-08-02: carry MAE -2.77% and CRPS -1.47%, snap MAE
    # -0.82% and CRPS -0.63%, all winning three holdouts of three, with the
    # protected pass stream unchanged to five decimal places. One documented
    # exception: target MAE moves +0.07% at 1/3, against target CRPS -0.29% at
    # 2/3. See docs/pipeline-followups-2026-08.md and
    # docs/postseason-history-assessment.md.
    postseason_role_features: bool = True
    # Preseason market consensus in the role and playing-time regressions. Off
    # until measured: it is the one input not derived from play-by-play, so a
    # gain would be real new information and a loss would say the market adds
    # nothing the history does not already carry. See
    # ``_enable_market_adp_features`` and ffmodel.features.market.
    # Leakage-safe injury history and preseason injury snapshot in the
    # availability regression. Screened at the availability layer first
    # (docs/injury-availability-2026-08.md): CRPS -2.39% pooled and 3/3 folds,
    # -5.15% on the injury-exposed half. Off until it clears the scoring gate,
    # because availability feeds exposure and a gain there is not a gain here.
    # Injured reserve, PUP and non-football-injury as deviations from the
    # pooled reserve flag. Promoted 2026-09-01: availability CRPS -1.12% and MAE
    # -1.46% on three holdouts of three, almost all of it injured reserve at
    # CRPS -9.24% and MAE -18.38%. See RESERVE_KIND_FEATURES for the one
    # population it does not help.
    reserve_kind_features: bool = True
    injury_availability_features: bool = False
    # Let the availability regression read the player's own availability
    # history, not only last season. See ``AVAILABILITY_HISTORY_FEATURES``:
    # -2.19% held-out MAE on five folds of five, from a column the frame has
    # carried all along. Kept as its own flag so the arm without it stays
    # reproducible, and so a frame built before the pathway features can still
    # be fitted by turning it off.
    availability_history_features: bool = True
    market_adp_features: bool = False
    # Per-position rank slopes and drafted effects. Measured and rejected:
    # worse on three drafted-pool holdouts of three at double the fit time,
    # because the encoding is collinear with the main effects and the shared
    # feature prior cannot hold the opposing coefficients it needs. The terms
    # themselves are redundant with the usage history. Left in place so the
    # negative result stays reproducible; see ffmodel.features.market and
    # docs/adp-ablation-2026-08.md before turning it on.
    market_adp_interactions: bool = False
    # The draft board in the *availability* regression, which the arm above
    # deliberately excludes.
    #
    # The exclusion was reasonable when written -- adding a feature everywhere
    # at once makes a null uninterpretable -- but it meant the ADP ablation's
    # null result never tested this layer, and this layer is where the defect
    # is. Fitting availability alone and scoring it on the rows it was trained
    # on shows the model reproducing the drafted/undrafted split *in sample*:
    # -6.7% on drafted and +5.3% on undrafted against a pooled -0.2%. A model
    # unbiased overall while missing in opposite directions on two halves of
    # its own training data is not mis-levelled, it is unable to tell the
    # halves apart, and no intercept correction can fix that.
    #
    # The board can. On 2024, held out, drafted-pool bias falls from -8.5% to
    # -4.1% and the in-sample flip largely collapses (running backs from
    # -5.6%/+5.3% to -1.4%/+0.5%). Receivers keep about half of theirs, so this
    # narrows the resolution failure rather than closing it.
    #
    # Kept a separate flag from ``market_adp_features`` so that arm's measured
    # result stays exactly reproducible.
    #
    # Promoted 2026-08-22 on the paired 2022-2024 scoring gate, zero divergences
    # in both arms: pooled MAE -0.93% and CRPS -1.11% winning every holdout,
    # drafted-pool CRPS -1.44% winning every holdout, undrafted MAE -1.83%
    # winning every holdout. Drafted-pool MAE is +0.36%/-0.72%/-0.94%, so it
    # wins two of three rather than all three -- the one exception, and it meets
    # the two-of-three stability rule rather than needing a waiver.
    market_adp_availability_features: bool = True
    # The draft board in the quarterback room -- the passing-share softmax, its
    # hurdle, and pass attempts per snap.
    #
    # Also excluded from the measured ADP arm, and with a clearer story than
    # most: the room's existing evidence about who starts is ``qb_depth_rank``
    # and ``qb_listed_starter``, both read off preseason depth charts, which
    # nflverse itself re-shaped in 2025 and which teams publish without meaning
    # them. The market's opinion on a starting quarterback is the thing it is
    # most confident about and least likely to be wrong on.
    #
    # Untested at the time of writing. Off, and to stay off until a layer-level
    # screen and then the scoring gate say otherwise.
    market_adp_qb_features: bool = False
    # Which exposure the availability and snap layers are built on.
    #
    # "roster" is ``games``, roster-active weeks, and is what every measurement
    # in this package was made against. "snap" is ``snap_games``, weeks with at
    # least one offensive snap.
    #
    # The case for changing it is that the roster label means different things
    # for the two halves of the population -- within a game of the truth for
    # drafted players, and employment rather than participation for undrafted
    # ones (an undrafted quarterback: 11.31 roster weeks, 4.36 with a snap).
    # See ``SeasonAvailabilityModel.games_column``.
    #
    # Untested. Off until a paired gate says otherwise, and it changes the
    # meaning of ``availability`` everywhere downstream, so a comparison across
    # this setting is not comparing like with like.
    availability_target: str = "roster"
    # Preseason team win totals in the team layer. The only market input that is
    # about teams rather than players, and the team layer is the one the ADP work
    # never reached. Off until measured; see docs/vegas-win-totals-2026-08.md.
    market_win_total_features: bool = False
    # Correct the softmax renormalization bias the role innovation introduces.
    # ``True`` enables every allocation layer; a tuple names a subset, because
    # the layers were measured to disagree — see
    # ``_enable_mean_preserving_innovation``. See ``mean_preserving_shares``.
    mean_preserving_innovation: bool | tuple[str, ...] = False
    # Solve for the input noise scale that realizes the churn the estimator
    # measured, rather than handing the measurement straight to the sampler.
    # Promoted 2026-08-03; see ``calibrate_innovation_scale`` and
    # docs/role-innovation-2026-08.md for the accepted gate exception.
    calibrated_innovation: bool = True
    # Widen the role innovation for players with no prior role of their own.
    # The tail under-coverage on total fantasy points is almost entirely theirs:
    # 28% of rows with no prior snap share fall outside a 95% interval against a
    # 5% nominal, 32 of 33 above it, while rows with an established role sit at
    # 2.6%. Off until validated in-window -- 2025 diagnosed it and must not size
    # it. See docs/out-of-sample-2025.md.
    cold_role_innovation: bool = True
    # "relative" or "measured"; see ``SeasonRosterShareModel``.
    cold_role_scale_mode: str = "measured"
    # Overrides the target and carry allocators' ``innovation_cap``. ``None``
    # keeps each stream's own default. See scripts/select_innovation_cap.py.
    innovation_cap: float | None = None
    regime_model: SeasonRegimeModel | None = None
    regime_coupler: SeasonRegimeRoleCoupling | None = None
    fit_seconds: dict[str, float] = field(default_factory=dict)

    def fit(self, data: SeasonAverageData, **sample_kwargs) -> "SeasonAverageVolumePipeline":
        # Input preflight first, before any feature enablement.
        #
        # The enablement guards below raise on a frame missing their columns,
        # which is right, but they were running first: a frame with no usable
        # volume inputs *and* no ADP columns reported the ADP problem, sending
        # the reader to rebuild a market feature when the real fault was that
        # the frame could not be fitted at all. Preflight reports every input
        # problem at once and should be what a caller sees first.
        problems = volume_input_problems(data)
        if problems:
            raise ValueError(
                "season-average volume inputs are not fittable:\n  - "
                + "\n  - ".join(problems)
            )
        if self.role_regime_coupling and self.regime_likelihood_features:
            raise ValueError("choose either post-hoc or upstream regime coupling, not both")
        self._apply_availability_target(data.player_rows)
        if self.postseason_role_features:
            self._enable_postseason_role_features()
        if self.availability_history_features:
            self._enable_availability_history(data.player_rows)
        if self.reserve_kind_features:
            missing = [
                name
                for name in RESERVE_KIND_FEATURES
                if name not in data.player_rows.columns
            ]
            if missing:
                raise ValueError(
                    f"reserve_kind_features is on but {missing} are absent from "
                    "the player rows. Rebuild the cache with "
                    "scripts/build_projection_cache.py, or set "
                    "reserve_kind_features=False to fit the pooled reserve flag "
                    "deliberately rather than by accident"
                )
            self.availability_model.extra_features = tuple(
                dict.fromkeys(
                    (
                        *self.availability_model.extra_features,
                        *RESERVE_KIND_FEATURES,
                    )
                )
            )
        if self.injury_availability_features:
            missing = [
                name
                for name in INJURY_AVAILABILITY_FEATURES
                if name not in data.player_rows.columns
            ]
            if missing:
                raise ValueError(
                    f"injury_availability_features is on but {missing} are "
                    "absent from the player rows; rebuild the cache rather than "
                    "fitting a model that would silently drop them"
                )
            self.availability_model.extra_features = tuple(
                dict.fromkeys(
                    (
                        *self.availability_model.extra_features,
                        *INJURY_AVAILABILITY_FEATURES,
                    )
                )
            )
        if self.market_adp_interactions and not self.market_adp_features:
            raise ValueError(
                "market_adp_interactions needs market_adp_features: an "
                "interaction without its main effects is not a model anyone "
                "meant to fit"
            )
        if self.market_adp_features:
            self._enable_market_adp_features(data.player_rows)
        if self.market_adp_availability_features:
            self._enable_market_adp_availability(data.player_rows)
        if self.market_adp_qb_features:
            self._enable_market_adp_qb(data.player_rows)
        if self.market_win_total_features:
            if "market_win_total" not in data.team_rows.columns:
                raise ValueError(
                    "market_win_total_features is on but market_win_total is "
                    "absent from the team rows. Build them with "
                    "scripts/augment_cache_features.py --feature win-totals "
                    "rather than fitting a model that would silently drop it"
                )
            self.team_model.market_features = True
        if self.mean_preserving_innovation:
            self._enable_mean_preserving_innovation()
        if self.calibrated_innovation:
            self._enable_calibrated_innovation()
        if self.cold_role_innovation:
            for model in (self.target_model, self.carry_model):
                model.cold_role_innovation = True
                model.cold_role_scale_mode = self.cold_role_scale_mode
        if self.innovation_cap is not None:
            self.target_model.innovation_cap = float(self.innovation_cap)
            self.carry_model.innovation_cap = float(self.innovation_cap)
        player_rows = data.player_rows
        if self.regime_likelihood_features:
            started = perf_counter()
            player_rows = add_walk_forward_regime_probabilities(player_rows)
            self.regime_model = SeasonRegimeModel().fit(data.player_rows)
            self._enable_regime_likelihood_features()
            self.fit_seconds["regime_features"] = perf_counter() - started
        started = perf_counter()
        self.team_model.fit(data.team_rows, **sample_kwargs)
        self.fit_seconds["team"] = perf_counter() - started
        started = perf_counter()
        self.availability_model.fit(player_rows, **sample_kwargs)
        self.fit_seconds["availability"] = perf_counter() - started
        started = perf_counter()
        self.snap_model.fit(player_rows, **sample_kwargs)
        self.fit_seconds["snap"] = perf_counter() - started
        started = perf_counter()
        self.workload_model.fit(player_rows, **sample_kwargs)
        self.fit_seconds["workload"] = perf_counter() - started
        started = perf_counter()
        self.qb_propensity_model.fit(player_rows, **sample_kwargs)
        self.fit_seconds["qb_propensity"] = perf_counter() - started
        started = perf_counter()
        self.carry_eligibility_model.fit(player_rows, **sample_kwargs)
        self.fit_seconds["carry_eligibility"] = perf_counter() - started
        started = perf_counter()
        self.target_model.fit(player_rows, **sample_kwargs)
        self.fit_seconds["target"] = perf_counter() - started
        started = perf_counter()
        self.carry_model.fit(player_rows, **sample_kwargs)
        self.fit_seconds["carry"] = perf_counter() - started
        if self.role_regime_coupling:
            started = perf_counter()
            self.regime_model = self.regime_model or SeasonRegimeModel()
            self.regime_model.fit(data.player_rows)
            self.regime_coupler = SeasonRegimeRoleCoupling().fit(
                data.player_rows, thresholds=self.regime_model.thresholds
            )
            self.fit_seconds["regime"] = perf_counter() - started
        return self

    MEAN_PRESERVING_LAYERS = ("workload", "target", "carry")

    def _enable_mean_preserving_innovation(self) -> None:
        """Turn the correction on for the allocation layers named by the flag.

        Three layers renormalize: the quarterback workload room, and the target
        and carry rooms. The snap and eligibility layers are per-player and
        never renormalize, so they have nothing to correct.

        Enabling all three at once was measured and rejected: it costs 4-7% CRPS
        on the quarterback passing streams. But that cost is the *workload*
        layer's, and the carry layer's contribution was a 0.58% MAE improvement
        on all three folds. Applying it per layer separates the two.

        The carry room is where the correction has the most to fix. Running
        backs lead it and quarterbacks are minor members, and renormalization
        moves mass from the leader outward — which is exactly the observed
        allocation error, a flat +0.5 carries per game onto every quarterback
        against -0.30 off every running back, in every fold.
        """
        selected = self.mean_preserving_innovation
        layers = (
            self.MEAN_PRESERVING_LAYERS
            if selected is True
            else tuple(selected or ())
        )
        unknown = set(layers) - set(self.MEAN_PRESERVING_LAYERS)
        if unknown:
            raise ValueError(
                f"unknown mean-preserving layers: {sorted(unknown)}; "
                f"choose from {list(self.MEAN_PRESERVING_LAYERS)}"
            )
        for name in layers:
            getattr(self, f"{name}_model").mean_preserving_innovation = True

    def _enable_calibrated_innovation(self) -> None:
        """Turn calibration on for the three layers that allocate over a simplex.

        The same three as the mean-preserving correction: the quarterback
        workload room, and the target and carry rooms. The snap and eligibility
        layers are per-player and never renormalize, so no dispersion is lost
        there and there is nothing to invert.
        """
        self.workload_model.calibrated_innovation = True
        self.target_model.calibrated_innovation = True
        self.carry_model.calibrated_innovation = True

    def _enable_postseason_role_features(self) -> None:
        """Offer the lagged postseason signal to the role-shaped submodels.

        Only the models that decide *who holds a role among the skill positions*
        see it. Three layers are deliberately excluded.

        Availability and the team layer: qualifying for the postseason is a
        property of the team's quality, so letting it into an availability
        regression would let "my team was good" stand in for "I stayed healthy".

        The quarterback layers: measured, not assumed. Handing them the feature
        cost 3.39% pass-attempt MAE and 3.27% workload-share MAE, losing all
        three holdouts on both, against a gate that allows no pass-stream
        regression beyond 0.5%. That room is close to winner-take-all and is
        already well determined by depth chart and prior snap share, so a signal
        present on 18% of rows and correlated with team strength rather than
        with who takes the snaps is noise there. The same features are worth
        2.80% carry MAE and 0.82% snap MAE, 3/3 each, in the rooms where the
        allocation is genuinely contested.
        """

        def merged(existing: tuple[str, ...]) -> tuple[str, ...]:
            return tuple(dict.fromkeys((*existing, *POSTSEASON_FEATURES)))

        self.snap_model.extra_features = merged(self.snap_model.extra_features)
        self.target_role_model.extra_features = merged(
            self.target_role_model.extra_features
        )
        self.carry_eligibility_model.extra_features = merged(
            self.carry_eligibility_model.extra_features
        )
        self.target_model.extra_features = merged(self.target_model.extra_features)
        self.carry_model.extra_features = merged(self.carry_model.extra_features)

    def _enable_market_adp_features(self, player_rows: pd.DataFrame) -> None:
        """Append preseason consensus to the role and playing-time regressions.

        The same rooms the postseason features go to, and for the same reason:
        these are the layers that decide who holds a role, which is what a draft
        board is an opinion about. Availability, the team layer and the
        efficiency layer are left alone in this arm -- ADP plausibly informs all
        three, but adding it everywhere at once would make a null result
        uninterpretable and a gain unattributable.

        The absent-column check is not defensive noise. A cache built before
        this feature existed has none of these columns, ``_matrix`` silently
        drops names it cannot find, and the arm would fit exactly the baseline
        and report a clean +0.00% null. That has already happened twice on this
        branch with other features, so it fails here instead.
        """
        wanted = ADP_FEATURES + (
            ADP_INTERACTION_FEATURES if self.market_adp_interactions else ()
        )
        missing = [name for name in wanted if name not in player_rows.columns]
        if missing:
            raise ValueError(
                f"market_adp_features is on but {missing} are absent from the "
                "player rows. These frames predate the feature -- rebuild the "
                "cache rather than fitting a model that would silently drop it "
                "and report the baseline as a null result"
            )

        def merged(existing: tuple[str, ...]) -> tuple[str, ...]:
            return tuple(dict.fromkeys((*existing, *wanted)))

        self.snap_model.extra_features = merged(self.snap_model.extra_features)
        self.target_role_model.extra_features = merged(
            self.target_role_model.extra_features
        )
        self.carry_eligibility_model.extra_features = merged(
            self.carry_eligibility_model.extra_features
        )
        self.target_model.extra_features = merged(self.target_model.extra_features)
        self.carry_model.extra_features = merged(self.carry_model.extra_features)

    AVAILABILITY_TARGETS = {
        "roster": ("games", "observed_availability"),
        "snap": ("snap_games", "snap_availability"),
    }

    def _apply_availability_target(self, player_rows: pd.DataFrame) -> None:
        """Point the availability and snap layers at the same exposure.

        These two settings are one decision. The availability model fits a
        count out of ``team_games``; the snap model divides the observed season
        snap share by the matching fraction to get a per-game rate; at
        prediction time the pipeline multiplies that rate back by the
        availability draws. Setting one without the other divides by one
        exposure and multiplies by another, and nothing downstream would raise
        -- the projection would simply be wrong by the ratio between them,
        which for undrafted quarterbacks is a factor of 2.6.
        """
        try:
            games_column, availability_column = self.AVAILABILITY_TARGETS[
                self.availability_target
            ]
        except KeyError:
            raise ValueError(
                f"availability_target must be one of "
                f"{sorted(self.AVAILABILITY_TARGETS)}, got "
                f"{self.availability_target!r}"
            ) from None
        missing = [
            name
            for name in (games_column, availability_column)
            if name not in player_rows.columns
        ]
        if missing:
            raise ValueError(
                f"availability_target={self.availability_target!r} needs "
                f"{missing}, absent from the player rows. These frames predate "
                "the column -- rebuild the cache rather than fitting against an "
                "exposure that is not there"
            )
        # Present but empty is the legacy snap source, which has season totals
        # and no per-week rows to count. The column exists so the schema is
        # stable; it carries nothing, and fitting against it would make every
        # player unavailable.
        empty = [
            name
            for name in (games_column, availability_column)
            if not pd.to_numeric(player_rows[name], errors="coerce").notna().any()
        ]
        if empty:
            raise ValueError(
                f"availability_target={self.availability_target!r} needs "
                f"{empty}, which are present but wholly missing. The legacy "
                "snap source cannot count weeks with a snap; build the frames "
                "from nflverse to use this target"
            )
        self.availability_model.games_column = games_column
        self.snap_model.availability_column = availability_column

    def _enable_availability_history(self, player_rows: pd.DataFrame) -> None:
        """Append the career availability mean to the availability regression.

        Raises rather than dropping silently. ``_matrix`` keeps only features
        present in the frame, so a cache built before the pathway features
        would fit a model nobody chose and report nothing about it.
        """
        missing = [
            name
            for name in AVAILABILITY_HISTORY_FEATURES
            if name not in player_rows.columns
        ]
        if missing:
            raise ValueError(
                f"availability_history_features is on but {missing} are absent "
                "from the player rows. Rebuild the cache, or set "
                "availability_history_features=False to fit the single-season "
                "layer deliberately rather than by accident"
            )
        self.availability_model.extra_features = tuple(
            dict.fromkeys(
                (
                    *self.availability_model.extra_features,
                    *AVAILABILITY_HISTORY_FEATURES,
                )
            )
        )

    def _enable_market_adp_availability(self, player_rows: pd.DataFrame) -> None:
        """Append preseason consensus to the availability regression only.

        Separate from ``_enable_market_adp_features`` because the two answer
        different questions. That one asks whether the board knows something
        about *roles* the usage history does not; this asks whether it knows
        something about *who stays on the field*.

        The same absent-column check, for the same reason: ``_matrix`` drops
        names it cannot find, so a cache built before these columns existed
        would fit the baseline and report a clean null.
        """
        missing = [name for name in ADP_FEATURES if name not in player_rows.columns]
        if missing:
            raise ValueError(
                f"market_adp_availability_features is on but {missing} are "
                "absent from the player rows. These frames predate the feature "
                "-- rebuild the cache rather than fitting a model that would "
                "silently drop it and report the baseline as a null result"
            )
        self.availability_model.extra_features = tuple(
            dict.fromkeys((*self.availability_model.extra_features, *ADP_FEATURES))
        )

    def _enable_market_adp_qb(self, player_rows: pd.DataFrame) -> None:
        """Append preseason consensus to the quarterback room.

        ``QBStarterModel`` is deliberately left out. It reads a fixed feature
        list with no ``extra_features`` hook, and more to the point it is a
        within-room categorical over who starts -- the same question the
        workload softmax already answers continuously, from the same inputs.
        Adding the board to both would put the same evidence in twice and make
        an attributable result unattributable.
        """
        missing = [name for name in ADP_FEATURES if name not in player_rows.columns]
        if missing:
            raise ValueError(
                f"market_adp_qb_features is on but {missing} are absent from "
                "the player rows. These frames predate the feature -- rebuild "
                "the cache rather than fitting a model that would silently "
                "drop it and report the baseline as a null result"
            )

        def merged(existing: tuple[str, ...]) -> tuple[str, ...]:
            return tuple(dict.fromkeys((*existing, *ADP_FEATURES)))

        self.workload_model.extra_features = merged(self.workload_model.extra_features)
        self.qb_propensity_model.extra_features = merged(
            self.qb_propensity_model.extra_features
        )

    def _enable_regime_likelihood_features(self) -> None:
        """Append the leakage-safe regime contract to upstream submodels."""

        def merged(existing: tuple[str, ...]) -> tuple[str, ...]:
            return tuple(dict.fromkeys((*existing, *REGIME_LIKELIHOOD_FEATURES)))

        self.availability_model.extra_features = merged(
            self.availability_model.extra_features
        )
        self.snap_model.extra_features = merged(self.snap_model.extra_features)
        self.workload_model.extra_features = merged(self.workload_model.extra_features)
        self.target_role_model.extra_features = merged(
            self.target_role_model.extra_features
        )
        self.carry_eligibility_model.extra_features = merged(
            self.carry_eligibility_model.extra_features
        )
        self.target_model.extra_features = merged(self.target_model.extra_features)
        self.carry_model.extra_features = merged(self.carry_model.extra_features)

    def predict_samples(
        self, data: SeasonAverageData, *, games=None, seed: int = 0
    ) -> SeasonAveragePrediction:
        prediction_rows = data.player_rows
        # Propagate here rather than in ``fit``: the flag only changes how
        # fitted shares are allocated at prediction time, so a pipeline loaded
        # from disk honours it too.
        for share_model in (self.target_model, self.carry_model, self.workload_model):
            share_model.per_game_allocation = self.per_game_allocation
        regime_probability: np.ndarray | None = None
        if self.regime_likelihood_features:
            if self.regime_model is None:
                raise RuntimeError("fit the upstream regime challenger before predicting")
            prediction_rows = add_regime_probabilities(prediction_rows, self.regime_model)
            regime_probability = self.regime_model.predict_proba(prediction_rows)
        availability: AvailabilityPrediction = self.availability_model.predict_samples(
            prediction_rows, seed=seed + 1
        )
        team_games = pd.to_numeric(
            availability.rows.get("team_games", pd.Series(17, index=availability.rows.index)),
            errors="coerce",
        ).fillna(17).to_numpy(dtype=float)
        active_fraction = availability.games_active / team_games[:, None]
        if self.snap_model.idata is None:
            snap_share = active_fraction
        else:
            snap = self.snap_model.predict_samples(
                availability.rows,
                active_fraction_samples=active_fraction,
                seed=seed + 2,
            )
            snap_share = snap.snap_share
        passing = self.workload_model.predict_share_samples(
            availability.rows,
            availability_samples=availability.availability,
            seed=seed + 3,
        )
        qb_workload_share = passing.shares.copy()
        if not availability.rows[PLAYER_ID_COLUMNS].equals(
            passing.rows[PLAYER_ID_COLUMNS]
        ):
            raise ValueError("availability and QB workload rows are not aligned")
        if self.qb_propensity_model.idata is None:
            qb_propensity = np.where(
                availability.rows["position"].eq("QB").to_numpy()[:, None],
                1.0,
                0.0,
            ) * np.ones_like(passing.shares)
        else:
            propensity = self.qb_propensity_model.predict_samples(
                availability.rows, seed=seed + 4
            )
            qb_propensity = propensity.propensity
            passing = _reweight_roster_prediction(
                passing, qb_propensity, support_position="QB"
            )
        if self.carry_eligibility_model.idata is None:
            carry_probability = np.ones_like(passing.shares)
            carry_eligible = carry_probability
        else:
            eligibility = self.carry_eligibility_model.predict_samples(
                availability.rows, seed=seed + 5
            )
            carry_probability = eligibility.probability
            carry_eligible = eligibility.eligible
        if self.target_role_model.idata is None:
            target_probability = np.where(
                availability.rows["position"].isin(("RB", "WR", "TE")).to_numpy()[
                    :, None
                ],
                1.0,
                0.0,
            ) * np.ones_like(passing.shares)
            target_eligible = target_probability
        else:
            target_role = self.target_role_model.predict_samples(
                availability.rows, seed=seed + 11
            )
            target_probability = target_role.probability
            target_eligible = target_role.eligible
        team = self.team_model.predict_average_samples(
            data.team_rows, games=games, seed=seed
        )
        target = self.target_model.predict_share_samples(
            availability.rows,
            snap_samples=snap_share,
            eligibility_samples=target_eligible,
            seed=seed + 6,
        )
        carry = self.carry_model.predict_share_samples(
            availability.rows,
            snap_samples=snap_share,
            eligibility_samples=carry_eligible,
            seed=seed + 7,
        )
        regime_prediction: SeasonRegimePrediction | None = None
        if self.role_regime_coupling:
            if self.regime_model is None or self.regime_coupler is None:
                raise RuntimeError("fit the role-regime challenger before predicting")
            regime_prediction = self.regime_model.predict_samples(
                availability.rows,
                draws=passing.shares.shape[1],
                seed=seed + 12,
            )
            regime_identifiers = regime_prediction.rows[PLAYER_ID_COLUMNS].reset_index(
                drop=True
            )
            if not (
                regime_identifiers.equals(passing.rows[PLAYER_ID_COLUMNS].reset_index(drop=True))
                and regime_identifiers.equals(target.rows[PLAYER_ID_COLUMNS].reset_index(drop=True))
                and regime_identifiers.equals(carry.rows[PLAYER_ID_COLUMNS].reset_index(drop=True))
            ):
                raise ValueError("regime and role-allocation rows are not aligned")
            passing.shares = self.regime_coupler.apply(
                passing.rows,
                passing.shares,
                regime_prediction.samples,
                stream="pass",
                group_index=passing.rows["_group_idx"].to_numpy(int),
            )
            target.shares = self.regime_coupler.apply(
                target.rows,
                target.shares,
                regime_prediction.samples,
                stream="target",
                group_index=target.rows["_group_idx"].to_numpy(int),
            )
            carry.shares = self.regime_coupler.apply(
                carry.rows,
                carry.shares,
                regime_prediction.samples,
                stream="carry",
                group_index=carry.rows["_group_idx"].to_numpy(int),
            )
        pass_group = passing.rows["_group_idx"].to_numpy(dtype=int)
        target_group = target.rows["_group_idx"].to_numpy(dtype=int)
        carry_group = carry.rows["_group_idx"].to_numpy(dtype=int)
        # Convert latent roster shares into the actual predictive quantity:
        # integer season counts. This correctly assigns non-zero probability
        # to zero-volume seasons for fringe players before averaging by games.
        pass_totals = _align_group_draws(
            passing.group_keys, team["rows"], team["pass_attempts"]
        )
        target_totals = _align_group_draws(
            target.group_keys, team["rows"], team["targets"]
        )
        carry_totals = _align_group_draws(
            carry.group_keys, team["rows"], team["rush_attempts"]
        )
        pass_counts = _allocate_season_counts(
            passing, pass_totals, seed=seed + 8
        )
        target_counts = _allocate_season_counts(
            target, target_totals, seed=seed + 9
        )
        carry_counts = _allocate_season_counts(
            carry, carry_totals, seed=seed + 10
        )
        if self.role_regime_coupling:
            _assert_group_count_conservation(passing, pass_counts, pass_totals, "pass")
            _assert_group_count_conservation(target, target_counts, target_totals, "target")
            _assert_group_count_conservation(carry, carry_counts, carry_totals, "carry")
        pass_games = _align_group_draws(
            passing.group_keys, team["rows"], team["games"]
        )[pass_group, None]
        target_games = _align_group_draws(
            target.group_keys, team["rows"], team["games"]
        )[target_group, None]
        carry_games = _align_group_draws(
            carry.group_keys, team["rows"], team["games"]
        )[carry_group, None]
        passes_team_game = pass_counts / pass_games
        targets_team_game = target_counts / target_games
        carries_team_game = carry_counts / carry_games
        identifiers = passing.rows[PLAYER_ID_COLUMNS].reset_index(drop=True)
        if not identifiers.equals(
            target.rows[PLAYER_ID_COLUMNS].reset_index(drop=True)
        ) or not identifiers.equals(
            carry.rows[PLAYER_ID_COLUMNS].reset_index(drop=True)
        ):
            raise ValueError("pass, target, and carry roster rows are not aligned")
        allocation_availability = np.clip(availability.availability, 0.01, 1.0)
        return SeasonAveragePrediction(
            team=team,
            player_rows=passing.rows,
            availability_probability=availability.probability,
            games_active=availability.games_active,
            availability=availability.availability,
            snap_share=snap_share,
            qb_workload_share=qb_workload_share,
            qb_pass_propensity=qb_propensity,
            target_role_probability=target_probability,
            carry_eligibility_probability=carry_probability,
            pass_attempt_share=passing.shares,
            target_share=target.shares,
            carry_share=carry.shares,
            pass_attempts=pass_counts,
            targets=target_counts,
            carries=carry_counts,
            pass_attempts_per_team_game=passes_team_game,
            targets_per_team_game=targets_team_game,
            carries_per_team_game=carries_team_game,
            pass_attempts_per_active_game=passes_team_game / allocation_availability,
            targets_per_active_game=targets_team_game / allocation_availability,
            carries_per_active_game=carries_team_game / allocation_availability,
            regime_probability=(
                regime_probability
                if regime_probability is not None
                else (None if regime_prediction is None else regime_prediction.probability)
            ),
            regime_samples=(
                None if regime_prediction is None else regime_prediction.samples
            ),
        )

    def diagnostics(self, *, min_bulk_ess: float = 100.0):
        availability_variables = (
            [
                "any_intercept",
                "any_position_effect",
                "any_beta",
                "rate_intercept",
                "rate_position_effect",
                "rate_beta",
                "rate_concentration",
            ]
            if "any_intercept" in self.availability_model.idata.posterior
            else ["intercept", "position_effect", "beta", "concentration"]
        )
        return {
            "team": sampling_quality(
                self.team_model.idata,
                [
                    "play_intercept",
                    "play_persistence",
                    "play_alpha_pg",
                    *(
                        ["play_transition_sd"]
                        if self.team_model.models_play_transition
                        else []
                    ),
                    "pass_intercept",
                    "pass_persistence",
                    "pass_transition_sd",
                    *(
                        ["sack_intercept", "sack_persistence"]
                        if self.team_model.models_sacks
                        else []
                    ),
                    "target_intercept",
                    "target_persistence",
                    "target_transition_sd",
                ],
                min_bulk_ess=min_bulk_ess,
            ),
            "workload": sampling_quality(
                self.workload_model.idata,
                ["beta"],
                min_bulk_ess=min_bulk_ess,
            ),
            **(
                {
                    "snap": sampling_quality(
                        self.snap_model.idata,
                        ["intercept", "position_effect", "beta", "concentration"],
                        min_bulk_ess=min_bulk_ess,
                    ),
                    "qb_propensity": sampling_quality(
                        self.qb_propensity_model.idata,
                        ["intercept", "beta", "concentration"],
                        min_bulk_ess=min_bulk_ess,
                    ),
                    "carry_eligibility": sampling_quality(
                        self.carry_eligibility_model.idata,
                        ["intercept", "position_effect", "beta"],
                        min_bulk_ess=min_bulk_ess,
                    ),
                }
                if self.snap_model.idata is not None
                and self.qb_propensity_model.idata is not None
                and self.carry_eligibility_model.idata is not None
                else {}
            ),
            **(
                {
                    "target_role": sampling_quality(
                        self.target_role_model.idata,
                        ["intercept", "position_effect", "beta"],
                        min_bulk_ess=min_bulk_ess,
                    )
                }
                if self.target_role_model.idata is not None
                else {}
            ),
            "availability": sampling_quality(
                self.availability_model.idata,
                availability_variables,
                min_bulk_ess=min_bulk_ess,
            ),
            "target": sampling_quality(
                self.target_model.idata,
                ["beta"],
                min_bulk_ess=min_bulk_ess,
            ),
            "carry": sampling_quality(
                self.carry_model.idata,
                ["beta"],
                min_bulk_ess=min_bulk_ess,
            ),
        }

    def save(self, directory: str | Path) -> Path:
        """Persist all posteriors and prediction-time feature metadata."""
        if any(
            model.idata is None
            for model in (
                self.team_model,
                self.availability_model,
                self.workload_model,
                self.target_model,
                self.carry_model,
            )
        ):
            raise RuntimeError("fit all season-average models before saving")
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        save_idata(self.team_model.idata, directory / "team.nc")
        save_idata(self.availability_model.idata, directory / "availability.nc")
        save_idata(self.workload_model.idata, directory / "workload.nc")
        save_idata(self.target_model.idata, directory / "target.nc")
        save_idata(self.carry_model.idata, directory / "carry.nc")
        optional_models = {
            "snap": self.snap_model,
            "qb_propensity": self.qb_propensity_model,
            "target_role": self.target_role_model,
            "carry_eligibility": self.carry_eligibility_model,
        }
        for name, model in optional_models.items():
            if model.idata is not None:
                save_idata(model.idata, directory / f"{name}.nc")
        metadata = {
            "architecture_version": 6,
            "team": {
                "teams": self.team_model.teams,
                "season_mean": self.team_model.season_mean,
                "play_prior_center": self.team_model.play_prior_center,
                "pass_prior_center": self.team_model.pass_prior_center,
                "sack_prior_center": self.team_model.sack_prior_center,
                "target_prior_center": self.team_model.target_prior_center,
                "models_sacks": self.team_model.models_sacks,
                "models_play_transition": self.team_model.models_play_transition,
                "sum_to_zero_team_effects": self.team_model.sum_to_zero_team_effects,
            },
            "target": self._share_metadata(self.target_model),
            "carry": self._share_metadata(self.carry_model),
            "availability": self._feature_metadata(self.availability_model),
            "workload": {
                **self._feature_metadata(self.workload_model),
                "role_innovation_scale": self.workload_model.role_innovation_scale,
                # The hurdle's availability term is standardised on the training
                # fold, so its centre and scale are fitted state. Without them a
                # reloaded pipeline would gate on (x - 0) / 1 and shift every
                # gate probability while raising no error.
                "hurdle_min_attempts": self.workload_model.hurdle_min_attempts,
                "couple_gate_to_availability": bool(
                    self.workload_model.couple_gate_to_availability
                ),
                "hurdle_availability_mean": self.workload_model.hurdle_availability_mean,
                "hurdle_availability_scale": self.workload_model.hurdle_availability_scale,
                "mean_preserving_innovation": bool(
                    self.workload_model.mean_preserving_innovation
                ),
                "calibrated_innovation": bool(
                    self.workload_model.calibrated_innovation
                ),
            },
            "role_regime": {
                "enabled": self.role_regime_coupling,
                **(
                    {
                        "model": self.regime_model.state_dict(),
                        "coupler": self.regime_coupler.state_dict(),
                    }
                    if self.role_regime_coupling
                    and self.regime_model is not None
                    and self.regime_coupler is not None
                    else {}
                ),
            },
            # Recorded so a served artifact says which inputs it was fitted on.
            # Prediction does not need it -- the fitted feature names, fills and
            # projection round-trip on each submodel -- but an artifact that
            # reads the market and does not say so is one nobody can audit.
            "market_adp": {
                "enabled": self.market_adp_features,
                "interactions": self.market_adp_interactions,
                "availability": self.market_adp_availability_features,
                "qb": self.market_adp_qb_features,
            },
            # Which exposure the availability and snap layers were fitted
            # against. A restored artifact that silently reverted to roster
            # games would divide by one exposure and multiply by another.
            "availability_target": self.availability_target,
            "market_win_total": self.market_win_total_features,
            "regime_likelihood": {
                "enabled": self.regime_likelihood_features,
                **(
                    {"model": self.regime_model.state_dict()}
                    if self.regime_likelihood_features and self.regime_model is not None
                    else {}
                ),
            },
        }
        for name, model in optional_models.items():
            if model.idata is not None:
                metadata[name] = self._feature_metadata(model)
        (directory / "metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
        )
        return directory

    @staticmethod
    def _share_metadata(model: SeasonRosterShareModel) -> dict[str, object]:
        return {
            "stream": model.stream,
            "players": model.players,
            "feature_names": model.feature_names,
            "feature_fill": model.feature_fill,
            "feature_mean": model.feature_mean,
            "feature_scale": model.feature_scale,
            "cold_role_prior": model.cold_role_prior,
            "availability_prior": model.availability_prior,
            "role_innovation_scale": model.role_innovation_scale,
            "per_snap_weight": model.per_snap_weight,
            "innovation_cap": model.innovation_cap,
            "mean_preserving_innovation": bool(model.mean_preserving_innovation),
            "calibrated_innovation": bool(model.calibrated_innovation),
            "cold_role_innovation": bool(model.cold_role_innovation),
            "cold_role_scale_mode": model.cold_role_scale_mode,
            "cold_role_multiplier": float(model.cold_role_multiplier),
            "cold_role_multiplier_cap": float(model.cold_role_multiplier_cap),
            "extra_features": list(model.extra_features),
        }

    @staticmethod
    def _feature_metadata(model) -> dict[str, object]:
        state = {
            "feature_names": model.feature_names,
            "feature_fill": model.feature_fill,
            "feature_mean": model.feature_mean,
            "feature_scale": model.feature_scale,
        }
        if hasattr(model, "positions"):
            state["positions"] = model.positions
        if hasattr(model, "extra_features"):
            state["extra_features"] = list(model.extra_features)
        if hasattr(model, "position_specific_concentration"):
            state["position_specific_concentration"] = bool(
                model.position_specific_concentration
            )
        if hasattr(model, "feature_projection"):
            projection = model.feature_projection
            state["feature_projection"] = (
                None if projection is None else np.asarray(projection).tolist()
            )
        return state

    @classmethod
    def load(cls, directory: str | Path) -> "SeasonAverageVolumePipeline":
        """Restore a fitted season-average pipeline."""
        directory = Path(directory)
        metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        architecture_version = int(metadata.get("architecture_version", 1))
        team = TeamSeasonAverageModel(**metadata["team"])
        team.idata = load_idata(directory / "team.nc")
        availability_state = metadata["availability"]
        availability = SeasonAvailabilityModel(
            positions=list(availability_state["positions"])
        )
        cls._restore_feature_metadata(availability, availability_state)
        availability.idata = load_idata(directory / "availability.nc")
        workload_state = metadata["workload"]
        workload = QBWorkloadShareModel(
            role_innovation_scale=float(workload_state["role_innovation_scale"]),
            hurdle_min_attempts=int(workload_state.get("hurdle_min_attempts", 25)),
            couple_gate_to_availability=bool(
                workload_state.get("couple_gate_to_availability", False)
            ),
            hurdle_availability_mean=float(
                workload_state.get("hurdle_availability_mean", 0.0)
            ),
            hurdle_availability_scale=float(
                workload_state.get("hurdle_availability_scale", 1.0)
            ),
            mean_preserving_innovation=bool(
                workload_state.get("mean_preserving_innovation", False)
            ),
            calibrated_innovation=bool(
                workload_state.get("calibrated_innovation", False)
            ),
        )
        cls._restore_feature_metadata(workload, workload_state)
        workload.idata = load_idata(directory / "workload.nc")
        shares = []
        for name in ("target", "carry"):
            state = metadata[name]
            model = SeasonRosterShareModel(
                state["stream"],
                extra_features=tuple(state.get("extra_features", ())),
                per_snap_weight=float(
                    state.get(
                        "per_snap_weight",
                        (
                            0.75 if state["stream"] == "target" else 1.0
                        )
                        if architecture_version >= 2
                        else 0.0,
                    )
                ),
                innovation_cap=float(
                    state.get(
                        "innovation_cap",
                        0.50 if state["stream"] in {"target", "carry"} else 2.0,
                    )
                ),
                mean_preserving_innovation=bool(
                    state.get("mean_preserving_innovation", False)
                ),
                calibrated_innovation=bool(
                    state.get("calibrated_innovation", False)
                ),
                cold_role_innovation=bool(
                    state.get("cold_role_innovation", False)
                ),
                cold_role_scale_mode=str(
                    state.get("cold_role_scale_mode", "relative")
                ),
                cold_role_multiplier_cap=float(
                    state.get("cold_role_multiplier_cap", 6.0)
                ),
            )
            model.players = list(state["players"])
            model.feature_names = list(state["feature_names"])
            for attribute in (
                "feature_fill",
                "feature_mean",
                "feature_scale",
                "cold_role_prior",
                "availability_prior",
            ):
                setattr(
                    model,
                    attribute,
                    {key: float(value) for key, value in state[attribute].items()},
                )
            model.role_innovation_scale = float(
                state.get("role_innovation_scale", model.role_innovation_scale)
            )
            model.cold_role_multiplier = float(
                state.get("cold_role_multiplier", model.cold_role_multiplier)
            )
            model.idata = load_idata(directory / f"{name}.nc")
            shares.append(model)
        optional = {}
        for name, model_class in (
            ("snap", SeasonSnapShareModel),
            ("qb_propensity", QBPassPropensityModel),
            ("target_role", SeasonTargetRoleModel),
            ("carry_eligibility", SeasonCarryEligibilityModel),
        ):
            model = model_class()
            if name in metadata and (directory / f"{name}.nc").exists():
                state = metadata[name]
                if "positions" in state:
                    model.positions = list(state["positions"])
                cls._restore_feature_metadata(model, state)
                model.idata = load_idata(directory / f"{name}.nc")
            optional[name] = model
        regime_state = metadata.get("role_regime", {})
        role_regime_coupling = bool(regime_state.get("enabled", False))
        regime_model = None
        regime_coupler = None
        if role_regime_coupling:
            if "model" not in regime_state or "coupler" not in regime_state:
                raise ValueError("saved role-regime pipeline is missing its prediction state")
            regime_model = SeasonRegimeModel.from_state(regime_state["model"])
            regime_coupler = SeasonRegimeRoleCoupling.from_state(regime_state["coupler"])
        likelihood_state = metadata.get("regime_likelihood", {})
        regime_likelihood_features = bool(likelihood_state.get("enabled", False))
        likelihood_regime_model = None
        if regime_likelihood_features:
            if "model" not in likelihood_state:
                raise ValueError("saved regime-likelihood pipeline is missing its classifier")
            likelihood_regime_model = SeasonRegimeModel.from_state(likelihood_state["model"])
        return cls(
            team_model=team,
            availability_model=availability,
            workload_model=workload,
            target_model=shares[0],
            carry_model=shares[1],
            snap_model=optional["snap"],
            qb_propensity_model=optional["qb_propensity"],
            target_role_model=optional["target_role"],
            carry_eligibility_model=optional["carry_eligibility"],
            role_regime_coupling=role_regime_coupling,
            market_adp_features=bool(
                metadata.get("market_adp", {}).get("enabled", False)
            ),
            market_adp_interactions=bool(
                metadata.get("market_adp", {}).get("interactions", False)
            ),
            market_adp_availability_features=bool(
                metadata.get("market_adp", {}).get("availability", False)
            ),
            market_adp_qb_features=bool(
                metadata.get("market_adp", {}).get("qb", False)
            ),
            availability_target=str(metadata.get("availability_target", "roster")),
            market_win_total_features=bool(metadata.get("market_win_total", False)),
            regime_likelihood_features=regime_likelihood_features,
            # Each allocation layer carries its own flag, so the pipeline-level
            # one only has to agree with what was actually restored.
            mean_preserving_innovation=workload.mean_preserving_innovation,
            calibrated_innovation=workload.calibrated_innovation,
            # The quarterback workload layer has no cold-role split, so the
            # pipeline-level flag reads from an allocator that does.
            cold_role_innovation=shares[0].cold_role_innovation,
            regime_model=(likelihood_regime_model or regime_model),
            regime_coupler=regime_coupler,
        )

    @staticmethod
    def _restore_feature_metadata(model, state: dict[str, object]) -> None:
        model.feature_names = list(state["feature_names"])
        if hasattr(model, "extra_features"):
            model.extra_features = tuple(state.get("extra_features", ()))
        if hasattr(model, "position_specific_concentration"):
            model.position_specific_concentration = bool(
                state.get("position_specific_concentration", False)
            )
        if hasattr(model, "feature_projection"):
            projection = state.get("feature_projection")
            model.feature_projection = (
                None if projection is None else np.asarray(projection, dtype=float)
            )
        for attribute in (
            "feature_fill",
            "feature_mean",
            "feature_scale",
        ):
            setattr(
                model,
                attribute,
                {key: float(value) for key, value in state[attribute].items()},
            )


PLAYER_ID_COLUMNS = ["season", "team", "player_key"]


def _per_game_shares(
    eta: np.ndarray,
    live: np.ndarray,
    exposure: np.ndarray,
    *,
    games: int,
    seed: int,
) -> np.ndarray:
    """Season share as the average of per-game allocations, not one allocation.

    The default path forms ``softmax(log w + log e)``: it multiplies a player's
    weight by his season-average availability and renormalises once. The
    quantity that stands for is the season average of what happens each week --
    the players who are actually there split the ball -- and those are not the
    same number, because softmax is nonlinear and ``softmax(E[presence]) !=
    E[softmax(presence)]``.

    The gap is signed and it is not small. For a two-player room it is available
    in closed form: a player whose full-strength share is ``s`` and who is
    present a fraction ``e`` of games is over-allocated by ``1 / (1 - s(1-e))``.
    That is 1.05x for a dilute receiver at 10% of a room, 1.45x for a workhorse
    back at 67%, and 1.72x for a starting quarterback at 92% -- the error grows
    with concentration, because a player who *is* most of the denominator
    shrinks the denominator when he sits, and hands himself back most of what he
    lost.

    Simulating presence removes the approximation rather than tuning it: draw
    who is available each week, allocate among them, average over weeks. The
    cost is one softmax per game instead of one per season.
    """
    if games <= 0:
        raise ValueError("allocation_games must be positive")
    rng = np.random.default_rng(seed)
    total = np.zeros_like(eta)
    for _ in range(games):
        present = rng.random(eta.shape) < exposure
        # A week in which a room empties has to allocate its carries to
        # somebody, so fall back to the roster the season-average path would
        # have used rather than dropping the team's volume on the floor.
        available = live & present
        empty = ~available.any(axis=1, keepdims=True)
        total += simplex_shares(eta, np.where(empty, live, available))
    return total / games


def _roster_sample_matrix(
    design: dict[str, object],
    samples: np.ndarray,
    *,
    fill: float,
    min_value: float = 1e-5,
) -> np.ndarray:
    """Place flat player-draw samples into the ragged roster tensor."""
    matrix = np.full(
        (design["mask"].shape[0], design["mask"].shape[1], samples.shape[1]),
        fill,
        dtype=float,
    )
    for group_i in range(design["mask"].shape[0]):
        active = design["mask"][group_i].astype(bool)
        indices = design["row_idx"][group_i, active]
        matrix[group_i, active] = samples[indices]
    return np.clip(matrix, min_value, 1.0)


def _reweight_roster_prediction(
    prediction: RosterSharePrediction,
    weights: np.ndarray,
    *,
    support_position: str | None = None,
) -> RosterSharePrediction:
    """Multiply roster shares by a second modeled rate and renormalize."""
    weights = np.asarray(weights, dtype=float)
    if weights.shape != prediction.shares.shape:
        raise ValueError("roster reweighting samples must align to shares")
    adjusted = prediction.shares * np.clip(weights, 0.0, None)
    if support_position is not None:
        support = prediction.rows["position"].eq(support_position).to_numpy()[:, None]
        adjusted = np.where(support, adjusted, 0.0)
    group_idx = prediction.rows["_group_idx"].to_numpy(dtype=int)
    for group in range(len(prediction.group_keys)):
        indices = np.flatnonzero(group_idx == group)
        totals = adjusted[indices].sum(axis=0)
        missing = totals <= 0
        if missing.any():
            fallback = prediction.shares[indices]
            if support_position is not None:
                fallback = np.where(
                    prediction.rows.loc[indices, "position"]
                    .eq(support_position)
                    .to_numpy()[:, None],
                    fallback,
                    0.0,
                )
            adjusted[np.ix_(indices, np.flatnonzero(missing))] = fallback[:, missing]
            totals = adjusted[indices].sum(axis=0)
        adjusted[indices] /= totals
    return RosterSharePrediction(
        rows=prediction.rows,
        group_keys=prediction.group_keys,
        shares=adjusted,
    )


def _align_group_draws(
    group_keys: pd.DataFrame, team_rows: pd.DataFrame, values: np.ndarray
) -> np.ndarray:
    lookup = pd.MultiIndex.from_frame(team_rows[GROUP_KEYS])
    requested = pd.MultiIndex.from_frame(group_keys[GROUP_KEYS])
    positions = lookup.get_indexer(requested)
    if (positions < 0).any():
        raise ValueError("player roster contains a team-season absent from team predictions")
    return values[positions]


def _allocate_season_counts(
    prediction: RosterSharePrediction, totals: np.ndarray, *, seed: int
) -> np.ndarray:
    """Allocate integer team-season totals over each coherent roster draw."""
    totals = np.asarray(totals)
    draws = prediction.shares.shape[1]
    if totals.shape != (len(prediction.group_keys), draws):
        raise ValueError("team totals must align to roster groups and posterior draws")
    rng = np.random.default_rng(seed)
    group_idx = prediction.rows["_group_idx"].to_numpy(dtype=int)
    n_groups = len(prediction.group_keys)

    # One multinomial call for every (group, draw) at once. ``rng.multinomial``
    # broadcasts an array of trial counts against a matrix of probability rows,
    # so the groups x draws Python loop — around 57,600 calls per prediction at
    # 32 team-seasons, 600 draws and three streams — collapses into a single
    # vectorized draw. Rooms are ragged, so they are padded to the widest and
    # the unused slots carry probability zero, which contributes no counts.
    members = [np.flatnonzero(group_idx == group) for group in range(n_groups)]
    width = max((len(indices) for indices in members), default=0)
    if width == 0:
        return np.zeros_like(prediction.shares, dtype=int)

    probability = np.zeros((n_groups, width, draws), dtype=float)
    for group, indices in enumerate(members):
        probability[group, : len(indices)] = prediction.shares[indices]
    total = probability.sum(axis=1, keepdims=True)
    probability = np.divide(
        probability, total, out=np.zeros_like(probability), where=total > 0
    )

    # (group, slot, draw) -> (group * draw, slot), the layout multinomial wants.
    flat_p = np.moveaxis(probability, 1, 2).reshape(n_groups * draws, width)
    flat_n = np.asarray(totals, dtype=np.int64).reshape(n_groups * draws)
    drawn = rng.multinomial(flat_n, flat_p)
    drawn = np.moveaxis(drawn.reshape(n_groups, draws, width), 2, 1)

    counts = np.zeros_like(prediction.shares, dtype=int)
    for group, indices in enumerate(members):
        counts[indices] = drawn[group, : len(indices)]
    return counts


def _assert_group_count_conservation(
    prediction: RosterSharePrediction,
    counts: np.ndarray,
    totals: np.ndarray,
    stream: str,
) -> None:
    """Fail fast if an experimental role coupling breaks team-count accounting."""

    group_index = prediction.rows["_group_idx"].to_numpy(dtype=int)
    observed = np.vstack(
        [counts[group_index == group].sum(axis=0) for group in range(len(prediction.group_keys))]
    )
    if not np.array_equal(observed, np.asarray(totals, dtype=int)):
        raise AssertionError(f"{stream} regime coupling violated team-count conservation")
