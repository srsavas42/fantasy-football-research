"""Season-wide per-game volume models.

This module estimates stable team-season rates and roster-level player roles.
It intentionally omits matchup covariates: weekly observations are used to
construct season counts, while the latent prediction target is average volume
per game over a full season.

The player likelihood is roster coherent. Quarterback pass attempts are
allocated from a continuous QB offensive-snap workload simplex; target and
carry counts follow logistic-normal Multinomials over every QB/RB/WR/TE on a
team. Prior-season role and active-game share are fitting exposures.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from ffmodel.features.season_average import SeasonAverageData, TEAM_KEYS
from ffmodel.features.volume import MODEL_POSITIONS
from ffmodel.models.base import (
    load_idata,
    logit,
    sample_model,
    sampling_quality,
    save_idata,
)
from ffmodel.models.season_availability import (
    AvailabilityPrediction,
    QBWorkloadShareModel,
    SeasonAvailabilityModel,
)
from ffmodel.models.volume_team import _sum_to_zero_basis

GROUP_KEYS = ["season", "team"]

STREAMS = {
    "pass": {
        "count": "pass_att",
        "role": "prior_pass_role",
        "draft": "draft_pass_prior",
        "fallback": {"QB": 0.800, "RB": 0.0005, "WR": 0.0005, "TE": 0.0002},
    },
    "target": {
        "count": "targets",
        "role": "prior_target_role",
        "draft": "draft_target_prior",
        "fallback": {"QB": 0.002, "RB": 0.055, "WR": 0.085, "TE": 0.050},
    },
    "carry": {
        "count": "rush_att",
        "role": "prior_carry_role",
        "draft": "draft_carry_prior",
        "fallback": {"QB": 0.055, "RB": 0.150, "WR": 0.003, "TE": 0.002},
    },
}

DEFAULT_SACK_RATE = 0.065

ADJUSTMENT_FEATURES = (
    "prior_availability",
    "prior_snap_share",
    "age",
    "experience",
    "team_change",
    "cold_start",
)

# The Bayesian posterior retains only effects that were stable and materially
# identified in the initial full-feature walk-forward fit. Richer nonlinear
# interactions remain the job of the optional XGBoost challenger.
BAYESIAN_FEATURES = (
    "prior_availability",
    "prior_snap_share",
)

BACKUP_QB_EXPOSURE = 0.08


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
    idata: object = None

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
        return d, team_idx, era, play_prior, pass_prior, sack_prior, target_prior

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
        ) = self._design(
            rows, fit=True
        )
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
            play_team = pm.Normal("play_team", 0.0, 0.06, shape=len(self.teams))
            play_transition_sd = pm.HalfNormal("play_transition_sd", 0.12)
            play_transition_z = pm.Normal(
                "play_transition_z", 0.0, 1.0, shape=len(d)
            )
            play_mu_pg = pm.math.exp(
                play_intercept
                + play_persistence * play_prior
                + play_era * era
                + play_team[team_idx]
                + play_transition_z * play_transition_sd
            )
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
            pass_team = pm.Normal("pass_team", 0.0, 0.12, shape=len(self.teams))
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
                sack_team = pm.Normal(
                    "sack_team", 0.0, 0.10, shape=len(self.teams)
                )
                sack_eta = (
                    sack_intercept
                    + sack_persistence * sack_prior
                    + sack_era * era
                    + sack_team[team_idx]
                )
                pm.Binomial(
                    "sacks_obs",
                    n=dropbacks[valid_sack],
                    p=pm.math.sigmoid(sack_eta[valid_sack]),
                    observed=sacks[valid_sack],
                )

            target_intercept = pm.Normal("target_intercept", target_center, 0.30)
            target_persistence = pm.Normal("target_persistence", 0.65, 0.30)
            target_era = pm.Normal("target_era", 0.0, 0.15)
            target_team = pm.Normal("target_team", 0.0, 0.10, shape=len(self.teams))
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
        play_eta = play_eta + rng.normal(size=play_eta.shape) * _stack(
            post, "play_transition_sd"
        )[None, :]
        pass_eta = _stack(post, "pass_intercept")[None, :]
        pass_eta = pass_eta + (
            pass_prior[:, None] * _stack(post, "pass_persistence")[None, :]
        )
        pass_eta = pass_eta + era[:, None] * _stack(post, "pass_era")[None, :]
        pass_eta = pass_eta + self._known_effect(post, "pass_team", team_idx)
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
        target_eta = _stack(post, "target_intercept")[None, :]
        target_eta = target_eta + (
            target_prior[:, None] * _stack(post, "target_persistence")[None, :]
        )
        target_eta = target_eta + era[:, None] * _stack(post, "target_era")[None, :]
        target_eta = target_eta + self._known_effect(post, "target_team", team_idx)
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
    players: list[str] = field(default_factory=list)
    feature_names: list[str] = field(default_factory=list)
    feature_fill: dict[str, float] = field(default_factory=dict)
    feature_mean: dict[str, float] = field(default_factory=dict)
    feature_scale: dict[str, float] = field(default_factory=dict)
    cold_role_prior: dict[str, float] = field(default_factory=dict)
    availability_prior: dict[str, float] = field(default_factory=dict)
    role_innovation_scale: float = 0.75
    idata: object = None

    def __post_init__(self):
        if self.stream not in STREAMS:
            raise ValueError(f"stream must be one of {sorted(STREAMS)}")

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
        self.feature_names = [name for name in BAYESIAN_FEATURES if name in d]
        for name in self.feature_names:
            values = pd.to_numeric(d[name], errors="coerce")
            fill = float(values.median()) if values.notna().any() else 0.0
            filled = values.fillna(fill)
            scale = float(filled.std(ddof=0))
            self.feature_fill[name] = fill
            self.feature_mean[name] = float(filled.mean())
            self.feature_scale[name] = scale if scale > 1e-8 else 1.0

        availability = pd.to_numeric(
            d.get("observed_availability", pd.Series(1.0, index=d.index)),
            errors="coerce",
        ).fillna(1.0)
        opportunity_rate = d[self.count_col].to_numpy(dtype=float) / np.clip(
            availability.to_numpy(dtype=float), 0.03, 1.0
        )
        group_total = pd.Series(opportunity_rate, index=d.index).groupby(
            [d[key] for key in GROUP_KEYS]
        ).transform("sum")
        observed_share = np.divide(
            opportunity_rate,
            group_total.to_numpy(dtype=float),
            out=np.zeros(len(d), dtype=float),
            where=group_total.to_numpy(dtype=float) > 0,
        )
        fallback = STREAMS[self.stream]["fallback"]
        for position in MODEL_POSITIONS:
            is_position = d["position"].to_numpy() == position
            is_cold = pd.to_numeric(
                d.get("cold_start", pd.Series(0, index=d.index)), errors="coerce"
            ).fillna(0).to_numpy(dtype=int) == 1
            values = observed_share[is_position & is_cold]
            if not len(values):
                values = observed_share[is_position]
            estimate = float(np.mean(values)) if len(values) else fallback[position]
            self.cold_role_prior[position] = max(estimate, 1e-4)
            availability = pd.to_numeric(
                d.loc[d["position"] == position, "prior_availability"],
                errors="coerce",
            ).dropna()
            self.availability_prior[position] = (
                float(availability.median()) if len(availability) else 0.75
            )
        self.role_innovation_scale = self._estimate_role_innovation(d)

    def _estimate_role_innovation(self, d: pd.DataFrame) -> float:
        """Training-only RMS log-share error after removing roster location."""
        role = self._role_prior(d)
        availability = pd.to_numeric(
            d.get("observed_availability", pd.Series(1.0, index=d.index)),
            errors="coerce",
        ).fillna(1.0).to_numpy(dtype=float)
        score = np.log(role) + np.log(np.clip(availability, 0.03, 1.0))
        expected = np.zeros(len(d), dtype=float)
        residual = np.zeros(len(d), dtype=float)
        for _, group in d.groupby(GROUP_KEYS, sort=False, dropna=False):
            indices = group.index.to_numpy(dtype=int)
            centered = score[indices] - score[indices].max()
            weights = np.exp(centered)
            expected[indices] = weights / weights.sum()
            counts = d.loc[indices, self.count_col].to_numpy(dtype=float)
            observed = (counts + 0.5) / (counts.sum() + 0.5 * len(indices))
            group_residual = np.log(observed) - np.log(expected[indices])
            residual[indices] = group_residual - group_residual.mean()
        return float(np.clip(np.sqrt(np.mean(residual ** 2)), 0.10, 2.0))

    def _role_prior(self, d: pd.DataFrame) -> np.ndarray:
        role = pd.to_numeric(d.get(STREAMS[self.stream]["role"]), errors="coerce")
        draft = pd.to_numeric(d.get(STREAMS[self.stream]["draft"]), errors="coerce")
        cold = d["position"].map(self.cold_role_prior).astype(float)
        prior = role.where(role > 0)
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
        X = self._matrix(d)
        player_idx = _codes(d["player_key"], self.players)
        if use_observed_availability:
            availability = pd.to_numeric(
                d.get("observed_availability", pd.Series(1.0, index=d.index)),
                errors="coerce",
            ).fillna(1.0)
        else:
            source = d.get(
                "projected_availability",
                d.get("prior_availability", pd.Series(np.nan, index=d.index)),
            )
            availability = pd.to_numeric(source, errors="coerce")
            availability = availability.fillna(d["position"].map(self.availability_prior))
            availability = availability.fillna(0.75)
        availability = np.clip(availability.to_numpy(dtype=float), 0.03, 1.0)
        d["_projected_availability"] = availability
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
            counts[group_i, :size] = d.loc[indices, self.count_col].to_numpy(dtype=int)
            mask[group_i, :size] = 1.0
            role[group_i, :size] = role_prior[indices]
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
            use_observed_starter=self.stream == "pass",
        )
        with pm.Model() as model:
            beta = pm.Normal("beta", 0.0, 0.35, shape=len(self.feature_names))
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
        if availability_samples is None:
            eta = eta + design["availability_offset"][..., None]
        else:
            availability_samples = np.asarray(availability_samples, dtype=float)
            if availability_samples.shape != (len(design["rows"]), draws):
                raise ValueError(
                    "availability samples must align to roster rows and posterior draws"
                )
            eta = eta + np.log(
                _roster_sample_matrix(design, availability_samples, fill=1.0)
            )
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
        eta = np.where(design["mask"][..., None] > 0, eta, -20.0)

        rng = np.random.default_rng(seed)
        innovation_z = rng.normal(
            size=(
                len(design["group_keys"]),
                design["innovation_basis"].shape[2],
                draws,
            )
        )
        innovation = np.einsum(
            "gkj,gjs->gks", design["innovation_basis"], innovation_z
        )
        eta = eta + innovation * self.role_innovation_scale
        eta -= eta.max(axis=1, keepdims=True)
        probability = np.exp(eta) * design["mask"][..., None]
        probability /= probability.sum(axis=1, keepdims=True)
        shares = np.zeros((len(design["rows"]), draws), dtype=float)
        for group_i in range(len(design["group_keys"])):
            active = design["mask"][group_i].astype(bool)
            indices = design["row_idx"][group_i, active]
            shares[indices] = probability[group_i, active]
        return RosterSharePrediction(
            rows=design["rows"], group_keys=design["group_keys"], shares=shares
        )


@dataclass
class SeasonAveragePrediction:
    team: dict[str, object]
    player_rows: pd.DataFrame
    availability_probability: np.ndarray
    games_active: np.ndarray
    availability: np.ndarray
    qb_workload_share: np.ndarray
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
    fit_seconds: dict[str, float] = field(default_factory=dict)

    def fit(self, data: SeasonAverageData, **sample_kwargs) -> "SeasonAverageVolumePipeline":
        started = perf_counter()
        self.team_model.fit(data.team_rows, **sample_kwargs)
        self.fit_seconds["team"] = perf_counter() - started
        started = perf_counter()
        self.availability_model.fit(data.player_rows, **sample_kwargs)
        self.fit_seconds["availability"] = perf_counter() - started
        started = perf_counter()
        self.workload_model.fit(data.player_rows, **sample_kwargs)
        self.fit_seconds["workload"] = perf_counter() - started
        started = perf_counter()
        self.target_model.fit(data.player_rows, **sample_kwargs)
        self.fit_seconds["target"] = perf_counter() - started
        started = perf_counter()
        self.carry_model.fit(data.player_rows, **sample_kwargs)
        self.fit_seconds["carry"] = perf_counter() - started
        return self

    def predict_samples(
        self, data: SeasonAverageData, *, games=None, seed: int = 0
    ) -> SeasonAveragePrediction:
        availability: AvailabilityPrediction = self.availability_model.predict_samples(
            data.player_rows, seed=seed + 1
        )
        passing = self.workload_model.predict_share_samples(
            availability.rows,
            availability_samples=availability.availability,
            seed=seed + 2,
        )
        if not availability.rows[PLAYER_ID_COLUMNS].equals(
            passing.rows[PLAYER_ID_COLUMNS]
        ):
            raise ValueError("availability and QB workload rows are not aligned")
        team = self.team_model.predict_average_samples(
            data.team_rows, games=games, seed=seed
        )
        target = self.target_model.predict_share_samples(
            availability.rows,
            availability_samples=availability.availability,
            seed=seed + 3,
        )
        carry = self.carry_model.predict_share_samples(
            availability.rows,
            availability_samples=availability.availability,
            seed=seed + 4,
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
            passing, pass_totals, seed=seed + 5
        )
        target_counts = _allocate_season_counts(
            target, target_totals, seed=seed + 6
        )
        carry_counts = _allocate_season_counts(
            carry, carry_totals, seed=seed + 7
        )
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
        availability_samples = np.clip(availability.availability, 0.01, 1.0)
        return SeasonAveragePrediction(
            team=team,
            player_rows=passing.rows,
            availability_probability=availability.probability,
            games_active=availability.games_active,
            availability=availability_samples,
            qb_workload_share=passing.shares,
            pass_attempt_share=passing.shares,
            target_share=target.shares,
            carry_share=carry.shares,
            pass_attempts=pass_counts,
            targets=target_counts,
            carries=carry_counts,
            pass_attempts_per_team_game=passes_team_game,
            targets_per_team_game=targets_team_game,
            carries_per_team_game=carries_team_game,
            pass_attempts_per_active_game=passes_team_game / availability_samples,
            targets_per_active_game=targets_team_game / availability_samples,
            carries_per_active_game=carries_team_game / availability_samples,
        )

    def diagnostics(self, *, min_bulk_ess: float = 100.0):
        return {
            "team": sampling_quality(
                self.team_model.idata,
                [
                    "play_intercept",
                    "play_persistence",
                    "play_alpha_pg",
                    "play_transition_sd",
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
            "availability": sampling_quality(
                self.availability_model.idata,
                ["intercept", "position_effect", "beta", "concentration"],
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
        metadata = {
            "team": {
                "teams": self.team_model.teams,
                "season_mean": self.team_model.season_mean,
                "play_prior_center": self.team_model.play_prior_center,
                "pass_prior_center": self.team_model.pass_prior_center,
                "sack_prior_center": self.team_model.sack_prior_center,
                "target_prior_center": self.team_model.target_prior_center,
                "models_sacks": self.team_model.models_sacks,
            },
            "target": self._share_metadata(self.target_model),
            "carry": self._share_metadata(self.carry_model),
            "availability": self._feature_metadata(self.availability_model),
            "workload": {
                **self._feature_metadata(self.workload_model),
                "role_innovation_scale": self.workload_model.role_innovation_scale,
            },
        }
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
        }

    @staticmethod
    def _feature_metadata(model) -> dict[str, object]:
        state = {
            "feature_names": model.feature_names,
            "feature_fill": model.feature_fill,
            "feature_mean": model.feature_mean,
            "feature_scale": model.feature_scale,
        }
        if isinstance(model, SeasonAvailabilityModel):
            state["positions"] = model.positions
        return state

    @classmethod
    def load(cls, directory: str | Path) -> "SeasonAverageVolumePipeline":
        """Restore a fitted season-average pipeline."""
        directory = Path(directory)
        metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
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
            role_innovation_scale=float(workload_state["role_innovation_scale"])
        )
        cls._restore_feature_metadata(workload, workload_state)
        workload.idata = load_idata(directory / "workload.nc")
        shares = []
        for name in ("target", "carry"):
            state = metadata[name]
            model = SeasonRosterShareModel(state["stream"])
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
            model.idata = load_idata(directory / f"{name}.nc")
            shares.append(model)
        return cls(
            team_model=team,
            availability_model=availability,
            workload_model=workload,
            target_model=shares[0],
            carry_model=shares[1],
        )

    @staticmethod
    def _restore_feature_metadata(model, state: dict[str, object]) -> None:
        model.feature_names = list(state["feature_names"])
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


def _roster_sample_matrix(
    design: dict[str, object], samples: np.ndarray, *, fill: float
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
    return np.clip(matrix, 1e-5, 1.0)


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
    counts = np.zeros_like(prediction.shares, dtype=int)
    group_idx = prediction.rows["_group_idx"].to_numpy(dtype=int)
    for group in range(len(prediction.group_keys)):
        indices = np.flatnonzero(group_idx == group)
        for draw in range(draws):
            probability = prediction.shares[indices, draw]
            probability = probability / probability.sum()
            counts[indices, draw] = rng.multinomial(
                int(totals[group, draw]), probability
            )
    return counts
