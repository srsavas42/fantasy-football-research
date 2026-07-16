"""Hierarchical model for team plays, pass attempts, and target totals.

The model is intentionally usable without betting-market inputs.  Team plays
follow a Negative Binomial distribution, pass attempts are conditional on
those plays through a Binomial distribution, and target totals are conditional
on pass attempts. Team offense and (when supplied)
opponent defense effects are partially pooled; home/rest/era terms are small
population-level adjustments.  Posterior simulation returns coherent integer
draws with ``targets <= pass_attempts <= plays``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ffmodel.features.volume import team_game_totals
from ffmodel.models.base import logit, sample_model

KEY_COLUMNS = ["season", "week", "team"]


def prepare_team_weeks(player_weeks: pd.DataFrame) -> pd.DataFrame:
    """Return one model row per team-week from player- or team-grain input.

    Context columns are optional.  In particular, no spread or total is read by
    this model.  ``is_home`` and ``rest_days`` use neutral defaults when absent.
    Callers may provide an ``opponent`` column to activate the hierarchical
    opponent-defense effect.
    """
    required = set(KEY_COLUMNS + ["team_pass_att", "team_rush_att"])
    if required <= set(player_weeks.columns):
        keep = KEY_COLUMNS + [
            c
            for c in (
                "team_pass_att",
                "team_rush_att",
                "team_targets",
                "team_target_support",
                "team_plays",
                "team_pass_rate",
                "team_target_rate",
                "team_opportunity_valid",
                "opponent",
                "is_home",
                "rest_days",
            )
            if c in player_weeks.columns
        ]
        out = player_weeks[keep].drop_duplicates(KEY_COLUMNS).copy()
    else:
        out = team_game_totals(player_weeks)
        optional = [
            c for c in ("opponent", "is_home", "rest_days") if c in player_weeks
        ]
        if optional:
            context = (
                player_weeks[KEY_COLUMNS + optional]
                .drop_duplicates(KEY_COLUMNS)
            )
            out = out.merge(context, on=KEY_COLUMNS, how="left")

    for column in ("team_pass_att", "team_rush_att"):
        out[column] = pd.to_numeric(out[column], errors="coerce").fillna(0.0)
    out["team_plays"] = (
        out["team_pass_att"] + out["team_rush_att"]
    ).round().clip(lower=0).astype(int)
    out["team_pass_att"] = (
        out["team_pass_att"].round().clip(lower=0).astype(int)
    )
    out["team_pass_att"] = np.minimum(out["team_pass_att"], out["team_plays"])
    out["team_pass_rate"] = np.divide(
        out["team_pass_att"],
        out["team_plays"],
        out=np.zeros(len(out), dtype=float),
        where=out["team_plays"].to_numpy() > 0,
    )
    for column in ("team_targets", "team_target_support"):
        if column in out:
            out[column] = (
                pd.to_numeric(out[column], errors="coerce")
                .fillna(0.0)
                .round()
                .clip(lower=0)
                .astype(int)
            )
    if "team_opportunity_valid" in out:
        out["team_opportunity_valid"] = (
            out["team_opportunity_valid"].fillna(True).astype(bool)
        )
    out["is_home"] = pd.to_numeric(
        out.get("is_home", pd.Series(0.0, index=out.index)), errors="coerce"
    ).fillna(0.0)
    out["rest_days"] = pd.to_numeric(
        out.get("rest_days", pd.Series(7.0, index=out.index)), errors="coerce"
    ).fillna(7.0)
    out["season"] = pd.to_numeric(out["season"], errors="raise").astype(int)
    out["week"] = pd.to_numeric(out["week"], errors="raise").astype(int)
    return out[out["team_plays"] > 0].sort_values(KEY_COLUMNS).reset_index(drop=True)


def _codes(values: pd.Series, categories: list[str]) -> np.ndarray:
    lookup = {value: index for index, value in enumerate(categories)}
    return (
        values.astype(str).map(lookup).fillna(-1).to_numpy(dtype=int)
    )


def _stack(posterior, name: str) -> np.ndarray:
    return posterior[name].stack(sample=("chain", "draw")).to_numpy()


@dataclass
class TeamVolumeModel:
    """Fitted joint team-plays/pass-rate model and prediction metadata."""

    teams: list[str] = field(default_factory=list)
    opponents: list[str] = field(default_factory=list)
    season_mean: float = 2020.0
    use_opponent: bool = False
    idata: object = None

    def _design(
        self, frame: pd.DataFrame, *, fit: bool = False, valid_only: bool = False
    ):
        d = prepare_team_weeks(frame)
        if valid_only and "team_opportunity_valid" in d:
            d = d[d["team_opportunity_valid"]].reset_index(drop=True)
        if d.empty:
            raise ValueError("no valid team-weeks are available for the team-volume model")
        if fit:
            self.teams = sorted(d["team"].astype(str).unique())
            has_opponent = "opponent" in d and d["opponent"].notna().any()
            self.use_opponent = bool(has_opponent)
            self.opponents = (
                sorted(d.loc[d["opponent"].notna(), "opponent"].astype(str).unique())
                if has_opponent
                else []
            )
            self.season_mean = float(d["season"].mean())
        team_idx = _codes(d["team"], self.teams)
        if self.use_opponent:
            opponent = d.get("opponent", pd.Series("__unknown__", index=d.index))
            opp_idx = _codes(opponent.fillna("__unknown__"), self.opponents)
        else:
            opp_idx = np.full(len(d), -1, dtype=int)
        X = np.column_stack(
            [
                (d["season"].to_numpy(dtype=float) - self.season_mean) / 5.0,
                d["is_home"].to_numpy(dtype=float),
                (d["rest_days"].to_numpy(dtype=float) - 7.0) / 7.0,
            ]
        )
        return d, X, team_idx, opp_idx

    def fit(self, team_weeks: pd.DataFrame, **sample_kwargs) -> "TeamVolumeModel":
        """Fit both likelihoods in one PyMC posterior."""
        import pymc as pm

        d, X, team_idx, opp_idx = self._design(
            team_weeks, fit=True, valid_only=True
        )
        plays = d["team_plays"].to_numpy(dtype=int)
        passes = d["team_pass_att"].to_numpy(dtype=int)
        target_column = (
            "team_target_support" if "team_target_support" in d else "team_targets"
        )
        if target_column not in d:
            raise ValueError(
                "team-volume fitting requires target totals; pass player-week features "
                "through team_game_totals or build_features first"
            )
        targets = d[target_column].to_numpy(dtype=int)
        if (targets > passes).any():
            raise ValueError("valid team-weeks must satisfy target totals <= team_pass_att")
        play_center = float(np.log(np.clip(plays.mean(), 1.0, None)))
        pass_center = float(logit(np.array([passes.sum() / plays.sum()]))[0])
        target_center = float(
            logit(np.array([targets.sum() / np.clip(passes.sum(), 1, None)]))[0]
        )

        with pm.Model() as model:
            play_intercept = pm.Normal("play_intercept", play_center, 0.35)
            play_team_sd = pm.HalfNormal("play_team_sd", 0.25)
            play_team_z = pm.Normal("play_team_z", 0.0, 1.0, shape=len(self.teams))
            play_team = pm.Deterministic("play_team", play_team_z * play_team_sd)
            play_beta = pm.Normal("play_beta", 0.0, 0.20, shape=X.shape[1])
            play_eta = play_intercept + play_team[team_idx] + pm.math.dot(X, play_beta)

            pass_intercept = pm.Normal("pass_intercept", pass_center, 0.5)
            pass_team_sd = pm.HalfNormal("pass_team_sd", 0.35)
            pass_team_z = pm.Normal("pass_team_z", 0.0, 1.0, shape=len(self.teams))
            pass_team = pm.Deterministic("pass_team", pass_team_z * pass_team_sd)
            pass_beta = pm.Normal("pass_beta", 0.0, 0.30, shape=X.shape[1])
            pass_eta = pass_intercept + pass_team[team_idx] + pm.math.dot(X, pass_beta)

            target_intercept = pm.Normal("target_intercept", target_center, 0.5)
            target_team_sd = pm.HalfNormal("target_team_sd", 0.25)
            target_team_z = pm.Normal("target_team_z", 0.0, 1.0, shape=len(self.teams))
            target_team = pm.Deterministic("target_team", target_team_z * target_team_sd)
            target_beta = pm.Normal("target_beta", 0.0, 0.25, shape=X.shape[1])
            target_eta = (
                target_intercept + target_team[team_idx] + pm.math.dot(X, target_beta)
            )

            if self.use_opponent:
                play_opp_sd = pm.HalfNormal("play_opp_sd", 0.20)
                play_opp_z = pm.Normal(
                    "play_opp_z", 0.0, 1.0, shape=len(self.opponents)
                )
                play_opp = pm.Deterministic("play_opp", play_opp_z * play_opp_sd)
                pass_opp_sd = pm.HalfNormal("pass_opp_sd", 0.25)
                pass_opp_z = pm.Normal(
                    "pass_opp_z", 0.0, 1.0, shape=len(self.opponents)
                )
                pass_opp = pm.Deterministic("pass_opp", pass_opp_z * pass_opp_sd)
                known = opp_idx >= 0
                safe_opp_idx = np.where(known, opp_idx, 0)
                play_eta = play_eta + play_opp[safe_opp_idx] * known
                pass_eta = pass_eta + pass_opp[safe_opp_idx] * known

            play_alpha = pm.Gamma("play_alpha", alpha=2.0, beta=0.1)
            pm.NegativeBinomial(
                "plays_obs", mu=pm.math.exp(play_eta), alpha=play_alpha, observed=plays
            )
            pm.Binomial(
                "passes_obs", n=plays, p=pm.math.sigmoid(pass_eta), observed=passes
            )
            pm.Binomial(
                "targets_obs", n=passes, p=pm.math.sigmoid(target_eta), observed=targets
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

    def predict_samples(self, team_weeks: pd.DataFrame, *, seed: int = 0) -> dict[str, object]:
        """Simulate coherent posterior team plays, passes, and targets."""
        if self.idata is None:
            raise RuntimeError("fit the team-volume model before predicting")
        d, X, team_idx, opp_idx = self._design(team_weeks, fit=False)
        post = self.idata.posterior
        play_eta = _stack(post, "play_intercept")[None, :]
        play_eta = play_eta + self._known_effect(post, "play_team", team_idx)
        play_eta = play_eta + X @ _stack(post, "play_beta")
        pass_eta = _stack(post, "pass_intercept")[None, :]
        pass_eta = pass_eta + self._known_effect(post, "pass_team", team_idx)
        pass_eta = pass_eta + X @ _stack(post, "pass_beta")
        if self.use_opponent:
            play_eta = play_eta + self._known_effect(post, "play_opp", opp_idx)
            pass_eta = pass_eta + self._known_effect(post, "pass_opp", opp_idx)

        mu = np.exp(np.clip(play_eta, -10.0, 10.0))
        alpha = _stack(post, "play_alpha")[None, :]
        prob = alpha / (alpha + mu)
        rng = np.random.default_rng(seed)
        plays = rng.negative_binomial(alpha, prob)
        pass_rate = 1.0 / (1.0 + np.exp(-np.clip(pass_eta, -20.0, 20.0)))
        passes = rng.binomial(plays, pass_rate)
        if "target_intercept" in post:
            target_eta = _stack(post, "target_intercept")[None, :]
            target_eta = target_eta + self._known_effect(post, "target_team", team_idx)
            target_eta = target_eta + X @ _stack(post, "target_beta")
            target_rate = 1.0 / (1.0 + np.exp(-np.clip(target_eta, -20.0, 20.0)))
            targets = rng.binomial(passes, target_rate)
        else:
            # Compatibility for lightweight synthetic posteriors in tests.
            # Fitted models always include the explicit target layer.
            targets = passes.copy()
        return {
            "rows": d,
            "plays": plays,
            "pass_attempts": passes,
            "targets": targets,
        }

    def predict_quantiles(self, team_weeks: pd.DataFrame, qs=(0.1, 0.5, 0.9)) -> pd.DataFrame:
        pred = self.predict_samples(team_weeks)
        out = pred["rows"][KEY_COLUMNS].copy()
        for label, samples in (
            ("plays", pred["plays"]),
            ("pass_att", pred["pass_attempts"]),
            ("targets", pred["targets"]),
        ):
            out[f"{label}_mean"] = samples.mean(axis=1)
            for q in qs:
                out[f"{label}_p{int(q * 100)}"] = np.quantile(samples, q, axis=1)
        return out.reset_index(drop=True)


def fit_team_volume(team_weeks: pd.DataFrame, **sample_kwargs) -> TeamVolumeModel:
    return TeamVolumeModel().fit(team_weeks, **sample_kwargs)
