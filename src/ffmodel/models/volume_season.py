"""Hierarchical Beta model for next-season opportunity share (returning players).

Predicts a returning player's season-Y+1 share (target share or carry share)
from season-Y signals. Shares live in [0, 1], so the likelihood is a Beta whose
mean is a logit-linear function of the predictors, with intercepts partially
pooled across positions (small-sample positions borrow strength). The posterior
predictive gives each player a full next-season share distribution, from which
projection quantiles and breakout probabilities are read off directly.

Predictors (all from season Y):
  prior_share_logit  logit of the player's season-Y share (regression to mean)
  late_share_logit   logit of the weeks>=10 share (a late-year role change that
                     projects to a full season)
  vacated            share freed on the Y+1 team by departed players
  age_c, age_c2      centered age and its square (position age curve)
  team_change        1 if the player switched teams
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ffmodel.models.base import logit, sample_model, squeeze_unit

PREDICTORS = ["prior_share_logit", "late_share_logit", "vacated", "competition",
              "age_c", "age_c2", "team_change"]


@dataclass
class BetaShareModel:
    """A fitted next-season share model plus everything needed to predict."""

    target_col: str          # "next_target_share" or "next_carry_share"
    prior_col: str           # "target_share" or "carry_share"
    late_col: str            # "late_target_share" or "late_carry_share"
    vacated_col: str         # "vacated_target_share" or "vacated_carry_share"
    comp_col: str            # "incoming_comp_target" or "incoming_comp_carry"
    positions: list[str] = field(default_factory=list)
    age_mean: float = 26.0
    idata: object = None

    # ---- design matrix ---------------------------------------------------
    def _design(self, df: pd.DataFrame, fit: bool = False):
        d = df.copy()
        if fit:
            self.age_mean = float(d["age"].mean())
            self.positions = sorted(d["position"].unique())
        d["age"] = d["age"].fillna(self.age_mean)
        age_c = (d["age"] - self.age_mean) / 10.0
        X = pd.DataFrame(
            {
                "prior_share_logit": logit(d[self.prior_col]),
                "late_share_logit": logit(d[self.late_col]),
                "vacated": d[self.vacated_col].fillna(0.0).to_numpy(),
                "competition": d.get(
                    self.comp_col, pd.Series(0.0, index=d.index)
                ).fillna(0.0).to_numpy(),
                "age_c": age_c.to_numpy(),
                "age_c2": (age_c ** 2).to_numpy(),
                "team_change": d.get("team_change", pd.Series(0, index=d.index)).to_numpy(),
            }
        )
        pos_idx = pd.Categorical(d["position"], categories=self.positions).codes
        return X[PREDICTORS].to_numpy(dtype=float), pos_idx

    # ---- fit -------------------------------------------------------------
    def fit(self, transitions: pd.DataFrame, **sample_kwargs) -> "BetaShareModel":
        import pymc as pm

        df = transitions.dropna(subset=[self.target_col, self.prior_col]).copy()
        X, pos_idx = self._design(df, fit=True)
        y = squeeze_unit(df[self.target_col].to_numpy())
        n_pos, n_pred = len(self.positions), X.shape[1]

        with pm.Model() as model:
            # Position-varying intercept, partially pooled (non-centered to keep
            # NUTS geometry well-conditioned and avoid divergences).
            mu_a = pm.Normal("mu_a", 0.0, 1.5)
            sd_a = pm.HalfNormal("sd_a", 1.0)
            z_a = pm.Normal("z_a", 0.0, 1.0, shape=n_pos)
            alpha = pm.Deterministic("alpha", mu_a + z_a * sd_a)
            # Population slopes. Share is strongly persistent year-to-year, so
            # the prior-share slope (predictor 0) is centered at 1.0 — the model
            # starts from persistence and adjusts for age / vacated / late-season.
            beta_mu = np.zeros(n_pred)
            beta_mu[PREDICTORS.index("prior_share_logit")] = 1.0
            beta_sd = np.full(n_pred, 1.0)
            beta_sd[PREDICTORS.index("prior_share_logit")] = 0.4
            beta = pm.Normal("beta", mu=beta_mu, sigma=beta_sd, shape=n_pred)
            phi = pm.Gamma("phi", alpha=2.0, beta=0.1)  # Beta precision

            eta = alpha[pos_idx] + pm.math.dot(X, beta)
            mu = pm.math.invlogit(eta)
            pm.Beta("obs", alpha=mu * phi, beta=(1 - mu) * phi, observed=y)

            sample_kwargs.setdefault("target_accept", 0.95)
            self.idata = sample_model(model, **sample_kwargs)
        return self

    # ---- predict ---------------------------------------------------------
    def predict_samples(self, transitions: pd.DataFrame) -> np.ndarray:
        """Posterior samples of next-season share, shape (n_players, n_draws)."""
        X, pos_idx = self._design(transitions, fit=False)
        post = self.idata.posterior
        alpha = post["alpha"].stack(s=("chain", "draw")).to_numpy()   # (n_pos, S)
        beta = post["beta"].stack(s=("chain", "draw")).to_numpy()     # (n_pred, S)
        phi = post["phi"].stack(s=("chain", "draw")).to_numpy()       # (S,)

        eta = alpha[pos_idx, :] + X @ beta                            # (n_players, S)
        mu = 1.0 / (1.0 + np.exp(-eta))
        rng = np.random.default_rng(0)
        a = np.clip(mu * phi[None, :], 1e-6, None)
        b = np.clip((1 - mu) * phi[None, :], 1e-6, None)
        return rng.beta(a, b)

    def predict_quantiles(self, transitions: pd.DataFrame,
                          qs=(0.1, 0.5, 0.9)) -> pd.DataFrame:
        samples = self.predict_samples(transitions)
        out = transitions[["player_name", "position"]].copy()
        out["pred_mean"] = samples.mean(axis=1)
        for q in qs:
            out[f"p{int(q * 100)}"] = np.quantile(samples, q, axis=1)
        return out.reset_index(drop=True)


def fit_target_share(transitions: pd.DataFrame, **kw) -> BetaShareModel:
    return BetaShareModel(
        "next_target_share", "target_share", "late_target_share",
        "vacated_target_share", "incoming_comp_target",
    ).fit(transitions, **kw)


def fit_carry_share(transitions: pd.DataFrame, positions=("RB",), **kw) -> BetaShareModel:
    # Carries are ~0 for WR/TE; restrict to backfield positions.
    sub = transitions[transitions["position"].isin(positions)].copy()
    return BetaShareModel(
        "next_carry_share", "carry_share", "late_carry_share",
        "vacated_carry_share", "incoming_comp_carry",
    ).fit(sub, **kw)
