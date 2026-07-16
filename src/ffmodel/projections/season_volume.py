"""Next-season volume projections and the breakout report.

Turns the fitted cross-season share models (`models.volume_season`) into the two
Phase 3A deliverables:
  * `project_next_season` — a per-player posterior distribution of next-season
    target / carry / opportunity share (P10/P50/P90 + mean), the input to
    pre-season EV and the draft pillar.
  * `breakout_report` — returning players ranked by the posterior probability
    that their opportunity share rises by more than a threshold, plus the
    mirror-image decline list.

Opportunity share is the sum of the target- and carry-share posteriors. We add
the sample matrices directly; this treats the two as independent, a reasonable
v1 approximation (a player rarely swings both sharply in the same direction).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ffmodel.models.volume_season import BetaShareModel


def _opportunity_samples(
    transitions: pd.DataFrame, target_model: BetaShareModel,
    carry_model: BetaShareModel | None,
) -> np.ndarray:
    """Per-player opportunity-share samples, shape (n_players, n_draws)."""
    tgt = target_model.predict_samples(transitions)
    if carry_model is None:
        return tgt
    carry = np.zeros_like(tgt)
    is_rb = transitions["position"].isin(carry_model.positions).to_numpy()
    if is_rb.any():
        carry[is_rb] = carry_model.predict_samples(transitions[is_rb])
    return tgt + carry


def project_next_season(
    transitions: pd.DataFrame,
    target_model: BetaShareModel,
    carry_model: BetaShareModel | None = None,
    qs=(0.1, 0.5, 0.9),
) -> pd.DataFrame:
    """Per-player next-season share projection with uncertainty bands."""
    opp = _opportunity_samples(transitions, target_model, carry_model)
    out = transitions[["player_name", "position", "team_next", "transition"]].copy()
    out["prior_opp_share"] = transitions["opportunity_share"].to_numpy()
    out["proj_opp_mean"] = opp.mean(axis=1)
    for q in qs:
        out[f"proj_opp_p{int(q * 100)}"] = np.quantile(opp, q, axis=1)
    # Also carry the target-share projection through for receiving-only use.
    tgt = target_model.predict_quantiles(transitions, qs=qs)
    out["proj_target_p50"] = tgt["p50"].to_numpy()
    return out.reset_index(drop=True)


def breakout_report(
    transitions: pd.DataFrame,
    target_model: BetaShareModel,
    carry_model: BetaShareModel | None = None,
    threshold: float = 0.05,
    min_prior_games: int = 4,
) -> pd.DataFrame:
    """Rank returning players by P(next opp share - prior > threshold).

    `threshold` is an absolute opportunity-share increase (0.05 = +5 percentage
    points of team opportunity). Adds P(decline) for the same threshold so the
    tail risk is visible alongside the upside.
    """
    df = transitions
    if "games" in df.columns:
        df = df[df["games"] >= min_prior_games]
    opp = _opportunity_samples(df, target_model, carry_model)
    prior = df["opportunity_share"].to_numpy()[:, None]
    delta = opp - prior

    report = df[["player_name", "position", "team_next", "transition"]].copy()
    report["prior_opp_share"] = prior[:, 0]
    report["proj_opp_p50"] = np.quantile(opp, 0.5, axis=1)
    report["exp_delta"] = delta.mean(axis=1)
    report["p_breakout"] = (delta > threshold).mean(axis=1)
    report["p_decline"] = (delta < -threshold).mean(axis=1)
    return report.sort_values("p_breakout", ascending=False).reset_index(drop=True)
