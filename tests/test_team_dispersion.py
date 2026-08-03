"""One likelihood, one dispersion term.

``plays_obs`` is NegativeBinomial — a Poisson-Gamma mixture that already carries
overdispersion through ``play_alpha_pg``. Adding a per-row log-normal term to
the same mean gives the likelihood two dispersion sources that the data cannot
separate. That is what a variance posterior pinned at 0.008 with r-hat 1.02 and
an intercept effective sample size of 376 looks like.

The pass, sack and target likelihoods are Binomial, which has no free dispersion
parameter, so their transition terms are doing real work and stay unconditional.
"""

import pytest

from ffmodel.models.volume_season_average import TeamSeasonAverageModel


def test_the_redundant_play_dispersion_term_is_off_by_default():
    assert TeamSeasonAverageModel().models_play_transition is False


def test_it_can_be_restored_for_ablation():
    assert TeamSeasonAverageModel(models_play_transition=True).models_play_transition


@pytest.mark.slow
def test_dropping_it_lets_the_play_block_converge():
    """The point of the change: r-hat clears the gate.

    Fitted on synthetic team-seasons whose play rates carry no extra
    row-level noise beyond the NegativeBinomial's own, which is what the real
    data turned out to look like.
    """
    import numpy as np
    import pandas as pd

    pytest.importorskip("pymc")
    import arviz as az

    rng = np.random.default_rng(3)
    records = []
    for season in range(2015, 2024):
        for team in [f"T{i:02d}" for i in range(32)]:
            prior = rng.normal(62.0, 3.0)
            plays = int(rng.negative_binomial(60, 60 / (60 + prior)) + prior * 16)
            records.append(
                {
                    "season": season, "team": team, "games": 16,
                    "opportunity_plays": plays,
                    "pass_attempts": int(plays * 0.58),
                    "targets": int(plays * 0.55), "sacks": 0,
                    "sacks_observed": False,
                    "valid_target_pass_attempts": int(plays * 0.58),
                    "valid_targets": int(plays * 0.55),
                    "prior_opportunity_plays_per_game": prior,
                    "prior_pass_rate": 0.58, "prior_target_rate": 0.95,
                }
            )
    rows = pd.DataFrame(records)

    model = TeamSeasonAverageModel().fit(rows, draws=400, tune=400, chains=2)
    worst = float(
        az.summary(model.idata, var_names=["play_intercept", "play_persistence"])
        ["r_hat"].max()
    )

    assert "play_transition_sd" not in model.idata.posterior
    assert worst < 1.05
