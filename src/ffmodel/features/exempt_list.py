"""How long a commissioner's exempt-list placement costs, as a distribution.

The exempt list is a holding state, not a punishment: a player is paid, stays on
the roster, does not count against the limit, and may not play, while the league
and usually a court work out what happens next. It has no announced length, so
unlike a suspension it cannot be subtracted as arithmetic. What follows is the
best that the available data supports, and the honest description of that is
"weakly informative", not "estimated".

Three things make this a small-sample problem rather than a regression.

**The status pools unrelated events.** ``EXE`` is not a synonym for the
commissioner's exempt list. The same status carries contract holdouts (Aaron
Donald, week 1 of 2017; Trent Williams, 2019), un-retirements (Jared Veldheer
2019, Frank Ragnow 2025), suspension appeals (Kareem Jackson 2023) and, in
2020, COVID-19 roster mechanics. The roster feed carries no field that separates
these from a conduct placement. :func:`exempt_episodes` therefore filters on
*shape* rather than on a reason it cannot see -- the one-week placements are
overwhelmingly the mechanical ones -- and exposes the filter so it can be
argued with.

**The sample is tens, not thousands.** After removing 2020 and the one-week
placements there are roughly a dozen episodes across ten seasons. Nothing with
covariates is identifiable on that. The model here has exactly one parameter.

**The longest episodes are censored.** A player still on the list in week 18 has
an unknown true duration, and those are precisely the cases that matter most for
a projection. Dropping them would bias every estimate downward; treating their
observed length as complete would do the same. They enter the likelihood as
survivals.

The model is a constant weekly hazard -- each week on the list carries the same
probability that the matter resolves -- which makes duration geometric and gives
a Beta posterior in closed form. A constant hazard is memoryless, and that
assumption is worth stating plainly because it is doing real work: it says a
player eight weeks into a legal process is no more and no less likely to be
cleared next week than one who was placed on Tuesday. With a dozen episodes
there is no power to estimate a shape on top of that, so the alternative is not
a better-fitting model but a less honest one.

The prior is the policy. The NFL's Personal Conduct Policy sets a six-game
baseline for the violations that draw an exempt placement, so the default prior
is centred there and weighted at six pseudo-weeks -- informative enough to keep
a three-episode sample from running away, weak enough that a dozen real
episodes move it. Read :attr:`ExemptListModel.prior_events` and
``prior_survived`` as "one resolution observed over six weeks of waiting".

Two things the fitted model does not tell you about itself, both measured by
``scripts/model_exempt_list.py``:

**The identification filter dominates the answer.** ``min_weeks`` is a judgment
call about which placements are conduct cases, and on 2016-2025 it moves the
mean more than any other choice in this module::

    min_weeks=1   n=41   mean 3.3 games   (pools holdouts and un-retirements)
    min_weeks=2   n=13   mean 7.2 games   (default)
    min_weeks=3   n=7    mean 8.9 games

A 2.7x swing from an unfalsifiable choice is the honest headline of this
model, and it is larger than the posterior width at any single setting. Quote
the range, not the point.

**The tail is heavier than the data.** A posterior predictive check on the
uncensored episodes puts the observed mean, median and quartiles comfortably
inside the predictive distribution (Bayesian p between 0.55 and 0.83), but the
observed *maximum* sits at p=0.95: the geometric generates longer worst cases
than have actually occurred. With one season-long episode in the sample that is
not strong evidence of misfit, and it does mean the upper quantiles are the
least trustworthy part of the output. Treat ``p_misses_season`` as an upper
bound.

What this is not: a forecast of a particular case. Whether a specific player
misses two games or a season turns on facts -- charges, a court calendar, an
appeal -- that no roster feed contains. Use the distribution to size the
uncertainty and the mean to fill ``suspended_games``, and override both when
the case gives you something better.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ffmodel.features.suspensions import (
    EXEMPT_NONDISCIPLINARY_CODES,
    EXEMPT_STATUS,
    _regular_season,
)

# The Personal Conduct Policy's stated baseline, in games. Used only to centre
# the prior; nothing downstream treats it as an observation.
CONDUCT_POLICY_BASELINE = 6

# 2020's exempt placements are COVID-19 roster mechanics and outnumber every
# other season combined.
COVID_SEASON = 2020


def _player_state(rosters: pd.DataFrame) -> pd.Series:
    code = (
        rosters["status_description_abbr"].astype("string").str.upper()
        if "status_description_abbr" in rosters.columns
        else pd.Series(pd.NA, index=rosters.index, dtype="string")
    )
    status = rosters.get("status", pd.Series(pd.NA, index=rosters.index))
    status = status.astype("string").str.upper()
    out = pd.Series("OTHER", index=rosters.index, dtype="string")
    out = out.mask(status.eq("ACT"), "ACTIVE")
    out = out.mask(status.isin(("RES", "INA")), "RESERVE")
    out = out.mask(
        status.eq(EXEMPT_STATUS) & ~code.isin(EXEMPT_NONDISCIPLINARY_CODES), "EXEMPT"
    )
    out = out.mask(code.isin(("R40", "R30")) | status.eq("SUS"), "SUSPENDED")
    return out


def exempt_episodes(
    rosters: pd.DataFrame,
    *,
    min_weeks: int = 2,
    drop_covid_season: bool = True,
    max_week: int = 18,
) -> pd.DataFrame:
    """Exempt-list episodes with duration, censoring and what followed.

    ``min_weeks`` is the identification filter and the main thing to argue
    with. At its default of 2 it keeps multi-week placements and drops the
    one-week ones, which are dominated by holdouts, un-retirements and appeal
    processing rather than conduct. Set it to 1 to see the pooled population,
    which will pull every estimate sharply toward zero.

    ``games_missed`` counts weeks from placement to the end of the episode,
    including any suspension the placement converted into and any week the
    player spent off the roster entirely -- from a projection's point of view
    those are the same lost game. ``censored`` marks an episode still running
    at ``max_week``, whose true length is unknown and longer than what is here.
    """
    empty = pd.DataFrame(
        columns=[
            "season", "player_name", "position", "team", "exempt_weeks",
            "first_week", "last_week", "games_missed", "weeks_remaining",
            "converted_to_suspension", "censored",
        ]
    )
    rows = _regular_season(rosters, max_week=max_week)
    if drop_covid_season:
        rows = rows[rows["season"].ne(COVID_SEASON)]
    if rows.empty:
        return empty
    rows = rows.copy()
    rows["state"] = _player_state(rows)
    name = "full_name" if "full_name" in rows.columns else "player_name"
    season_length = rows.groupby("season")["week"].max()

    touched = rows.loc[rows["state"].eq("EXEMPT"), ["season", name]].drop_duplicates()
    out = []
    for season, player in touched.itertuples(index=False):
        block = rows[rows["season"].eq(season) & rows[name].eq(player)]
        by_week = dict(zip(block["week"], block["state"]))
        weeks = sorted(w for w, s in by_week.items() if s == "EXEMPT")
        first, last = int(weeks[0]), int(weeks[-1])
        end = int(season_length[season])
        # A week the player appears on no roster at all is a lost game too, and
        # for an indefinite absence it is how the feed represents most of it.
        missed = sum(
            1
            for w in range(first, end + 1)
            if by_week.get(w, "ABSENT") in ("EXEMPT", "SUSPENDED", "ABSENT")
        )
        out.append(
            {
                "season": int(season),
                "player_name": player,
                "position": block["position"].iloc[0]
                if "position" in block.columns
                else pd.NA,
                "team": block["team"].iloc[0] if "team" in block.columns else pd.NA,
                "exempt_weeks": len(weeks),
                "first_week": first,
                "last_week": last,
                "games_missed": missed,
                "weeks_remaining": end - first + 1,
                "converted_to_suspension": any(
                    by_week.get(w) == "SUSPENDED" for w in range(last + 1, end + 1)
                ),
                "censored": last >= end,
            }
        )
    if not out:
        return empty
    frame = pd.DataFrame(out)
    frame = frame[frame["exempt_weeks"].ge(min_weeks)]
    return frame.sort_values(["season", "player_name"]).reset_index(drop=True)


@dataclass
class ExemptListModel:
    """Constant-hazard duration model for an exempt-list placement.

    One parameter, a weekly resolution hazard, with a Beta posterior in closed
    form. The prior defaults to the Personal Conduct Policy's six-game baseline
    at a weight of six pseudo-weeks; see the module docstring for why the model
    is this small and what the memorylessness costs.
    """

    # "One resolution observed over six weeks of waiting."
    prior_events: float = 1.0
    prior_survived: float = float(CONDUCT_POLICY_BASELINE)
    posterior_events: float = field(default=0.0, init=False)
    posterior_survived: float = field(default=0.0, init=False)
    episodes: int = field(default=0, init=False)
    censored: int = field(default=0, init=False)

    def fit(self, episodes: pd.DataFrame) -> "ExemptListModel":
        """Accumulate resolutions and weeks-survived from observed episodes.

        An uncensored episode of length k contributes one resolution and k-1
        weeks survived. A censored one contributes no resolution and k weeks
        survived, which is what keeps the open-ended cases from biasing the
        hazard upward.
        """
        if episodes.empty:
            raise ValueError(
                "no exempt episodes to fit; loosen min_weeks or widen the seasons"
            )
        length = pd.to_numeric(episodes["games_missed"], errors="coerce")
        censored = episodes["censored"].astype(bool).to_numpy()
        if length.isna().any() or (length <= 0).any():
            raise ValueError("episode lengths must be positive")
        length = length.to_numpy(dtype=float)
        self.episodes = int(len(episodes))
        self.censored = int(censored.sum())
        self.posterior_events = self.prior_events + float((~censored).sum())
        self.posterior_survived = self.prior_survived + float(
            (length[~censored] - 1.0).sum() + length[censored].sum()
        )
        return self

    @property
    def hazard_mean(self) -> float:
        """Posterior mean weekly probability that the placement resolves."""
        return self.posterior_events / (self.posterior_events + self.posterior_survived)

    def predict_samples(
        self, size: int = 10_000, *, weeks_remaining: int = 18, seed: int = 0
    ) -> np.ndarray:
        """Draw games missed, capped at the games actually left to miss.

        The hazard is drawn from its Beta posterior and the duration from the
        geometric it implies, so the returned spread carries both the
        week-to-week randomness and the fact that a dozen episodes do not pin
        the hazard down.
        """
        if weeks_remaining <= 0:
            raise ValueError("weeks_remaining must be positive")
        if self.posterior_events <= 0:
            raise RuntimeError("fit the model before predicting")
        rng = np.random.default_rng(seed)
        hazard = rng.beta(self.posterior_events, self.posterior_survived, size=size)
        duration = rng.geometric(np.clip(hazard, 1e-6, 1.0))
        return np.minimum(duration, weeks_remaining).astype(int)

    def summary(self, *, weeks_remaining: int = 18, seed: int = 0) -> dict:
        draws = self.predict_samples(weeks_remaining=weeks_remaining, seed=seed)
        return {
            "episodes": self.episodes,
            "censored": self.censored,
            "hazard_mean": round(float(self.hazard_mean), 4),
            "mean_games_missed": round(float(draws.mean()), 2),
            "median_games_missed": int(np.median(draws)),
            "p10": int(np.quantile(draws, 0.10)),
            "p90": int(np.quantile(draws, 0.90)),
            "p_misses_at_least_4": round(float((draws >= 4).mean()), 3),
            "p_misses_season": round(float((draws >= weeks_remaining).mean()), 3),
            "weeks_remaining": weeks_remaining,
        }
