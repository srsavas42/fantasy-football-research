"""What a waiver claim was actually worth, graded after the season.

A lineup decision grades itself: the points land the same week and the reward
signal is immediate. A waiver claim does not. Its value is spread over every week
that follows -- the receiver picked up in week 6 pays in weeks 9, 11 and 13, and
the back dropped to make room pays the other team instead. Handing the agent the
week's points and calling it a day teaches it nothing about the add, because the
add's effect on that week is usually zero.

So a claim is graded against the rest of the season, in the currency the claim
was made in: **lineup points**. A player who is added and never started is worth
nothing no matter what he scored, because a bench player scores nothing for
anybody. That is the whole reason this is not simply "the added player's points
minus the dropped player's points".

Two formulations, and the difference between them is worth understanding before
picking one.

``marginal`` (the default)
    Replay each remaining week twice: once with the roster as it really was, and
    once in the world where the claim was never made. Score both cards with the
    points players really put up. The credit is the difference. This is the
    honest counterfactual -- it asks what the *swap* was worth, not what the
    player was worth -- and it is the one that does not reward churn: claiming a
    twelve-point receiver to displace an eleven-point receiver is worth one
    point, which is what it is worth.

``gross``
    Tally the added player's points in the weeks he actually started, and
    subtract the dropped player's points in the weeks he would have started. The
    formulation as originally specified, and it is a fine intuition pump, but it
    systematically over-credits: a claim that starts a fifteen-point player who
    displaced a fourteen-point player scores +15 rather than +1, so an agent
    trained on it learns to add anybody startable, constantly.

**This is a training signal, not an observation.** It is computed from weeks the
agent had not played when it made the claim, so the environment can only produce
it because it holds the future. It must never reach a policy through
:meth:`FantasyLeagueEnv.observe`, and nothing here is wired into the observation
-- grading happens after the episode ends, which is the only time it is
available anyway.

**What the counterfactual approximates.** The alternate world keeps every later
decision fixed and only swaps the two players back. It does not re-run the
season: had the claim not been made, the roster housekeeping might have moved
somebody else, and a different week-8 roster might have changed the week-9 claim.
Replaying that faithfully means re-simulating the season per claim, which is both
expensive and unstable -- a tiny change cascades into a different season and the
credit stops being attributable to the claim at all. Holding everything else
fixed is the standard treatment, and the number it produces is "what this swap
was worth given how the season otherwise went".

**Claims are graded one at a time, and the parts do not sum to the whole.** Each
claim is scored against a world where only *it* was reversed, so a season of
fourteen claims produces fourteen numbers whose total is not what the fourteen
claims were jointly worth -- two claims that each cover the same hole will each
be credited with covering it. This is the usual price of marginal attribution and
it is the right trade for a learning signal, which needs per-decision credit
rather than a season-level total. Read the sum as a diagnostic, not as a
season's waiver profit.
"""

from __future__ import annotations

from dataclasses import dataclass

from ffmodel.league.lineup import optimal_lineup, score_lineup
from ffmodel.league.roster import BENCHED

MARGINAL, GROSS = "marginal", "gross"


@dataclass(frozen=True)
class ClaimCredit:
    """One claim, and what the rest of the season said about it."""

    week: int
    add_key: str
    drop_key: str
    marginal: float
    gross: float
    weeks_added_started: int
    weeks_dropped_would_start: int

    def value(self, mode: str = MARGINAL) -> float:
        return self.marginal if mode == MARGINAL else self.gross


def grade_claims(env, mode: str = MARGINAL) -> list[ClaimCredit]:
    """Grade every explicit waiver claim the agent made this episode.

    Only explicit claims the agent made *and that landed*. A claim can lose its
    player to a team ahead of it in the waiver queue, and grading a swap that
    never happened would credit the agent for a roster it does not have. The
    automatic housekeeping in :mod:`ffmodel.league.roster` is excluded for a
    different reason: it is forced by the rules rather than chosen, so crediting
    it would pay the agent for the environment's work.
    """
    if not env.done:
        raise RuntimeError("grade the claims after the season, not during it")
    credits = []
    for entry in env.ledger:
        for claim in entry.get("claims", []):
            credits.append(_grade_one(env, claim, int(entry["week"])))
    return credits


def _grade_one(env, claim, week: int) -> ClaimCredit:
    add_key, drop_key = claim.add_key, claim.drop_key
    marginal = gross_credit = gross_debit = 0.0
    started = would_start = 0

    for entry in env.ledger:
        if int(entry["week"]) < week:
            continue
        current = int(entry["week"])
        actual = env._actual_points(current)
        scores = entry["scores"]
        roster = entry["roster"]

        # The world as it went.
        real = entry["lineup"]
        real_points = score_lineup(real, actual)
        if add_key in real.starting_keys():
            started += 1
            gross_credit += float(actual.get(add_key, 0.0))

        # The world where the swap never happened: the same roster, with the
        # dropped player back in the added player's place. Everything else about
        # the season is held fixed -- see the module docstring.
        #
        # The two rosters must stay the same size, and keeping them that way is
        # not bookkeeping. A later claim may already have cut the player this one
        # brought in; then there is nobody to take back out, the alternate roster
        # is one player larger than the real one, and a larger roster fields a
        # better lineup every remaining week. Left unhandled that made every
        # claim look worse the longer the season ran -- an oracle that claimed
        # the highest scorer available graded *negative*, which is how this was
        # found.
        if drop_key in roster:
            # A later move brought him back. The swap has been undone, and
            # crediting this claim for a roster it no longer caused is double
            # counting against whichever move undid it.
            continue
        alternate = list(roster)
        if add_key in alternate:
            alternate.remove(add_key)
        else:
            # The claim spent a roster spot. In the world without it that spot
            # holds the dropped player rather than whoever is now least valuable,
            # so the least valuable is who steps aside.
            alternate.remove(min(alternate, key=lambda key: (scores.get(key, 0.0), key)))
        alternate.append(drop_key)
        alternate_scores = dict(scores)
        if drop_key not in alternate_scores:
            alternate_scores[drop_key] = _valuation(env, drop_key, current)
        # A player who cannot play cannot be started in either world.
        if not env.availability.is_available(drop_key, current):
            alternate_scores[drop_key] = BENCHED

        counterfactual = optimal_lineup(
            alternate, env.positions, alternate_scores, env.config.slots
        )
        marginal += real_points - score_lineup(counterfactual, actual)
        if drop_key in counterfactual.starting_keys():
            would_start += 1
            gross_debit += float(actual.get(drop_key, 0.0))

    return ClaimCredit(
        week=week,
        add_key=add_key,
        drop_key=drop_key,
        marginal=float(marginal),
        gross=float(gross_credit - gross_debit),
        weeks_added_started=started,
        weeks_dropped_would_start=would_start,
    )


def _valuation(env, key: str, week: int) -> float:
    """What the agent's own opponent model would have said about a player.

    Needed because the dropped player is off the roster and so was never scored.
    Built from weeks strictly before ``week``, so the counterfactual lineup is
    set on the information a manager had, not on the points about to be scored --
    the credit is allowed to use the future, the *decision* inside it is not.
    """
    history = env._history_before(week)
    scored = env.opponent.score([key], history, week, env.board)
    return float(scored.get(key, 0.0))


def total_credit(credits: list[ClaimCredit], mode: str = MARGINAL) -> float:
    return float(sum(credit.value(mode) for credit in credits))
