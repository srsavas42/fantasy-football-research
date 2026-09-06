# A week is not one deadline

*September 2026. Code: `src/ffmodel/league/kickoff.py`, `lineup.py`, `env.py`,
`scripts/validate_slots.py`. Numbers: `reports/league_slots.json`.*

The environment used to take one lineup per week, locked before any game kicked
off. Real leagues do not work that way: **every player locks when his own game
starts**, so the Sunday afternoon flex is still open while the morning games are
being played, and Monday night is open all weekend. That is now how a week
resolves.

Whether it *matters* is a separate question with a specific answer, and the
answer is no — for a reason worth understanding.

---

## The mechanism

A regular-season week has **6.8 distinct kickoff times** on average (min 5, max
9). A fifteen-man roster spans about 4.9 of them, which leaves ~3.9 decision
points after the first, of which ~2.9 offer a genuine choice — two or more
unlocked players who could fill the same slot. Every week of the three holdout
seasons has at least one.

Kickoff times come from nflverse's schedule and are cached; a week is indexed
into slots, and a player belongs to his club's slot. The environment then
resolves a week slot by slot: decide, play that slot's games, decide again with
those results banked.

`optimal_lineup` gained two constraints — `locked_in` (already playing, stays in
the card) and `locked_out` (already played, can never enter it). Only the *set*
of starters is constrained, not which slot each occupies: a back already playing
counts the same whether the card calls him RB2 or the flex.

**Sequencing is a rule, available to every team.** A policy declares
`reactive = True` if it wants the later decision points; the environment
re-scores only those, so a week costs what it always did for everybody else. And
it is exactly equivalent for them: re-running the same scoring function on the
same information returns the same lineup, which is checked as a
bit-identical-episode test rather than asserted.

## What the extra deadlines are worth: nothing, yet

The value of a second deadline is the value of information that arrives before
it. Between kickoffs, exactly one thing does: **the score**. Nobody learns
anything about a player's Sunday afternoon by watching Sunday morning.

Knowing the score matters only because the league pays a win bonus rather than
paying points — thirty behind with one game left wants the volatile player,
thirty ahead wants the steady one. So six variants of that idea were measured
against deciding once, 3 seasons × 20 seeds, paired:

| policy | wins | SE | t | points |
|---|---|---|---|---|
| chase-buy 0.25 | −0.033 | 0.033 | −1.00 | +3.9 |
| chase-buy 0.5 | −0.133 | 0.056 | −2.40 | −1.1 |
| chase 0.25 | −0.167 | 0.064 | −2.62 | +6.1 |
| chase-resid 0.25 | −0.167 | 0.068 | −2.45 | +6.3 |
| chase 0.5 | −0.267 | 0.092 | −2.91 | −0.1 |
| chase-resid 0.5 | −0.317 | 0.094 | −3.38 | −2.6 |

The best is **−0.03 wins, indistinguishable from zero**, and every stronger tilt
is worse. Chasing the win does not pay here.

Two things had to be ruled out before believing that.

**A units bug that looked like a finding.** The first tilt added raw volatility —
fantasy points, running to fifteen — to a score that is a linear combination of
standardised features and lives within about one unit. The tilt was **3.1× the
entire spread of the score**, so it did not adjust the ranking, it replaced it
with a volatility ranking. Standardising it makes `strength` dimensionless, and
the sweep above is over the fixed version.

**Volatility is mostly just quality.** It correlates **+0.64** with the agent's
own score: the players who swing hardest are largely the good ones. So "sell
variance when ahead" is substantially "bench your best players", which is why
several variants *gain points and lose games* — the exact opposite of what
variance-chasing is supposed to trade. Residualising volatility against the
score removes that, and the residualised version is no better, which is what
makes the null a null rather than a broken heuristic.

## The option is valuable; the information is not

An agent given **perfect knowledge of whoever has not yet kicked off**, at every
decision point after the first, gains **+2.83 wins and +264 points** (t = 14.2).

That is not a ceiling on sequencing, and reading it as one would be wrong. Only
two clubs play before Sunday, so by the second decision point nearly the whole
roster is still movable and this is close to a full oracle — it measures
projection quality, not deadlines.

What it does say is that the *option* the extra deadlines create is wide open.
Nothing about the current information exploits it, because nothing informative
arrives. It becomes worth something the moment something does:

- **Official inactives**, published 90 minutes before kickoff. The environment's
  `is_out` comes from the Wednesday–Friday game-status report; a gameday scratch
  is a different feed and is not in the panel. This is the obvious next data
  acquisition, and unlike score-chasing it is real information about a player
  rather than about the scoreboard.
- **Late weather.** The gated wind hinge measured earlier is worth −0.60% CRPS
  above 15 mph on the rows it touches, and a forecast six hours out is better
  than one on Wednesday.

Until one of those lands, the sequencing is correct, free, and inert. That is a
fine thing for it to be — it is a rule the environment previously got wrong, and
getting it right costs nothing and stops the question being asked again.
