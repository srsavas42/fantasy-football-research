# Does the agent's edge survive a league that plays as well as it does?

*September 2026*

**Verdict: a self-play training run is not worth doing.** 97% of arm D's win
margin survives the hardest field available, and the one thing that field
genuinely takes away — value from an uncontested waiver wire — does not convert
into wins. The measurement cost an afternoon; the training run it rules out
would have cost 4.3x the episode budget for a target inside the noise.

## The question

Every number reported for the trained agent was measured against one field:
`SeasonPolicy` in the other eleven seats — the draft board for three weeks, then
a one-game average — and a wire nobody else ever competes for. Arm D is the most
capacious thing trained here (a hidden layer, 107 parameters), and capacity is
exactly what lets a policy learn an opponent's blind spots instead of the game.
So **+1.656 wins** was consistent with two stories that matter very differently:

- **general skill** — D sets better lineups and makes better claims, and would
  against anybody
- **a field-specific exploit** — D found that `SeasonPolicy` is three weeks
  behind on role changes and never contests a pickup, and the margin is a
  statement about the opponent rather than about fantasy football

Absolute wins cannot separate these. A league is zero-sum: make all twelve teams
better and every record slides toward 7-7 by construction. So the measured
quantity here is a **margin**, run in three fields. In each, two policies play
the same seat — the trained agent, and the `ewma2` control it has always been
scored against — and what is reported is the paired difference between them. A
margin that holds as the field improves is skill. A margin that collapses was
rent.

## The three fields

| field | the other eleven seats |
|---|---|
| `standard` | `SeasonPolicy` lineups; touch the wire only when a rule forces it. What every prior number was measured against. |
| `lineups` | The trained policy sets their cards. Still no strategic claims. |
| `full` | The trained policy sets their cards **and** transacts — same claim rule, one add per team per phase, same rolling priority queue, same 48-hour lock. |

90 seats (2023/2024/2025 x 30 seeds), 540 episodes, everything paired on
(season, seed) — the same seeds in all three fields, so the fields are paired
against each other as well as the seats.

### Making it a mirror rather than a rigged fight

Two asymmetries had to be removed before any of this meant anything, and both
would have biased the answer toward "the edge collapses":

1. **The agent's fresh claim was protected from its own housekeeping; an
   opponent's was not.** Left alone, every opponent would claim a player and cut
   him in the same phase — making the field worse for a reason that has nothing
   to do with policy.
2. **The agent sees a capped `_waiver_shortlist`; the hook was handing opponents
   the entire wire.** A field with a strictly larger choice set than the seat
   under test is not a mirror, it is a harder game.

A sanity check that the plumbing is right falls out of the design: the agent's
**points are identical to the decimal** between `standard` and `lineups`
(1646.699 both times). What opponents start cannot change your own score, only
whether you beat them. Points move only in `full`, where opponents take players
off the wire.

## The result

| field | seat | wins | points | rank | titles |
|---|---|---:|---:|---:|---:|
| `standard` | learned | 8.756 | 1646.7 | 3.51 | 20.0% |
| | ewma2 | 6.800 | 1467.5 | 6.76 | 4.4% |
| `lineups` | learned | 8.489 | 1646.7 | 3.40 | 26.7% |
| | ewma2 | 6.200 | 1467.5 | 7.86 | 3.3% |
| `full` | learned | 7.578 | 1601.3 | 5.01 | 17.8% |
| | ewma2 | 5.678 | 1461.8 | 8.06 | 6.7% |

Absolute wins compress exactly as a zero-sum league says they must — the learned
seat falls 8.76 → 7.58 and the control 6.80 → 5.68 — which is why the margin is
the thing to read:

| field | margin (wins) | SE | t | margin (points) | t | share of standard |
|---|---:|---:|---:|---:|---:|---:|
| `standard` | **+1.956** | 0.176 | +11.12 | +179.2 | +16.43 | 100% |
| `lineups` | **+2.289** | 0.171 | +13.38 | +179.2 | +16.43 | **117%** |
| `full` | **+1.900** | 0.231 | +8.22 | +139.5 | +13.63 | **97%** |

Positive and significant in all nine season-by-field cells. The edge is not a
`SeasonPolicy` exploit.

The margin **grows** when opponents get better at lineups (+0.333 wins,
t=+2.19, paired). That is not a paradox: the agent's own points are unchanged,
so a field that scores more hurts a weak seat more than a strong one — the
control's wins fall twice as fast as the learned agent's.

## The one thing a competent field does take away

Contesting the wire is the only place the agent measurably loses something, and
there the effect is real:

**Points lost to a contested wire (`full` − `lineups`, same seeds):**

| seat | points | t |
|---|---:|---:|
| learned | **−45.4** | **−4.26** |
| ewma2 | −5.7 | −0.55 |

The learned agent loses eight times what the control does, because it was the
one extracting value from a wire nobody else wanted. That is the exploit, it is
genuine, and it is worth about a quarter of its points margin.

**But it does not become wins.** How the margin itself moves, paired:

| comparison | wins | t | points | t |
|---|---:|---:|---:|---:|
| `lineups` − `standard` | +0.333 | +2.19 | 0.000 | — |
| `full` − `lineups` | −0.389 | −1.67 | −39.7 | −3.00 |
| **`full` − `standard`** | **−0.056** | **−0.22** | −39.7 | −3.00 |

Against the field every prior number was reported on, the hardest field costs
**0.056 wins, t = −0.22**. Nothing. The wire loss (−0.389 wins at t=−1.67, below
the bar on its own) is very nearly cancelled by the +0.333 the agent gains when
opponents set better cards.

## Why the training run is not worth it

1. **The premise fails.** Self-play training exists to stop a policy from
   overfitting a fixed opponent. 97% of the margin survives, so there is
   very little overfitting to remove.
2. **The identified headroom is inside the noise.** The one real effect is
   −39.7 points of margin. As wins that is −0.056 against the reported
   baseline, on a measurement whose SE is 0.23.
3. **It costs 4.3x.** Timed on 2024, three seeds each: `standard` 1.61
   s/episode, `lineups` 1.93, `full` **6.86**. The same compute buys roughly
   four times as many generations against the standard field, or four more
   arms — either of which has a better expected return than chasing 0.056 wins.

## What this does *not* establish

The field here is a **frozen copy of D playing D's own policy**. It cannot
discover a counter-strategy, so this rules out *"D's edge is specific to
`SeasonPolicy`"* — which it decisively does — but it says nothing about whether
D is robust to an adversary **trained against it**. That is a different and much
larger build: a population of policies, an outer loop, and a way to stop the
pair from co-adapting into a corner. Nothing measured here argues for paying
that cost, but nothing measured here rules out that an adaptive adversary would
find something a frozen one cannot.

## What shipped

`FantasyLeagueEnv` gained an `opponent_claims` hook. `None` is the old
environment exactly — verified in `tests/test_league_selfplay.py`, which also
pins that an always-declining hook is bit-identical to no hook at all, that the
48-hour lock binds opponents, and that an opponent's claim survives its own
housekeeping. `LinearAgent.claim` was refactored into `claim_for`, so self-play
runs the agent's real decision from another seat rather than a re-implementation
of it.

No trained artifact changed. `scripts/selfplay_agent.py` and
`reports/league_selfplay.json` are the run.
