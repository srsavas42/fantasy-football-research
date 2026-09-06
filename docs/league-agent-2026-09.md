# Training an agent to set lineups and work the wire

*September 2026. Code: `src/ffmodel/league/{features,agent,train}.py`,
`scripts/{train_agent,evaluate_agent}.py`. Numbers:
`reports/league_agent.json`, `reports/league_agent_eval.json`. Parameters:
`artifacts/league_agent.json`.*

Trained on 2016–2022, evaluated on 2023–2025 — the weekly model's own holdout.
Paired on seed throughout, because a season swings ±3 wins on the draft slot
alone.

**Result: +0.59 wins and +30 points a season on seasons it never saw**
(t = 3.59), which is 25% of the 2.33-win band a perfect start/sit would have
taken.

---

## Making an episode affordable

Profiling put **85% of an episode in one line**: a grouped `ewm().mean()`
re-derived for every player, every week, every team, every episode. Those are the
same numbers every time, because what a player had averaged going into week 9 of
2021 is a fact about 2021, not about which seed shuffled the draft. Computed once
per season into a table, an episode drops **3.5s → 0.67s**.

The table is a shortcut around the environment's one hard rule, so it is proved
rather than asserted: every average is checked against reading the truncated
history frame, and across 14,130 player-weeks the largest disagreement is
**exactly zero**. A whole episode run both ways is bit-identical.

Two bugs on the way, both silent:

- **A player on bye has no row**, so the first version left him without an
  average — and the roster mechanic, which decides who to cut on exactly these
  numbers, cut players for being idle.
- Filling that gap exposed the ordering: **the running statistic has to be filled
  across the gap and lagged afterwards**. Lagging first propagates "everything
  before week 6" into week 7, when the truth for week 7 is "everything up to
  week 6". It misprices players only in the weeks after a gap, and only downward.

## What the agent is, and why it is small

The environment asks for one number per rostered player and does the constrained
assignment itself, so a learned policy is a **ranking function**: a linear score
over fifteen per-player features, plus one threshold for how big a gap has to be
before making a claim. Sixteen parameters.

Small deliberately. The whole prize is 2.33 wins against ±3 wins of seed noise,
and that ratio decides how many parameters the measurement can support. A network
would fit the draft order.

The waiver decision **shares the weights** — "is this free agent worth more than
my worst spare" is the same judgement the lineup makes — which is also what stops
the agent claiming players it would never start.

## Why cross-entropy search rather than a gradient

Three properties of the problem, none of them a preference:

- **The lineup is an exact argmax.** No useful gradient runs through it. A method
  that only needs the *return* of a parameter vector never touches it.
- **The reward is buried in noise.** Anything reading a single episode as
  evidence will chase seeds.
- **There are sixteen parameters.** Population search covers that fine, and the
  sample efficiency a gradient buys is not worth the machinery.

Three variance reductions matter more than the optimiser does: **common random
numbers** (every candidate in a generation plays the same seats), **a control
variate** (fitness is the return *minus the standard opponent's return in the
same seat*), and **a fresh batch each generation** so a candidate cannot survive
on suiting one sample.

The search starts from a heuristic — weight 1.0 on a two-game average, which
lands within noise of the field — rather than from zeros. That is what makes the
headline readable as *what the search added*, and the control below is what
verifies it.

### The waiver credit is not in the reward, on purpose

The episodic return already contains every point a claim produced, because a
claimed player who starts scores into the lineup. Adding the marginal credit on
top would count the same points twice and reward churn exactly as gross credit
would. The credit is a **credit-assignment** device — it answers "which decision
earned this" for a method that needs to attribute a return to individual actions,
and a method scored on the whole season does not need that attribution. It is
reported as a diagnostic instead.

That diagnostic earns its keep here: the control policy below banks **+195**
marginal credit a season and wins **nothing**. Banking credit is not winning.

---

## The result

90 seats (3 holdout seasons × 30 seeds), everything paired:

| policy | wins | points | rank | claims/season |
|---|---|---|---|---|
| oracle (perfect start/sit) | 9.29 | 1701 | 2.54 | — |
| **learned** | **7.54** | **1511** | **5.98** | 13.9 |
| field (the standard opponent) | 6.96 | 1480 | 6.39 | — |
| ewma2 (same machinery, one weight) | 6.92 | 1480 | 6.83 | 13.0 |

| paired comparison | wins | SE | t |
|---|---|---|---|
| learned − field | **+0.589** | 0.164 | **+3.59** |
| learned − ewma2 | **+0.622** | 0.137 | **+4.53** |
| ewma2 − field | −0.033 | 0.179 | −0.19 |
| oracle − field | +2.333 | 0.150 | +15.57 |

**The control is the important row.** `ewma2` is the agent's own
parameterisation with a single weight on a two-game average — same features
available, same waiver machinery, same threshold — and it lands *exactly* on the
field, at −0.03 wins. So the feature set and the add/drop plumbing are worth
nothing by themselves, and the entire +0.59 is attributable to the learned
parameters rather than to anything that came with the scaffolding.

### There is a real generalisation gap

| | wins vs. field | % of oracle band |
|---|---|---|
| train seasons (2016–2022, searched over) | +1.09 (t = 8.56) | 41% |
| holdout seasons (2023–2025) | +0.51 (t = 3.05) | 22% |

Roughly half the training-season edge does not survive. Sixteen parameters over
seven seasons still finds something season-specific, and the honest headline is
the holdout number. Per season it is +0.76 (2023), +0.12 (2024), +0.64 (2025) —
positive in all three, but 2024 is within noise on its own, which is what ±3 wins
a season does to a 0.5-win effect.

## Where the edge actually comes from

Zeroing groups of learned weights, same 90 seats:

| variant | wins vs. field | t |
|---|---|---|
| learned (full) | +0.589 | +3.59 |
| − `played_rate` | +0.500 | +3.26 |
| − position intercepts | +0.122 | +0.64 |
| − `adp_value` | **−0.233** | −1.43 |
| − both of those | −0.289 | −1.61 |
| only `ewma2` + `adp_value` + positions | +0.567 | +3.66 |
| never claims (lineups only) | +0.233 | +1.85 |

Three things fall out:

**The draft board keeps its value all season, and the field throws it away.** The
learned weight on `adp_value` is +0.40, and removing it costs 0.82 wins — more
than the agent's entire edge, taking it below the field. The standard opponent
consults the board for three weeks and then never again; that is its single
largest mistake. A preseason consensus is a season's worth of information that
fourteen weeks of scoring does not replace.

**Position intercepts are the second-largest term**, worth 0.47 wins. They do
nothing within a position — a constant cannot reorder a ranking — so their entire
effect is on the two decisions that compare *across* positions: who fills the
flex, and which free agent is worth which bench player. The learned values prefer
tight ends at the flex (+0.34) over backs (+0.05) and receivers (−0.00).

**Eleven of the fifteen features are doing nothing.** `ewma2`, `adp_value` and
the position intercepts alone reproduce +0.567 of the full +0.589. The extra
half-lives, season-to-date mean, last week's points and experience add ~0.02
wins, which at SE 0.16 is indistinguishable from nothing.

### Most of the edge is on the waiver wire

Holding the lineup policy but never claiming gives **+0.23 wins**; the full agent
gives +0.59. So waivers are **+0.36 of the +0.59** — roughly 60%.

Set against the ceilings measured earlier, the agent captures:

- **47% of the 0.77-win waiver band** (0.36 of 0.77)
- **10% of the 2.33-win start/sit band** (0.23 of 2.33)

That asymmetry is the most useful thing the run produced. Working the wire is the
easier problem — it mostly needs noticing that a free agent is outscoring a bench
player, which a linear score does well. Start/sit is where the projection quality
actually binds, and a linear function of six-odd summary statistics captures
almost none of it. **That is where the shipped weekly model should be pointed
next**, and the environment is already set up to take it: `ProjectionPolicy`
accepts a walk-forward projection frame and drops into the same seat.

---

## What this does not show

- **The draft is untouched.** Every result holds the draft fixed at the ADP snake
  and varies only what happens afterwards. The oracle still loses 4.7 games a
  season with a perfect card, and most of that is the roster it was handed.
- **The agent does not control its own housekeeping.** IR placement and forced
  replacements use the shared EWMA valuation, not the agent's weights, because
  the action space carries a score per *rostered* player and nothing about
  players it does not hold.
- **One training run, one seed.** The search converged by generation ~17 (σ
  collapsed from 0.32 to 0.02) and the reported parameters are from a single run
  with seed 0. The holdout evaluation is independent of the search, so the +0.59
  is not a selected-best number, but run-to-run spread in the *learned parameters*
  is unmeasured.

---

# The weekly model as an input (2026-09, second pass)

Everything above was measured with an agent that had never seen the weekly model.
Its features were exponentially weighted scoring and the draft board -- roughly
the naive baseline the weekly model beats by 11.6% CRPS -- and the ablation said
exactly where that would show: 47% of the available waiver band captured, but
only 10% of the start/sit band, and start/sit is where projection quality binds.

So the projections went in. Two horizons, because the agent makes two different
decisions:

- **Next week**, from the shipped hurdle, for the lineup.
- **Rest of season**, from the direct total with phase and ADP, for the waiver
  wire. "Should I add this player" is not a question about Sunday; it is about
  what he is worth over the remaining weeks, and the agent previously had no way
  to ask it.

Both are walk-forward: each holdout season is predicted by a model fitted on
strictly earlier seasons. That costs the first two seasons, so projections start
in 2018 and the training window is 2018-2022 with 2023-2025 held out.

## It roughly doubles the agent

Both arms trained identically -- same seasons, same seeds, same generations --
with the projection weights held at zero inside the search for the control, so
the two differ by the feature alone.

| holdout, 75 paired seats | wins vs. field | % of the 2.31-win band |
|---|---|---|
| without projections | +0.60 (t = 3.33) | 26% |
| **with projections** | **+1.13 (t = 6.11)** | **49%** |

Paired directly, the projections are worth **+0.53 wins and +53 points a season**
(t = 3.26 and 8.27). The train-season gap is +0.50, so unlike most of what has
been measured here this one does not shrink out of sample.

The learned weights say the agent uses both horizons — `projection` +0.50 and
`ros_projection` +0.38, third and sixth largest of eighteen. And `adp_value`
collapses from +0.40 in the old agent to +0.11, which is the right thing to
happen: the draft board's information is already inside the projection, so the
agent stops reading it twice.

## What changed underneath, and why the old numbers are not comparable

The environment is not the one the first agent was measured in. Since then
transactions have no weekly cap, a dropped player sits on waivers for 48 hours,
a week has two transaction phases rather than one, and lineups lock per kickoff
rather than all at once. The field moved with it: the standard opponent now takes
7.01 wins on the holdout rather than 6.96, and the oracle band is 2.31 rather
than 2.33.

So the honest comparison is the one above -- both arms retrained and re-measured
in the current environment on the same seats -- not +1.13 against the old +0.59.

## What is still open

- **Blending the rest-of-season projection with the rank curve.** The shipped
  configuration blends the direct total with the ADP rank curve at a per-horizon
  weight; this uses the unblended total. The blend is documented as the better
  model, so this is leaving something on the table.
- **The draft.** Still fixed at the ADP snake. The oracle loses about 4.2 games a
  season with a perfect card, and most of that is the roster it was handed.
- **The agent still does not control its own housekeeping.** IR placement and
  forced replacements use the shared EWMA valuation, not the agent's weights,
  because the action space carries a score per *rostered* player and nothing
  about players it does not hold. Now that the agent has a rest-of-season
  projection for everybody, that limitation is worth removing.
