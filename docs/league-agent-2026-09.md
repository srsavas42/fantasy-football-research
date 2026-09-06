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

---

# Should waivers be their own model? (2026-09)

Two questions, and they have different answers.

## Where the rest-of-season projection is doing its work

Ablating the trained agent, 75 holdout seats:

| variant | wins vs. field | what it costs |
|---|---|---|
| learned (full) | +1.133 | — |
| − `ros_projection` | +0.267 | **0.87** |
| − `projection` | +0.267 | 0.87 |
| − both projections | −0.493 | 1.63 |
| lineups only (no claims) | +0.293 | 0.84 |
| ... − `ros_projection` | +0.147 | **0.15** |
| ... − `projection` | −0.040 | 0.33 |
| claims only (flat lineup) | −1.240 | — |

**Rest-of-season is worth 0.87 wins overall and 0.15 to lineups**, so roughly
**83% of its value is in the add/drop decision**. That is what it was added for
and it is doing exactly that job. The next-week projection splits more evenly:
0.33 of its 0.87 is lineup work, the rest waivers -- a claim is partly a bet on
this week too, since a player claimed on Wednesday usually starts on Sunday.

Waivers are now **74% of the agent's whole edge** (+0.84 of +1.13), up from 60%
before the projections existed. And a flat lineup cannot be rescued by good
claims: ranking every player identically and claiming well still finishes 1.24
wins *below* the field.

**So the rest-of-season blend belongs in this model.** It is the input the
decision carrying three quarters of the agent leans on hardest, and the shipped
configuration blends the direct total with the ADP rank curve while this uses the
unblended total. That is the clearest remaining upgrade to the projection layer.

## Whether the waiver decision wants its own weights: no

Three arms, trained identically -- same seasons, seeds, generations, warm start --
differing only in how many weights the add/drop decision is allowed to hold apart
from the lineup. Waiver weights are carried as *deltas* on the shared ones, so
zero recovers the shared model exactly and the search only has to find a reason
to depart.

| arm | parameters | train | holdout | train − holdout |
|---|---|---|---|---|
| shared | 18 | +1.24 | **+1.13** | +0.11 |
| horizon split (both projections) | 20 | +1.09 | **+1.15** | −0.06 |
| full separate waiver head | 35 | **+1.55** | **+1.04** | **+0.51** |

Paired on seed against the shared model, neither split is distinguishable from
it: horizon split +0.013 wins (t = 0.08), full split −0.093 wins (t = −0.67).

The full split is the instructive row. It is **the best arm on the seasons it
searched over and the worst on the ones it did not** — +1.55 training, +1.04
holdout, a generalisation gap of 0.51 against 0.11 for the shared model. Seventeen
extra parameters bought training performance that did not survive contact with a
new season. That is what overfitting looks like when the noise floor is ±3 wins a
season, and it is the reason the parameter budget was set at "small" in the first
place.

**One model, shared weights.** The two decisions do rank players differently --
the horizon-split arm learned real deltas, +0.31 on the next-week projection and
+0.17 on rest-of-season, meaning the waiver decision wants to lean on projections
harder than the lineup does — but the difference is not worth a parameter. It is
also worth keeping the property that sharing guarantees: an agent that ranks
add/drop on the same function it sets lineups with cannot claim a player it would
never start, and nothing in these results is worth giving that up for.

The honest caveat: this says a *linear* waiver head is not worth separating. A
waiver decision that used information the lineup has no use for -- roster
construction, positional scarcity, what the other eleven teams are short of --
would be a different model rather than a different weighting of the same
features, and this experiment says nothing about that.

---

# The rest-of-season blend, and a rolling waiver queue (2026-09)

## The blend, and the bug in applying it everywhere

The shipped rest-of-season model is the direct total blended with the draft-board
rank curve at a variance-optimal weight, estimated per horizon by holding out the
most recent training season. Measured here: **0.32 early, 0.72 mid, 0.73 late** --
the board earning its keep in September and giving it back as the season supplies
usage it never saw.

Applying it to every player was wrong, and expensively so. The curve is a
statement about a draft board and its weight is fitted on drafted players; an
unranked player is placed at *the deepest rank the curve ever saw*, which is a
fallback rather than a forecast. Blending that in at the early-season weight
replaces two thirds of an undrafted player's projection with a replacement-level
constant.

| population | MAE vs. unblended | RMSE |
|---|---|---|
| all skill rows | **+12.9%** | +5.0% |
| drafted (the fitted population) | −1.9% | −2.9% |
| early weeks | **+24.4%** | +8.0% |

And undrafted players are precisely the waiver wire, which is the decision the
rest-of-season projection exists to serve. Restricted to drafted players it does
what it should:

| population | MAE | RMSE |
|---|---|---|
| all skill rows | **−1.13%** | **−2.05%** |
| drafted | −1.90% | −2.85% |
| undrafted | 0.00% | 0.00% |
| drafted, early | **−4.78%** | −4.97% |

Rank correlation with the actual remaining total rises 0.766 → 0.768, and the
gain sits where the theory puts it: early, when the model has no in-season data
and the board is all there is.

**What it is worth to the agent is not measurable.** Holding the trained weights
fixed and swapping only the projection cache, 75 seats: **+0.15 wins (t = 1.17)
and −21 points (t = −3.05)**. The projection is better and the agent is not
detectably better for it. That comparison is if anything tilted *toward* the
blend, since the weights were fitted on it. A 2% improvement in a feature the
agent reads at weight 0.44 is simply below what ±3 wins a season can resolve.

Worth keeping anyway -- it is the better projection, it costs nothing at
inference, and the ROS weight rose from +0.38 to +0.44 when the agent was
retrained on it -- but not worth claiming as a win.

## Waiver priority is now a rolling queue

It starts as the **inverse of the draft order** -- the team that picked last off
the board picks first off the wire -- and any team that adds a player moves to the
back, relative order preserved among those who moved and those who did not.

The point is that priority becomes a *resource*. Reverse standings recomputed
every week let a bad team hold first pick indefinitely and never pay for using
it; a rolling queue means spending priority on a marginal pickup sends the next
player worth having to somebody else.

### The agent was jumping its own queue

Its claims were applied on submission, before any other team transacted. That
handed it first pick of the wire every phase of every season -- the same
systematic edge the housekeeping order had, reintroduced through a different
door. Claims are now queued and resolve at the agent's turn, **where they can
fail** because a higher-priority team took the player first. A failed claim is no
longer recorded as having happened, so the waiver credit stops grading swaps the
agent never made.

Queued claims are validated against the roster the queue *will* produce rather
than the one standing now, since with no cap a policy can make several claims in
a phase and the second is naturally about what the first leaves behind.

## Where the agent stands now

90 seats, current environment, paired on seed:

| policy | wins | points | rank | title rate |
|---|---|---|---|---|
| oracle | 9.38 | 1702 | 2.57 | 56.7% |
| **learned** | **8.18** | **1547** | **4.31** | **20.0%** |
| field | 7.10 | 1484 | 6.12 | 12.2% |
| ewma2 (control) | 6.80 | 1467 | 6.76 | 4.4% |

| paired comparison | wins | SE | t |
|---|---|---|---|
| learned − field | **+1.078** | 0.162 | +6.65 |
| learned − ewma2 | +1.378 | 0.150 | +9.16 |
| ewma2 − field | −0.300 | 0.158 | −1.90 |
| oracle − field | +2.278 | 0.147 | +15.45 |

**+1.08 wins, 47% of the band**, down from +1.18 in the environment where the
agent jumped the waiver queue. That drop is the unfair edge being removed, and
the new number is the honest one.

The control moved too, and interestingly: `ewma2` now finishes *below* the field
(−0.30, t = −1.90) where before it sat exactly on it. Claiming naively is mildly
harmful once priority is a resource you can waste -- which is the mechanic doing
its job.
