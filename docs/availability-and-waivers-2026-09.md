# Competent opponents, and what a waiver claim is worth

*September 2026. Code: `src/ffmodel/league/{availability,roster,credit}.py`,
`scripts/validate_league.py`, `scripts/validate_waivers.py`. Numbers:
`reports/league_baselines.json`, `reports/league_waivers.json`.*

The league environment measured policies against eleven opponents who started
players on bye. That is not a hard environment, it is a strawman, and every
number it produced flattered whoever was being measured. This closes it, adds the
roster rules that go with it, and builds the reward an add/drop agent will train
on.

---

## 1. The opponents were leaving 10% of their lineup on the floor

Counting the agent's own started slots across a 2024 season under the standard
opponent strategy:

| started slots | before |
|---|---|
| on bye (club did not play) | 6.3% |
| rostered but inactive | 4.0% |
| **scored a guaranteed zero** | **10.3%** |

Byes were the larger half, and byes are the part nobody gets wrong: the schedule
is published in August.

### What counts as knowable

The fix is not "model who will be inactive" — that would hand every policy
information no manager has. It is to state the two things a manager genuinely
holds before the lineup locks, and nothing else.

**Byes** are inferred from the panel rather than fetched: a club contributes rows
every week it plays, so the one week it contributes none is its bye. Exact for
all thirty-two clubs in every season tested. It refuses to guess when a club is
absent for more than one week, because a bye and a gap in the panel are
indistinguishable and picking wrong would bench a player who played, invisibly,
every season.

**Out designations** come from the club's own game-status report, which
`ffmodel.weekly.news` measured as landing a median 28 hours before kickoff. It is
precise where it fires: of the rows flagged Out across 2023–2025, **zero**
recorded a stat line.

Everything else stays the manager's risk, and that boundary is load-bearing.
Among draftable players, only **25%** of scoreless weeks were ever flagged Out.
The rest are healthy scratches, first-quarter hamstrings, and starters who drew
no targets — none of it knowable on Wednesday.

### The roster rules that follow

Benching the absent is most of the value on its own: ordinary roster depth covers
the great majority of absences with no transaction at all. The mechanic handles
the rest:

1. **Bench whoever cannot play**; the assignment seats the next man up.
2. **Activate anyone whose injury has cleared**, cutting the lowest-valued bench
   player to make room. That cut is what makes IR not free.
3. **Cover a hole that is left.** If a starting slot has nobody to fill it, park
   the injured player on IR and claim the best free agent who can play. A bye
   never sends anyone to IR — it is one week and resolves itself, and real
   leagues would not allow it.

Result: **empty starting slots fall to zero per season**, byes started fall to
under one, and the field gains about 98 points a season. Teams make ~10 roster
moves a season, which is in the range a real manager makes.

### Three things this got wrong first, and they are worth knowing

**An empty slot is invisible if you look for it the obvious way.** Scoring absent
players at the bottom and checking for unfilled slots finds nothing: a team whose
only kicker is on bye still *fills* its kicker slot, because the assignment will
seat him rather than leave the slot empty. The hole only appears when the lineup
is built from the players who can actually play.

**Waiver order has to be reverse standings.** Twelve teams share one free-agent
pool, so whoever runs first gets the best replacement. Running them in team order
handed seat 0 — the agent's — the top of the wire every week of every season,
which is an edge roughly the size of the thing being measured.

**Roster decisions cannot use the lineup policy's valuation.** The standard
opponent ranks by the draft board for three weeks, and the board has nothing to
say about a player nobody drafted. So the housekeeping cut an undrafted
thirty-point-a-week breakout the week after it was claimed, because the board
ranked him below every rostered player. Recent production is the only scale on
which a rostered player and a waiver pickup are comparable, so roster decisions
use it throughout while lineup decisions stay the policy's own.

---

## 2. Averaging only the weeks a player was active makes things worse

The argument for it is good: absence is now stated by the environment and handled
by the roster rules, so counting it *again* inside the average is
double-counting, and that is what makes a returning starter look unstartable in
the week he comes back.

The argument is right about which weeks and wrong about how many. Three readings
of the same history, paired on seed over 3 seasons × 20 seeds:

| history counted | vs. `all`, wins | vs. `all`, points |
|---|---|---|
| `all` — every week on an NFL roster | — | — |
| `available` — every week except ruled-out and bye | −0.02 (t = −0.44) | −0.4 (t = −0.37) |
| `active` — only weeks he recorded a stat line | +0.05 (t = −0.54) | **−15.9 (t = −4.20)** |

**`active` costs 16 points a season**, decisively on points and invisibly on wins
— a 1.1-point-per-week change is far below the noise in a head-to-head margin,
which is exactly why points is the metric to read here. The reason is the 75%:
`active` drops every scoreless week, but only a quarter of those were knowable
absences. The rest is real risk, and dropping it measures "what is he worth when
he produces" — a different and more flattering question than a lineup decision
asks.

**`available` is the version of the idea that survives**, removing exactly the
weeks the environment already accounts for. It is also worth nothing (t = 0.37),
and for a satisfying reason: those players are benched anyway, so the weeks it
excludes barely enter the decision. The double-counting was real and it was tiny.

The default is `all`. All three are one keyword apart on `EwmaPolicy`.

---

## 3. Grading a waiver claim

A lineup decision grades itself — the points land the same week. A claim does
not: the receiver picked up in week 6 pays in weeks 9, 11 and 13, and the back
dropped to make room pays somebody else. So a claim is graded against the rest of
the season in **lineup points**, because a player who is added and never started
is worth nothing no matter what he scored.

Two formulations:

- **`marginal`** replays every remaining week twice — as it happened, and in the
  world where the swap was never made — and credits the difference.
- **`gross`** tallies the added player's points in the weeks he started and
  subtracts the dropped player's in the weeks he would have.

On real data, a good claim graded **+14.6 marginal and +38.8 gross**: gross
over-credits by 2.6× because it pays the full score of a player who merely
displaced a slightly worse one. An agent trained on it learns to add anybody
startable, constantly. `marginal` is the default.

The two also disagree in the other direction, and instructively. Cutting the
better of a roster's two quarterbacks — held as a regression test, where the
scenario can be built on demand — grades **−89.9 gross and −30.3 marginal**. The
backup fills the one quarterback slot, so most of the score gross counts as lost
was never lost: what the lineup actually gave up is the gap between the two, not
the starter's whole output. **Dropping your highest scorer is not the disaster it
looks like when the position is covered**, and only one of these two formulations
knows that.

### Does the reward behave?

Four waiver policies in the same seats, 3 seasons × 10 seeds, paired against
standing pat:

| waiver policy | wins | vs. stand-pat | marginal credit |
|---|---|---|---|
| oracle (claims whoever will really score) | 7.80 | **+0.77** (t = 3.16) | +67 |
| best available by recent form | 7.37 | +0.33 (t = 1.22) | +105 |
| random from the shortlist | 7.13 | +0.10 (t = 0.31) | +67 |
| stand pat | 7.03 | — | 0 |

The ordering is right and **waivers are worth up to 0.77 wins** on top of the
lineup decision. Marginal credit correlates positively with both wins (r = +0.26)
and points (r = +0.25).

**One limitation, visible in that table.** The oracle wins most but banks less
marginal credit than the merely-good policy. That is not a bug, it is what
marginal attribution does: each claim is graded against a world where only *it*
was reversed, so when a policy claims a star every week, each individual claim
looks replaceable — there is always another star. The per-claim number is the
right per-decision learning signal; **its season total is not a policy ranking**,
and should be read as a diagnostic rather than as waiver profit.

### One counterfactual bug worth recording

The first version graded the oracle waiver policy **negative**. The cause: when a
later claim had already cut the player an earlier claim brought in, the alternate
roster gained a player it never gave back, and a larger roster fields a better
lineup every remaining week. Every claim therefore looked worse the longer the
season ran. The counterfactual now keeps both rosters the same size, and a swap
that a later move has undone stops accruing credit rather than accruing it
against a roster it no longer caused.

**None of this reaches an observation.** The credit is computed from weeks the
agent had not played when it made the claim; it is available only after the
episode ends, and `grade_claims` raises if called during one.

---

## 4. Where this leaves the headroom

| | naive opponents | competent opponents |
|---|---|---|
| field's points per season | 1369 | 1467 |
| oracle start/sit, vs. field | +2.95 wins | **+2.58 wins** |
| oracle waivers, vs. standing pat | not measured | **+0.77 wins** |

Closing the strawman cost the oracle 13% of its band: some of what a perfect
start/sit used to be worth was just not starting players on bye, and the
opponents do that for themselves now. What is left is genuine.

So a lineup-and-waiver agent is playing for **at most ~3.3 wins of 14**, against
a seed noise of ±3 wins a season. Every comparison has to be paired on shared
seeds; nothing else is measurable at this scale.

---

## 5. The waiver wire (added 2026-09)

The first version of this kept one list of free agents, let anybody take anybody,
and capped adds at one a week. Both halves were wrong.

**Leagues cap nothing.** A manager may churn the back of a roster as often as he
is willing to cut somebody; the roster size is the only budget. The cap is gone.

**A cut player does not land in a pool available to the fastest click.** He sits
on waivers for 48 hours. Time is now tracked in hours rather than weeks, because
two days is not a number of weeks and the mechanic is invisible at weekly
granularity. A week has two transaction phases -- Wednesday and Friday -- and the
48-hour period is exactly what separates them:

| cut on | clears | available before that week's games? |
|---|---|---|
| Wednesday | Friday | yes |
| Friday | Sunday, after kickoff | no, not until the following Wednesday |

**A player dropped within 24 hours of being added skips waivers.** Without it a
manager could quarantine anybody by adding and immediately cutting him -- and
this is not a corner case, because the automatic housekeeping did exactly that
every time it claimed a replacement and then needed the spot back in the same
breath.

### What it fixed

The period does the job it exists for. Housekeeping cutting a player and
re-claiming him inside the same week falls from **5.0 times a season to zero**,
while the total number of roster moves is unchanged at about ten. That churn was
visible in the transaction logs before and had no defence; it is now impossible
rather than merely discouraged.

One bug worth recording: seeding the wire with the draft at hour zero made every
cut in week one look like a 24-hour undo, so a manager could have run his whole
bench through free agency in the opening week without anybody ever hitting
waivers. The draft happens days before the first transaction, and saying so is
what makes week one behave like every other week.

### What was not built

Claims on a player who is *currently* on waivers are not queued and awarded by
priority. The specification asked for a period during which a dropped player is
unavailable, and that is what this is: a lockout. A priority auction over
waivered players is the natural next step and would change who wins contested
pickups, but nothing here needed it, so it is not pretending to exist.
