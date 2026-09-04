# Target competition: what the allocator sees, and what it does not

Prompted by a reading of the 2026 projection: a rookie receiver drafted 20th
overall (Makai Lemon, PHI) projects 80.0 targets on a 0.237 snap share, next to
DeVonta Smith's 87.7 on 0.662 — and Smith's projected target share (0.177) is
*below* his own prior-season share (0.237) in a season where A.J. Brown left the
room. Two questions follow: does the allocator handle competition at all, and is
the rookie's number a defect?

## What the allocator does with competition

Mechanically, correctly. The target softmax renormalises over the current
roster, so a departure genuinely frees share and the survivors split it. The
score is

    log(role_prior) + log(snap_share) + X·beta + innovation

so projected playing time is already priced as an additive log-offset — a
rookie's low snap share is not being ignored.

Behaviourally, not at all. Nothing in `X` distinguishes "third of four wide
receivers" from "third of eleven skill players", because the softmax normalises
over the whole team and has no notion of a positional room. When a room loses
its alpha, the vacated share is redistributed in proportion to the survivors'
own priors; no term lets the best remaining option take more than its
proportional cut.

Four features that would say this are built and reach **no model**:
`prior_target_room_competition`, `prior_target_team_competition`,
`prior_rec_room_quality_advantage`, `prior_rush_room_quality_advantage`.

## Screening them

`scripts/screen_target_room_quality.py` scores each against the deterministic
prior allocation — role prior times observed exposure, renormalised over the
roster — which is what the softmax returns with no covariates and no noise.

The control block includes `log(prior_share)` as well as position, and it is not
optional. The residual is a log ratio, so a small prior has far more room above
than below and shows a positive residual from pure mean reversion; several
candidates are monotone functions of that same prior share
(`role_uncertainty` is literally `1 − room_share`). Uncontrolled,
`prior_target_role_uncertainty` reports **+0.431**. Controlled it reports
**+0.197**. Everything is also reported on the signed difference, because a
candidate that survives only the log metric is a floor artifact.

| feature | log ratio | signed difference |
|---|---:|---:|
| `prior_target_role_uncertainty` | **+0.197** *** | **+0.173** *** |
| `prior_rec_room_quality_advantage` | +0.035 | +0.062 ** |
| `prior_role_room_quality_advantage` | +0.028 | — |
| `prior_target_room_competition` | −0.025 | +0.009 |
| `prior_target_team_competition` | −0.037 * | +0.009 |

The two competition features are team- or room-level constants — every receiver
on a team shares the value — so they cannot separate the players the softmax is
allocating between, and they measure as nothing. They are left out.

Worth naming plainly: the intuition that prompted this work — *the best receiver
in a depleted room absorbs more than his proportional share* — is the **weak**
result here, at +0.062. The strong one is plain within-room standing, which is
structural rather than behavioural: the softmax normalises team-wide, so
room-level position is information it genuinely does not have.

`TARGET_ROOM_FEATURES` carries the two survivors behind
`room_structure_features`.

### The gate rejects it

Against the same frames on 2022/2023/2024, with every other stream identical to
the last digit — so the arm changed only what it meant to:

| metric | pooled | per fold | folds better |
|---|---:|---|---:|
| target MAE | **+4.63%** | +2.77 / +4.57 / +6.75 | 0/3 |
| target CRPS | **+2.37%** | +0.92 / +2.32 / +4.03 | 0/3 |

Coverage moves slightly away from nominal at both levels.

Why it fails is the useful part, and it is a caution about the screen rather
than about the idea. `prior_target_role_uncertainty` correlates **−0.605** with
`log(role_prior × exposure)` — the offset the softmax already carries as a
*fixed* term. Offering it as a free covariate lets the model partially re-weight
an input it was handed as known, which is exactly the double-counting the target
stream's `beta_scale` of 0.05 exists to prevent. A 4.6% MAE regression from a
feature shrunk that hard is the signature of collision with the offset, not of a
weak signal.

The screen could not have caught this. It residualises *against* the prior
allocation, so what it measures is what that allocation gets wrong — and a
feature can carry that signal honestly and still be unusable by a model whose
score is built on the allocation itself. Any future room-structure instrument
needs to be orthogonal to the offset, or to enter somewhere other than as a
covariate on it.

The flag and the code stay, off.

## The quarterback

`teammate_qb_quality_signal` reaches three receiving efficiency responses in
code — and **nothing in production**: `teammate_quality_features` defaults to
`False` and neither `project_season.py` nor `fit_production.py` sets it. So in
the shipped 2026 projection the quarterback reaches no response at all.

`scripts/screen_qb_context_rb.py` asks where he should. Partial correlations for
running backs, controlling for the player's own lagged rate:

| response | r | p | |
|---|---:|---:|---|
| carry share | −0.024 | 0.42 | nothing |
| target share | −0.003 | 0.94 | nothing |
| `rush_yards_per_carry` | +0.065 | 0.12 | nothing |
| **`rush_td_rate`** | **+0.171** | **3.8e−05** | |
| `rec_yards_per_target` | +0.149 | 1.4e−03 | promoted effect, as a positive control |

The quarterback does not move a back's workload, and does not move his yards per
carry. He moves the rate at which carries become touchdowns — a red-zone story,
not a ball-carrying one. `rush_td_rate` reads only `prior_rush_epa_per_carry`
and `prior_rush_first_down_rate`, so this is new information rather than a
re-transform. The signal is flat at +0.17 across three control sets including
the back's own `prior_rush_short_yardage_share`, so it is not his goal-line role
under another name.

`rush_td_rate` joins `TEAMMATE_QUALITY_TARGETS`. Yards per carry does not.

The same screen settles a question deferred earlier — whether rushing
quarterbacks check down to their backs less. A passer's prior-season rushing
load against his back's target share is −0.047 (p=0.18): right sign, not there.

## Is the rookie number a defect? No — the bias runs the other way

The suspicion was that `cold_role_innovation` inflates cold rows. It gives
players with no prior role a much wider innovation scale — promoted for interval
coverage, which it genuinely fixed — but the noise is added on the input side of
a softmax, and a softmax is not linear, so a wider scale raises a row's
draw-*average* share and not only its spread. Against the deterministic prior
allocation the observed cold-vs-warm lift is **2.65x**, and `exp(sigma^2/2)` at
the fitted cold scale predicts about 3x. The arithmetic matches the artifact.

The arithmetic is right and the conclusion drawn from it was wrong.
`scripts/measure_cold_start_bias.py` measures held-out *bias* — projected mean
against observed mean — split by cold-start status and by projected volume,
because most cold rows are players who never take a snap and their bias says
nothing about the rows a drafter reads. Targets and carries per team game,
2022/2023/2024:

| population | targets | carries |
|---|---:|---:|
| warm, top quartile by projection | **+13.0 / +6.4 / +11.2** | **+12.9 / +3.0 / +4.2** |
| all warm | +7.1 / +2.4 / +2.0 | +13.7 / +3.4 / +3.9 |
| all rows | +2.6 / −3.9 / +1.5 | +0.0 / +0.4 / +0.2 |
| cold, top quartile by projection | −8.7 / −13.1 / +3.2 | −55.8 / −11.8 / −25.7 |
| all cold | −25.6 / −32.4 / −1.9 | −60.5 / −20.8 / −25.0 |
| rookies | −20.9 / −29.4 / −6.9 | −62.5 / −31.0 / −37.8 |

Cold-start players are **under**-projected, on every fold and both streams. The
established players at the top of the board are **over**-projected, by 3 to 13%,
on every fold and both streams. The 2.65x lift is not inflating rookies past
where they belong; it is partially closing a gap that is still open.

So a rookie taking share from an established WR1 is the allocator moving in the
direction the holdouts say it should, and not far enough — which inverts the
reading that prompted this section. The row to distrust in a projection is not
the rookie: it is the established starter at the top, whose targets and carries
are systematically high.

## `mean_preserving_innovation`: a third rejection

The correction solves for a per-player offset restoring the draw-average to the
noiseless allocation, so it removes exactly the softmax mean shift and nothing
else. It is read only in `_role_share_prediction` and never during fit, so one
posterior serves both arms and predicting twice with the same seed makes the
comparison exactly paired.

It makes every cell above worse. Target cold bias goes from −25.6/−32.4/−1.9 to
−58.1/−61.7/−42.6; warm bias goes *up* rather than down. Pooled error against
the baseline:

| stream | population | MAE | CRPS | folds better |
|---|---|---:|---:|---:|
| target | all | +5.14% | +5.49% | 0/3 |
| target | cold | +12.46% | +16.14% | 0/3 |
| target | cold, top quartile | +24.79% | +21.10% | 0/3 |
| carry | all | +0.55% | +2.02% | 1/3 |
| carry | cold | −2.28% | +5.32% | 3/3 |
| carry | cold, top quartile | −3.83% | +3.40% | 2/3 |

`docs/role-innovation-2026-08.md` rejected this flag twice, on cost to the
passing streams and then on the carry bias it failed to explain. This is a third
rejection on a population neither earlier run broke out, and the clearest of the
three: the mean shift it removes is load-bearing. The correction is
mathematically right about what the softmax does and wrong about whether the
model wants it undone — the deterministic allocation it restores is itself
biased low on exactly these rows, so removing the noise-driven lift exposes the
underlying error rather than fixing anything.

The carry MAE column is the one place it helps, and it helps by −2 to −4% while
costing CRPS on the same rows. That is the same trade
`docs/role-innovation-2026-08.md` already declined at the carry layer, for the
same reason.

## What is still open

The under-projection of cold rows and the over-projection of established
starters is a real, replicated, two-sided bias and nothing here fixes it. It is
not an allocation-noise question — that is what this measurement ruled out — so
it belongs to the allocator's other inputs: the role prior, the snap projection,
or the innovation *scale* rather than its mean.

## Is the reallocation itself wrong? No — it is already proportional

A natural suspicion is that vacated targets get spread evenly instead of in
proportion to what each survivor already holds. They do not, and this is settled
by algebra before any measurement: the share is `exp(s_i) / sum_j exp(s_j)`, and
removing a player deletes one term from the denominator only. Every survivor's
numerator is untouched, so every survivor's share scales by the identical factor
`1 / (1 - p_departed)`.

Put A.J. Brown back on the 2026 Philadelphia roster and take him out again:

| player | with Brown | without | gain | multiplier |
|---|---:|---:|---:|---:|
| DeVonta Smith | 0.2071 | 0.2600 | **+0.0529** | 1.2555 |
| Dallas Goedert | 0.1419 | 0.1781 | +0.0363 | 1.2555 |
| Saquon Barkley | 0.0866 | 0.1088 | +0.0221 | 1.2555 |
| Dontayvion Wicks | 0.0863 | 0.1084 | +0.0221 | 1.2555 |
| Makai Lemon | 0.0799 | 0.1003 | +0.0204 | 1.2555 |

Brown's 0.2035 split evenly across 16 survivors would be +0.0127 each. Smith
gets +0.0529, four times that, because he holds four times the prior share. The
multiplier is identical to four decimal places for every player in the room.

So the departure *is* credited to Smith, and generously. His deterministic prior
allocation is **0.2600** — above the 0.2435 he actually ran in 2025.

Nor is it exposure. Holding every returning player at his 2025 snap share
instead of the projection moves Smith to 0.2571, slightly *down*, because the
projection lowers everyone's snaps together and the softmax only reads the
ratios.

### Where the number actually goes

| | Smith target share |
|---|---:|
| 2025 observed | 0.2435 |
| prior allocation, Brown removed | 0.2600 |
| same at his 2025 snap share | 0.2571 |
| **model output** | **0.1770** |

The whole 0.2600 → 0.1770 move happens inside the fitted model — in `X·beta` and
the innovation — not in the reallocation and not in the exposure.

Most of that is shrinkage the model is right to apply. The prior allocation is a
badly biased forecast at the top: regressing observed share on it in logs over
2015-2025 gives a slope of **0.708**, and by band,

| prior allocation | n | mean prior | mean observed | ratio |
|---|---:|---:|---:|---:|
| 0.02–0.05 | 776 | 0.0336 | 0.0445 | 1.323 |
| 0.10–0.15 | 501 | 0.1243 | 0.1172 | 0.943 |
| 0.20–0.25 | 229 | 0.2253 | 0.1916 | 0.851 |
| 0.25+ | 277 | 0.3053 | 0.2481 | 0.813 |

A room leader who profiles at 0.26 should not be projected at 0.26.

How much shrinkage: the model is close to right, once the benchmark is built
the way a forecast actually works.

The calibration table above uses each row's *realized* snap share as exposure.
That was the correct choice for screening a feature — routing it through a snap
projection would fold that model's error into the answer — but it makes the
table the wrong benchmark for a projection, because it gives the allocation
perfect foresight of playing time that no forecast has. Rebuilding it on
prior-season exposure, which is what a forecast holds:

| exposure used | n | mean observed share | ratio to prior |
|---|---:|---:|---:|
| realized (perfect foresight) | 226 | 0.2160 | 0.839 |
| **prior-season (what a forecast has)** | **203** | **0.1835** | **0.714** |

Against the second row — the honest comparison — the model's 0.1770 for Smith is
**3.5% low**, not the 18% the first row suggests. That is calibration, not a
defect, and an earlier version of this section claiming otherwise was comparing
against a benchmark the model could never reach.

### What "66% of snaps" means

`snap_share` is whole-season by construction: player offensive snaps summed over
the weeks he played, divided by team snaps summed over *every* team game. Missed
games depress it, so the column conflates per-game usage with availability.

The pipeline is consistent about this — the softmax exposure offset is the
whole-season share and season counts are `share × team season total`, so
availability enters exactly once and is not applied twice — but it makes the
number easy to misread. Smith:

| | |
|---|---:|
| 2025 snap share | 0.8424 (17 of 17 games, so a clean per-game rate) |
| 2026 projected snap share | 0.6618 |
| 2026 projected games | 14.44 |
| **implied per-game snap rate** | **0.7789** |

His projected usage *when he plays* is 78%, not 66%. The rest of the gap is the
availability projection, and that projection is well calibrated: established
starters who played every game go on to average 14.16 games, against the 14.44
the model gives him. Availability barely persists at all — the correlation
between a starter's prior and next-season availability is **+0.079** — so the
hard regression toward the league mean of 13.85 games is the right behaviour,
not an insult to a player who has been durable.

## The late-season role weight: walk-forwarded, rejected

Separately from all of the above, the user asked whether the role priors should
weight recent (second-half) performance more heavily for younger players, since
rookies often break out late. `prior_target_role` and its siblings already blend
`0.65 * full_season_share + 0.35 * late_season_share` (late meaning weeks 10+),
a single constant applied to every player regardless of experience.

`scripts/screen_late_season_weight.py` solved in closed form for the weight
minimising squared error against next season's share and found the shipped 0.35
far too high everywhere — pooled optimum 0.035–0.095 depending on specification,
falling with experience in every arm (roughly 0.14–0.18 entering year two,
near zero from year five on). `scripts/screen_recency_weighting.py` then built a
smooth exponentially-weighted alternative from the weekly panel and found R²
rising monotonically with half-life in every experience bucket, on both target
and carry share — the flat full-season share outperforms every finite half-life
tried, including at every age.

Both screens pointed the same direction, so it was walk-forwarded:
`scripts/validate_late_season_weight.py` recomputes only the three role columns
(`prior_target_role`, `prior_carry_role`, `prior_pass_role`) on the cached
2022/2023/2024 frames and refits the full volume pipeline, so it is a controlled
comparison against `wf_roombase.json`, the shipped 0.35 baseline on identical
frames.

**A flat 0.10 weight for everyone — the pooled screen's own answer — moves
nothing material and is wrong-signed on target:**

| stream | metric | pooled | per fold | folds better |
|---|---|---:|---|---:|
| target | MAE | +0.20% | +0.11% / +0.47% / +0.01% | 0/3 |
| target | CRPS | +0.22% | +0.12% / +0.31% / +0.21% | 0/3 |
| carry | MAE | +0.05% | +0.06% / −0.09% / +0.17% | 1/3 |
| carry | CRPS | +0.05% | +0.02% / +0.01% / +0.11% | 0/3 |

`pass_qb` and `snap` are identical to the last digit on every fold, confirming
the change reached only what it was meant to. Every cell sits under the gate's
0.25% materiality floor, and target is wrong-signed in all three folds rather
than merely inconclusive.

The reason the closed-form optimum barely registers in the fitted model:
`prior_target_role` enters `_role_prior`'s geometric blend at a weight of only
0.25 for targets — `prior_target_per_snap` carries the other 0.75 — so a large
move in this one column is a small move in the softmax score. Both screens above
were run against raw shares, which is the right question for the feature in
isolation, but the fitted model has already routed most of its role signal
through the per-snap rate instead. If a recency effect is real, it more likely
belongs there than in the late-season blend weight.

An age-varying arm (0.20 for experience ≤3, 0.02 otherwise) confirms it:

| stream | metric | pooled | per fold | folds better |
|---|---|---:|---|---:|
| target | MAE | +0.10% | −0.10% / +0.43% / −0.04% | 2/3 |
| target | CRPS | +0.14% | −0.04% / +0.33% / +0.14% | 1/3 |
| carry | MAE | +0.08% | +0.04% / +0.01% / +0.19% | 0/3 |
| carry | CRPS | +0.07% | +0.04% / +0.03% / +0.16% | 0/3 |

Every cell is under the materiality floor and none wins every fold — the
promotion gate needs both. Splitting the weight by age doesn't do what the
raw-share screens suggested it should, for the same reason the flat 0.10 arm
didn't: the blend column this weight controls is a minor input to the softmax
score next to the per-snap rate. Both arms of `prior_target_role`'s
late-season weight are closed out: shipped 0.35 stands.

## The coaching tree, finally testable

`scripts/screen_zone_teammate_coach.py` had to leave the coaching-tree question
open: "it needs the scheme-lineage tables, whose scraper cannot run here because
the environment's network policy denies the Wikipedia host." Those tables now
exist (scraped on a GitHub Actions runner, `data/coaching/wikipedia/`), so this
closes it.

Two framings were tested, and they are not the same question.

### Role churn: null

`scripts/screen_coaching_role_churn.py` asks whether a change of scheme carrier
*widens* role dispersion — whether last season's role describes this season's
role less well after the offense changes hands. That framing is attractive
because the volume model already has the machinery to use it: `cold_role_innovation`
already says "rows with no prior role deserve a wider innovation," so "rows whose
offense just changed hands" would be the same claim about the same parameter,
rather than a covariate fighting the softmax offset (which is how room structure
failed, above).

It is null. Level effects are absent everywhere (|r| < 0.02, p > 0.45 for every
flag on every stream). Dispersion shows one marginal hit — target share against
a new offensive coordinator, |resid| 0.355 vs 0.331, r = +0.037 at p = 0.043 —
but that is one hit in nine tests, and carry share moves the *opposite* way
(−0.027). Snaps show nothing at any flag. Read as noise.

`has_midseason_change` is deliberately excluded from that screen: a coach fired
in week 8 is not knowable in August, so a feature reading it would score well in
validation and be unavailable when serving.

### Shape transfer: one real survivor

`scripts/screen_coaching_tree_transfer.py` asks the sharper question — does an
arriving play-caller bring his previous offense's *shape* with him? Three
team-season shapes, each about how volume is distributed rather than how much
there is, predicted from the scheme coach's own prior NFL stops:

| shape | raw r | partial r | p |
|---|---:|---:|---:|
| **`rb_target_share`** | +0.225 | **+0.204** | **0.027** |
| `target_hhi` | −0.070 | −0.085 | 0.36 |
| `rush_rate` | +0.118 | +0.086 | 0.36 |

Partial is beyond the team's own previous **three** seasons — a strong control,
since a team's distribution shape persists well past one year and a single-season
control leaves enough of that in the residual to flatter the coach term.

Target concentration and run/pass balance are fully absorbed by team persistence:
whatever a coach's reputation, his new team throws to one alpha or spreads it
around about as much as it already did. What transfers is **how much of the
target pie goes to running backs**.

Two checks decide whether that survivor is real, and it passes both:

**The mechanical confound.** A coach who *stayed* has his own prior seasons at
this same team inside his lineage, so his "carried" shape is partly the team's
own history under another name. Restricting to stops at *other* franchises makes
the effect **stronger**, +0.163 → +0.204 — the opposite of what a self-correlation
artifact does.

**Free-parameter sensitivity.** The recency half-life and the role filter are
both arbitrary, so both are swept. Half-lives from 2 years to flat give +0.189
to +0.210; role filters give +0.203 (OC only) to +0.231 (adding quarterbacks
coach, n=145, p=0.006). Nothing here is doing the work. That the recency
weighting is nearly irrelevant — flat is marginally *best* — is itself the
finding: a play-caller's back-usage is a stable career trait, not recent form.

### What it would take to use it

Not a plain covariate. `coach_rb_target_share` is constant within a team-season,
and the target softmax normalises within team-season, so a main effect cancels
exactly — the same reason `prior_target_room_competition` measured as nothing
above. To move anything it has to enter as an interaction with position
(`coach_rb_target_share × is_RB`), which shifts backs relative to receivers
inside the room rather than shifting the room.

That is a real, specific, leakage-safe candidate for the target allocator, and
it is the first coaching result in this repo that is about volume distribution
rather than a team effect wearing a hat. It has not been walk-forwarded. n=121
team-seasons at p=0.027 is thin, and team-level features are the family that has
failed every forecast test in this line of work — so the screen is a reason to
run the gate, not a reason to skip it.

### Depth of target does not travel with the play-caller

The obvious follow-up to "running-back share transfers" is depth of target:
play-callers visibly differ in how far downfield they throw, and a
short-throws-and-run-after offense looks like a scheme signature rather than a
roster accident. Five depth shapes were added to the same transfer screen:

| shape | raw r | partial r | p |
|---|---:|---:|---:|
| `team_adot` | +0.022 | +0.056 | 0.55 |
| `yac_share` | +0.071 | +0.117 | 0.21 |
| `wr1_adot` | −0.099 | −0.092 | 0.33 |
| `wr3_adot` | +0.117 | +0.116 | 0.25 |
| `adot_spread` (wr1 − wr3) | −0.005 | −0.006 | 0.95 |

Receivers are ranked by *observed* targets, not by the `depth_rank` column:
that one is a week-1 snapshot and correlates only 0.31 with realized target
order, so published depth charts describe the offense far worse than what the
offense actually did.

All null, but n=121 team-seasons resolves only |r| > 0.18, so that alone could
not distinguish "no effect" from "modest effect". `scripts/screen_coach_adot_player_level.py`
settles it with a much better-powered within-player design: the same receiver,
same team, back-to-back seasons with a real target load in both, whose scheme
carrier changed between them — controlling for his own prior aDOT and the
team's. Everything is held fixed except who is calling the plays.

| predictor | n | partial r | p |
|---|---:|---:|---:|
| incoming coach's carried `team_adot` | 172 | **−0.005** | 0.95 |
| incoming coach's carried `yac_share` | 172 | **+0.014** | 0.86 |

Flat zeros, not merely non-significant.

One reading fits both the hits and the misses. What transferred —
`rb_target_share` — is an allocation the play-caller makes unilaterally: he can
check down to whoever is in the backfield on any roster he inherits. What did
not transfer — depth of target, target concentration, run/pass balance — all
need personnel he has to be given. A coach cannot install a vertical passing
game without an arm and a burner, and Miami's short-and-YAC offense is not
separable from having signed Hill and Waddle: the scheme and the personnel
arrived together.

That principle is worth carrying into any future coaching feature: screen the
decisions a coach owns outright, not the ones his roster has to underwrite.

The caveat that keeps this from being final is measurement. All of the above is
*realized* aDOT, which mixes play design with execution and with who was open.
A designed-depth measure — route depth off tracking data — could plausibly
transfer where realized depth does not. This repo has no such column.

### Four more shapes, four more nulls — and a correction

The transfer principle stated above ("what transfers is the allocation the
play-caller makes unilaterally") was drawn from a single hit and is too broad.
Four more shapes, same design, same controls:

| shape | raw r | partial r | p |
|---|---:|---:|---:|
| `te_target_share` | +0.113 | +0.093 | 0.27 |
| `wr_target_share` | +0.131 | +0.113 | 0.18 |
| `rookie_target_share` | −0.038 | −0.055 | 0.52 |
| `rookie_carry_share` | −0.020 | −0.019 | 0.82 |

Tight-end and receiver target share are *equally* unilateral allocations and
neither transfers. So the principle needs narrowing: what travels is not
positional allocation in general, it is specifically **the backfield's
involvement in the passing game**, which is a property of progression and
protection design rather than of who is on the roster. Even tight-end usage —
which feels like the most schematic thing a coach does — is dominated by
whether the team happens to have a good tight end.

`rookie_target_share` was tested because the cold-start population is this
model's largest documented bias (under-projected on every fold, both streams,
above), and "some staffs trust rookies earlier" would have aimed straight at
it. It does not transfer either.

### The base rate this result has to be read against

Counting everything now screened — coach identity on three efficiency
responses, role churn on three streams across three flags, and eight team-season
shapes across two designs — roughly a dozen coaching hypotheses have been put to
the data and **one** survived, at p=0.027. Under the null, a dozen tests would
be expected to throw about one hit under p=0.05 by chance.

That is not a reason to discard `rb_target_share`: it held its sign and
magnitude across a half-life sweep, a role-filter sweep, and the removal of the
mechanical same-team confound, which noise does not usually do. But it is the
reason the walk-forward is the arbiter and the screen is not, and it is the
reason no further shape-hunting is planned. The marginal untested shape now has
a poor prior.

### The deep-history correction: the effect is real, and I was measuring it badly

The feature above was built against team shapes from the walk-forward frames,
which start in 2015. That silently discarded most of the evidence: of 1,630
external play-calling stops behind 2016-2025 response seasons, **1,087 are
pre-2015** and had no shape to attach, median around 2009. Only 65.8% of
response team-seasons got any usable stop. The binding limit was never the
coaching data — it was how far back the *player* data reached.

nflverse player weeks run to 1999 and carry the three columns the shape needs,
so `scripts/screen_coaching_deep_history.py` rebuilds them that far back. On raw
shares the result was alarming:

| team shapes from | n | partial r | p |
|---|---:|---:|---:|
| 2015 | 140 | +0.249 | 0.003 |
| 2010 | 190 | +0.176 | 0.016 |
| 2005 | 194 | **−0.039** | 0.60 |
| 1999 | 197 | **−0.021** | 0.77 |

Adding stops makes each coach's carried value *less* noisy, and attenuation
runs the other way, so a real effect should have strengthened. Collapsing to
zero looked like the original had been noise.

It was an era artifact. League-wide running-back target share drifts hard —
0.230 in 1999 against 0.175 in 2024 — so averaging a 2005 stop with a 2020 stop
on the raw scale adds eras together, not tendencies. Z-scoring each stop within
its own season fixes it:

| team shapes from | n | partial r | p |
|---|---:|---:|---:|
| 2015 | 140 | +0.267 | 0.0016 |
| 2010 | 190 | +0.222 | 0.0023 |
| 2005 | 194 | +0.203 | 0.0049 |
| **1999** | **197** | **+0.211** | **0.0032** |

So the effect is considerably better established than first reported: n=197
rather than 121, p=0.003 rather than 0.027, and stable across every window
depth. Against the base rate noted above — a dozen hypotheses, one hit near
p=0.03 — a hit at p=0.003 on 63% more rows is a different proposition.

Recency weighting still barely matters (+0.223 at a 3-year half-life against
+0.216 at ten), and this time that means something: the earlier half-life sweep
ran on a window where every stop was within nine years and had almost no range
to resolve. With real range, flat still holds, so "stable career trait" survives
a test that could actually have refuted it.

Two consequences for the shipped feature, which used raw shares over 2015+ stops
only: it needs era-normalised inputs, and it needs the deep window. Both are
corrections to how the quantity is measured rather than new hypotheses.

*(Noted in passing: 2004 reports a league mean of 0.400 at sd 0.548 for a
bounded share, which is impossible and marks corrupt rows in that season.
Z-scoring is self-limiting against it and the result is stable across windows
that include and exclude it, so it is flagged rather than chased.)*

### Mentor pooling, and what it turned up instead

The coverage gap in the shipped feature — only some back rows get a carried
value, because the coach needs prior stops with a shape attached — looked like a
job for the `mentor_head_coach` edges: a first-time coordinator has no
play-calling record of his own, but he has a tree.

Widening the role filter is the direct test of that, since a stop in *any*
offensive role is exposure to someone else's offense:

| lineage roles admitted | n | partial r | p |
|---|---:|---:|---:|
| play-calling only (OC / HC) | 122 | +0.208 | 0.024 |
| + quarterbacks coach | 152 | +0.200 | 0.015 |
| + all offensive position coaches | 162 | +0.191 | 0.016 |
| any stop at all | 170 | +0.160 | 0.039 |

Coverage rises and the signal dilutes, which is the right shape for a real
effect: a coach's own play-calling record should inform more than the offense he
interned in. Power peaks in the middle, where n grows faster than r decays.

But the isolated case — coaches with *no* play-calling stop anywhere, the
population mentor pooling exists to serve — has **n=19**. Too few to validate,
and that number is itself the finding: nearly every NFL scheme carrier has prior
coordinator or head-coaching experience somewhere, so first-timers were never
what made coverage thin.

Chasing what actually made it thin produced the deep-history correction above,
which is worth more than mentor pooling would have been. Mentor edges stay
unused: they address a population of 19.

### Load management and the committee question: null

The second candidate. Availability is this package's least persistent layer
(starter prior-to-next r = +0.079), so a staff effect there would predict
something nothing else does; and "is this coach a committee guy" is probably the
most widely held coaching belief in fantasy football.

| shape | raw r | partial r | p |
|---|---:|---:|---:|
| `starter_availability` | −0.093 | −0.094 | 0.27 |
| `rb_carry_top_share` | +0.014 | **+0.003** | 0.97 |
| `rb_carry_hhi` | +0.022 | −0.010 | 0.91 |
| `top5_snap_rate` | +0.014 | +0.021 | 0.81 |

All null, three of them flat. Backfield concentration is the emphatic one: a
coach's committee-versus-bell-cow reputation does not follow him at all.

Read against the one hit, the pattern sharpens usefully. A play-caller carries
**how often the backfield is thrown to** — a progression-design choice — and not
**which back gets the work**, which is a personnel decision made with the roster
he is handed. That is the same line the depth-of-target nulls drew, and it is
now drawn from four directions.

### The gate rejects it

Both arms on identical rebuilt frames (fingerprint verified equal), 500 draws /
500 tune, holdouts 2022/2023/2024:

| stream | metric | pooled | per fold | folds better |
|---|---|---:|---|---:|
| target | MAE | **+0.056%** | +0.007 / +0.089 / +0.072 | 0/3 |
| target | CRPS | **+0.016%** | −0.008 / +0.038 / +0.019 | 1/3 |
| carry, pass_qb, snap, availability | both | identical on every fold | — | — |

The gate wants material (>0.25%) *and* every fold. This is neither: the pooled
effect is roughly **four times smaller than the materiality floor**, and MAE is
worse on all three folds. `coaching_scheme_features` stays `False`.

Sampling was halved to fit this container's kill window, which costs Monte Carlo
precision, and the plan was to call for a full-sampling re-run if the result
landed *near* the floor. It did not — it sits far inside the null zone, and all
three MAE folds share a sign, which is not the signature of noise concealing a
benefit. The reduced precision does not change the call.

### Why a screen at +0.211, p=0.003 buys nothing in the model

This is the useful part, and it generalises past coaching.

The screen controlled for the **team's** previous three seasons. The model does
something much stronger: it conditions on each **player's** own prior role. For
a returning back, his own prior target share already encodes most of what his
offence does with the backfield — including whatever his coach's tendency
contributed to it. The coach term was adding information over a team-level
baseline and then being asked to add information over a player-level one, and
almost all of it was already there.

So the screen was not wrong; it was answering an easier question than the gate
asks. **A screen that controls for team history is not a proxy for a model that
already conditions on player history.** Any future feature that lives at
team-season granularity faces the same gap, and the screen threshold that would
justify building it needs to be far above "clears a team-level control".

That closes the coaching-history line. Roughly a dozen hypotheses screened, one
survivor, and the survivor does not translate into forecast accuracy. The
scraper, the tables and the screens stay — they are what make the conclusion
trustworthy rather than assumed — but nothing from them ships.

## The cold-start under-projection: the claim curve counts draft capital twice

`measure_cold_start_bias.py` established the bias — cold rows under-projected on
every fold and both streams, established starters over-projected to match, which
inside a softmax is not two findings but one, since shares sum to one within a
team. `scripts/diagnose_cold_start_prior.py` locates it.

**It is the role prior, not the snap model.** Rebuild the deterministic
allocation with each row's *observed* snap share — perfect foresight on playing
time — and the gap barely moves:

| population | target | carry |
|---|---:|---:|
| warm | +7.2% | +5.8% |
| cold | **−39.3%** | **−26.9%** |
| rookies | −27.5% | −28.0% |

Nothing about projecting playing time is responsible.

**Conditional on playing, draft capital barely moves per-snap usage.** Among
cold players with 50+ snaps:

| slot | curve pays | observed | paid/obs |
|---|---:|---:|---:|
| round 1 | 0.1491 | 0.1176 | 1.27 |
| rounds 2–3 | 0.0735 | 0.0894 | 0.82 |
| rounds 4–7 | 0.0248 | 0.0854 | **0.29** |
| undrafted | 0.0093 | 0.0658 | **0.14** |

Observed rate spans 1.8× end to end. The curve spans 17×. A late-round or
undrafted player who earns a role is paid a seventh of what he produces.

**The cause is a units mismatch, visible in the source.**
`ffmodel.features.draft_calibration` fits every curve against `target_share` /
`carry_share` — volume shares, which already contain playing time. But
`_role_prior` consumes the result as a *per-snap rate*; its own comment says so
outright ("the cold-start prior stands in for `per_snap_role` … so it has to be
a per-snap rate"). The softmax score is `log(role_prior) + log(exposure)`, so
exposure lands twice: once baked into a curve fitted on shares, and again as the
offset.

Round 1 against undrafted:

| | ratio |
|---|---:|
| observed snap share (exposure) | 7.14× |
| observed per-snap target rate | 1.79× |
| **what the claim curve applies** | **29.96×** |

So the model applies roughly **214×** of draft capital where the data supports
about **13×**, and the softmax hands what it strips from late-round rows to
whoever else is in the room — which is exactly why the same measurement finds
established starters over-projected. One error, two symptoms.

This also explains two earlier results that looked unrelated. The wide
`cold_role_innovation` scale helps because a lognormal mean shift is a crude
patch over a systematically-too-low location; and `mean_preserving_innovation`
made every cold cell *worse* because removing that patch exposes the underlying
error rather than fixing it. Neither was ever about allocation noise.

**A second, smaller defect sits beside it.** `rookie_seasons` keeps only rows
with a non-null `overall_pick`, so undrafted players are excluded from the fit
entirely and then served by extrapolating the exponential to a stand-in pick of
220. That is **61% of cold rows** taking a value from beyond the end of the
fitted data, validated against nothing.

## The refit fails the pipeline gate

`scripts/validate_volume_fix_walkforward.py`, paired arms on identical frames
(fingerprint verified equal), holdouts 2022/2023/2024, 500 draws / 500 tune.
The baseline arm rebuilds the pre-refit share curves in place from
`LEGACY_SHARE_FIT_CURVES` via `--legacy-rookie-prior`, so the two arms differ in
the curve and in nothing else.

| stream | metric | pooled | 2022 | 2023 | 2024 | better |
|---|---|---:|---:|---:|---:|---:|
| target | MAE | **+5.55%** | +2.76 | +0.12 | **+14.98** | 0/3 |
| target | CRPS | −2.24% | −4.40 | −5.60 | +4.15 | 2/3 |
| carry | MAE | **+2.94%** | +0.50 | +6.00 | +2.63 | 0/3 |
| carry | CRPS | +1.14% | −1.60 | +3.55 | +1.83 | 1/3 |
| pass_qb | MAE | **+1.92%** | +0.82 | +0.26 | +4.97 | 0/3 |
| pass_qb | CRPS | −0.36% | −1.17 | −1.36 | +1.71 | 2/3 |
| snap | MAE, CRPS | +0.000% | — | — | — | — |
| availability | MAE, CRPS | +0.000% | — | — | — | — |

Positive is worse. **The gate rejects it**: the promotion rule requires winning
every fold, and MAE loses every fold on all three volume streams. `snap` and
`availability` being identical to the last decimal confirms the arms are cleanly
isolated — the curve only touches the volume priors. `pass_qb` moves because the
refit also changed `draft_pass_prior` (QB pass 0.78 → 0.6198), which was not the
defect under investigation.

Coverage moves the wrong way too, and my first reading of it was wrong. Nominal
is 0.80, and the baseline already over-covers at 0.847; the refit widens to
0.869, further from nominal, not closer.

| | base | refit |
|---|---:|---:|
| target cov80 | 0.847 | 0.869 |
| target cov95 | 0.954 | 0.968 |
| carry cov80 | 0.880 | 0.884 |
| pass_qb cov80 | 0.810 | 0.787 |

So there is no metric on which this is a clean win. CRPS improves on 2022 and
2023 and gives it back on 2024, in the same fold where MAE blows out by 15%.

## Why it fails: the rate fit conditions on survival

The units diagnosis is arithmetic and it still stands. What the refit got wrong
is the population it fitted on. Rate mode keeps only rookies with 50+ snaps —
necessary, because a rookie with four snaps has a per-snap rate that is pure
noise — but that is conditioning on *having earned a role*. The curve it
produces answers "given a rookie plays, how does draft slot move his per-snap
usage?", and the answer is genuinely "barely, 1.8× end to end". The pipeline
then applies that curve to every rookie, including the ones who never play.

The effect on where prior mass sits, WR targets:

| slot | legacy | refit | ratio |
|---|---:|---:|---:|
| pick 1 | 0.2200 | 0.1412 | 0.64× |
| pick 32 | 0.1312 | 0.1306 | 1.00× |
| pick 64 | 0.0770 | 0.1205 | 1.57× |
| pick 100 | 0.0423 | 0.1101 | 2.61× |
| pick 150 | 0.0184 | 0.0971 | 5.29× |
| undrafted | 0.0057 | 0.0814 | **14.24×** |

Over the 1,731 cold skill rows in the cache — 61.6% of them undrafted — the mean
prior triples, 0.0243 → 0.0803, and the undrafted share of all cold prior mass
goes 22.1% → **54.8%**. The softmax normalises within team-season, so that is
not a harmless rescale: it is camp bodies taking share from real players, in
every room, on every fold. A 15% MAE blowout on 2024 is what that looks like.

So the steep curve was doing two jobs, and the diagnosis only named one. It
prices per-snap usage — where it is indeed 17× too steep — *and* it prices the
probability that draft capital converts into a role at all, over and above what
the snap model projects. The snap model cannot fully separate a seventh-rounder
who will start from one who will be cut, because in the offseason very little
distinguishes them except draft slot. Flattening the curve on rate removed both
jobs, and the second one was load-bearing.

That is also the honest verdict on the earlier held-out allocation numbers
(cold bias 28.9% → 7.2%). Bias is a statement about the *mean* of the cold
population, and lifting several hundred players who will never take a snap
raises that mean toward the truth while making almost every individual
projection worse. Bias improved and accuracy did not. The two are not the same
question and this measurement conflated them.

## What the identity actually requires

Measured over every rookie in the cache — zero-snap players included, which is
the population the prior is applied to — round 1 against undrafted:

| bucket | n | snap share | target share | implied per-snap rate |
|---|---:|---:|---:|---:|
| rd1 | 74 | 0.5528 | 0.1379 | 0.2494 |
| rd2–3 | 195 | 0.3699 | 0.0724 | 0.1956 |
| rd4–7 | 394 | 0.1776 | 0.0321 | 0.1810 |
| undrafted | 348 | 0.0638 | 0.0101 | 0.1589 |
| **rd1/undrafted** | | **8.66×** | **13.59×** | **1.57×** |

So the refit's 1.67× is the right number *for a per-snap rate*, and the units
diagnosis is vindicated on observed quantities. But the identity
`share = exposure × rate` only closes inside the model if the model's
**projected** exposure carries the 8.66× itself, and that cannot be assumed.
The snap model is projecting players with no NFL history; shrinking an unknown
row toward its position mean is exactly what a hierarchical model does. Whatever
spread projected exposure fails to carry has to live somewhere, and under the
legacy curve it lived in the steepness the diagnosis called a double count.

`scripts/diagnose_rookie_exposure_spread.py` measures that directly.
