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

How much shrinkage is the open question. Players whose prior allocation landed
in [0.23, 0.29) went on to average **0.2160** (median 0.2172, n=226). The model
gives Smith 0.1770, about 18% below that. The log-log fit predicts 0.1879, but
that is a conditional geometric mean and sits below the arithmetic mean by
construction, so 0.2160 is the number a posterior *mean* should be compared
against.

That 18% is suggestive and not yet a finding. Conditioning on the prior and
conditioning on the model's own projection are different questions with opposite
regression artifacts, so this must not be read as agreeing or disagreeing with
the `warm_projected_top` over-projection measured above — those are counts per
team game conditioned on the projection, these are shares conditioned on the
prior. Settling it needs a walk-forward that reports projected-versus-observed
*share* conditioned on the projected share, which nothing here has run.
