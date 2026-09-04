# The weekly layer (2026-08-27)

Two responses, one panel, built independently of the season-average pipeline.
Nothing here reads a season projection and nothing here is constrained by one.

**Model 1 — next week.** Points in week `w`, given every week before it. The
start/sit decision.

**Model 2 — rest of season.** Points from week `w` to the end of the regular
season, given every week before it. At `w = 1` this is the draft question; from
about week 5 it is the waiver question.

Code in `src/ffmodel/weekly/`. Validated by `scripts/validate_weekly.py`,
checked against the draft board by `scripts/compare_weekly_to_adp.py`, blended by
`scripts/blend_weekly_with_market.py`, diagnosed by
`scripts/diagnose_weekly_calibration.py`, run by `scripts/project_week.py`.

## The bar: a naive draft board

Every claim here is measured against ADP, because ADP is what a manager already
has for free. The weekly restatement is the season layer's rank curve prorated:
a per-position log fit of season points on draft rank, divided by games for one
week, multiplied by games remaining for the rest of a season, with spread from
the curve's own residuals at nearby ranks.

It is stale by construction and increasingly so — published in August, it knows
everything in week 1 and nothing new by week 12 — so **the comparison is always
reported by week**, never pooled alone. And it is scored on the **drafted pool**,
because beating a board on players it declined to rank is close to free.

### Where the models beat it, and where they do not

Against the ADP curve on the drafted pool, walk-forward over 2023/2024/2025:

| response | pooled | weeks 1–4 | weeks 5–10 | weeks 11–18 |
|---|---|---|---|---|
| **next week** | wins 7/7 | wins 6/7 | wins 7/7 | wins 7/7 |
| **rest of season (blended)** | wins 7/7 | wins 4/7, ties 1 | wins 7/7 | wins 7/7 |

Next week, pooled drafted pool: MAE −10.8%, CRPS −19.0%, within-position
Spearman +75%, top-24 hit rate +26.6%, and CRPS wins on 3/3 folds individually.
Late in the season the margin widens to CRPS −24.9%.

**The three metrics that do not clear the bar, all in weeks 1–4:**

| response | metric | model | ADP | margin | folds |
|---|---|---:|---:|---:|---|
| next week | top-24 hit rate | 0.278 | 0.329 | −15.3% | lost 3/3 |
| rest of season | within-position ρ | 0.535 | 0.543 | −1.5% | lost 2/2 |
| rest of season | top-24 hit rate | 0.459 | 0.483 | −5.0% | lost 1/2, tied 1 |

The pattern is specific and worth stating precisely: **early in the season the
model orders the whole position better than the board and picks the very top
worse.** Within-position Spearman is a win in weeks 1–4 on 3/3 folds for next
week; the top-24 hit rate is a loss on 3/3. A draft board is a consensus ranking
built by people concentrating on the players who go early, and that is exactly
where it stays sharper than a model reading box scores. Everywhere else, and on
every loss metric everywhere, the models win.

This is the season layer's finding reappearing at a new cadence, and it was
attacked the two ways that package already has evidence for.

**ADP as a feature, not just a rival.** The board carries the one thing usage
history cannot: an offseason. A trade, a rookie, a vacated backfield reaches no
box score until it is too late to help. Entering `adp_log_rank` and
`adp_drafted` interacted with an early-season indicator — so the fit leans on the
board while history is thin and discounts it after — cut the rest-of-season
early-week deficit from −3.05% MAE to −0.51%, and from −4.78% CRPS to −0.69%.

**A blend, weighted per horizon.** The variance-optimal weight on the model
against the curve is the slope of `observed - curve = a + b(model - curve)`.
Estimated separately per horizon, from earlier holdouts only, it produces the
schedule this section argues for without being told to:

| horizon | 2024 weight | 2025 weight |
|---|---:|---:|
| weeks 1–4 | 0.65 | 0.51 |
| weeks 5–10 | 1.00 | 0.93 |
| weeks 11–18 | 1.00 | 1.00 |

The model earns its weight as the season gives it something the board never saw.
Pooled over the two scorable folds, the blend beats the curve on **every metric
overall and in mid and late**, and in weeks 1–4 beats it on MAE (+2.0%), RMSE
(+1.0%), CRPS (+2.1%) and 80% coverage while ties on 95% and trailing on the two
ordering metrics above. It also beats the unblended model on all three folds.
2023 is unscored throughout: no earlier holdout exists to take its weight from.

Mixture, not averaging — averaging paired draws produces a distribution narrower
than either input and wrecks calibration, which the season layer measured at
0.689 against a nominal 0.80.

## Where the error actually is (2026-08-27, follow-up)

`scripts/diagnose_weekly_errors.py` attributes the error to segments and reports
the signed bias, because over- and under-projection want opposite fixes. It
overturns the framing above and finds a different, larger problem.

### The next-week model does not get worse early. The board gets better.

| window | model MAE | model CRPS | ADP MAE | ADP CRPS |
|---|---:|---:|---:|---:|
| weeks 1–4 | 4.967 | 3.376 | 5.590 | 4.061 |
| weeks 5–10 | 4.974 | 3.329 | 5.976 | 4.407 |
| weeks 11–18 | 5.055 | 3.412 | 6.317 | 4.708 |

The model is flat across the season to within a rounding error. **Every bit of
the narrowing early-season gap is the draft board being good in September and
decaying from there** — its CRPS worsens by 16% from week 1 to week 18 while the
model's moves 1%. "The model struggles early" is the wrong description of the
next-week response and the wrong thing to go and fix.

It remains true for rest of season, where weeks 1–4 really are harder in absolute
terms — but most of that is horizon length (a 17-game sum against a 3-game one),
not a defect that appears in September.

### The real failure is role change, and it is not seasonal

Bias is projected minus observed, so positive is over-projection. Both windows,
relevant population plus anyone the board drafted:

| segment | share of error | observed | projected | bias (early) | bias (later) |
|---|---:|---:|---:|---:|---:|
| **role grew (share +10pt)** | 14% | 15.0 | 8.0 | **−6.99** | **−6.87** |
| **role shrank (share −10pt)** | 22% | 2.4 | 6.4 | **+4.00** | **+4.00** |
| did not play | 20–22% | 0.0 | 3.9 | +3.91 | +3.70 |
| returning after 1 week out | 7–9% | 3.8 | 5.7 | +1.88 | +2.80 |
| handcuff: lead back out | 3.5% | 6.7 | 4.8 | −1.95 | −2.19 |

The two role-change rows are the same number in both windows. This is not an
early-season problem that fades; it is a permanent property of a model whose
usage features are all exponentially weighted averages of what a player has
already done. A promotion enters the features only as it is produced.

The board is worse on both (−7.3/−7.7 on role growth, +4.3/+5.2 on role
decline), so this is not something ADP solves — it is what any
backward-looking method does.

### How long it takes to catch up: about two weeks

Following the model's bias forward from the week a player's share of his team's
carries or targets first steps up by ten points and reaches twenty percent:

| weeks since the role changed | n | bias | MAE | observed |
|---|---:|---:|---:|---:|
| 0 (the week it happens) | 591 | **−7.41** | 8.11 | 14.57 |
| 1 | 566 | −1.98 | 5.80 | 10.37 |
| 2 | 543 | −0.49 | 5.48 | 9.17 |
| 3 | 533 | −0.30 | 5.81 | 8.97 |
| 5 | 476 | −0.06 | 5.61 | 8.76 |

**The entire cost is the first week.** Seven and a half points low the week a
role opens, two the week after, unbiased by the second. The four-game half-life
is doing exactly what it was specified to do, and the damage is concentrated
where no amount of retuning the decay will help — in the week before there is
any new data at all.

### Handcuffs specifically

A backup running back in a week his team's established lead rusher (lagged carry
share ≥ 0.35) is inactive: **the model projects him 1.9–2.2 points low**, on 86
early rows and 503 later ones, about 3.5% of total error. Smaller than the
general role-growth miss because many activations do not produce a full takeover
— a committee absorbs the carries. The board is slightly worse (−2.2/−2.3).

The gap between the handcuff miss (−2.1) and the role-growth miss (−7.0) is the
useful part: knowing the starter is out is worth much less than knowing the
backup will actually get the volume, and only the second is a large error.

### The two hypotheses that did not hold

**Rookies and thin history are not the problem.** On players with no career
weeks at all the model is essentially unbiased (+0.10 early) and beats the board
outright (CRPS 2.15 against 3.19) — the board over-projects them by +2.94. Thin
history (1–7 games) is the same story: bias −0.49, CRPS 2.98 against the board's
3.94. Imputing missing history to the training median with an explicit
"no history" indicator, plus the ADP columns, handles these rows better than the
consensus that was built to price them.

Note that the headline `relevant` population cannot see this at all — it requires
four prior appearances and so excludes every rookie by construction. The
diagnostic has a `--population drafted-or-relevant` mode for exactly this reason.

**Low-ADP over-projection is real but small, and only for the undrafted.**
Players the board declined to rank are over-projected by **+0.71 early** against
−0.17 later, which is the "last year's part-timer projected into a role he has
already lost" effect and it is worth about seven tenths of a point on players
averaging 2.6. Within the drafted board there is no rank gradient in the bias at
all: ADP 1–36 −0.26, 37–84 −0.60, 85–150 −0.89, 151+ +0.04.

### What this points at

Ranked by share of error, and none of it is a decay-rate question:

1. **Role change, in the first week only** (36% of error across both directions).
   Wants a leading indicator of role, not a better average of the trailing one:
   snap share and depth charts are both cached and unused, and a depth chart is
   published before the game.
2. **Absence** (20–22%). The model projects 3.9 points for players who score
   zero, and over-projects returners by 1.9–2.8. The availability half reads
   appearance history and nothing else; the weekly injury report is cached and
   unused.
3. **Everything else.** No segment above accounts for more than 3.5%.

## Feeding each simulated week into the next (2026-08-28)

The aggregated simulator's failure is under-dispersion, and better features did
not touch it: coverage moved 0.575 to 0.579 while the mean improved. That points
at the construction rather than the inputs. Both the hierarchical and the
independent arms hold a player's history **frozen at week `w`** for every
simulated game — they draw outcomes but never let those outcomes change what the
player looks like going into the next one.

`RecursiveSeason` closes that loop. Each simulated week updates the player's own
history exactly as a real week would — the drawn outcome moves his
recency-weighted average, his last-observation column, his play rate, his
games-played count and his weeks-since-played — and the following week is
predicted from the updated state. A hot streak inside a draw raises the
projection generating the next week of that same draw, so trajectories fan out
the way careers do and the spread of season totals is **produced by the process
rather than asserted by a variance component**.

It is affordable because both halves of the hurdle are linear in the standardised
design: moving feature `j` by `d` moves the linear predictor by
`coefficient_j * d / scale_j`, so a week's update is a handful of array
operations rather than a rebuild of the feature frame for every draw.

### It is the only thing that has moved the calibration

Relevant population, three holdouts, n = 13,859:

| arm | MAE | CRPS | cov80 | cov95 | PIT dev | top-24 |
|---|---:|---:|---:|---:|---:|---:|
| **direct total + everything** | **30.09** | **21.27** | **0.790** | **0.945** | **0.148** | 0.369 |
| **recursive weekly** | 30.99 | 22.61 | 0.673 | 0.854 | 0.298 | **0.386** |
| aggregated weekly | 30.28 | 23.23 | 0.575 | 0.752 | 0.475 | 0.386 |
| hierarchical (original) | 30.45 | 23.07 | 0.579 | 0.757 | 0.481 | 0.353 |

Against the frozen-feature simulator it improves on **3/3 folds on every
distributional metric**: 80% coverage 0.575 → 0.673, closing 44% of the gap to
nominal; 95% coverage 0.752 → 0.854, closing 52%; PIT deviation down 37%; CRPS
down 2.7%; and the over-projection bias halved, +2.56 → +1.24. Early in the
season the effect is larger still — coverage 0.443 → 0.580, CRPS 40.73 → 38.27.

The idea is right, it is the first thing to move a number that three rounds of
feature work left untouched, and it costs about 2% of MAE for it: the wider draws
push slightly more mass against the zero floor.

### Fitting the error the recursion cannot generate

The recursion propagates uncertainty about **scoring**, because points are what
it simulates. It cannot propagate uncertainty about **role**, because usage and
the offence stay frozen. What cannot be generated can still be measured.

`_calibrate_drift` simulates the most recent training season with the drift
switched off and compares the spread the simulator produced against the spread
the outcomes actually had. A level shift of `d` per game moves a `G`-game total
by `G · d`, so

```
Var(observed − predicted) = Var_simulated + G² · drift²
```

and the remainder identifies `drift` directly — a method of moments on the
simulator's own shortfall, in the same spirit as the Beta concentration and
persistent-SD estimators, but measured against *this* construction rather than
assumed from weekly residuals. Those weekly residuals are exactly what the
recursion already reproduces, which is why the term has to be fitted here and not
inherited. It is estimated on the last training season only, never on a holdout,
and a negative remainder returns zero rather than a nonsensical negative SD.

The fitted value is stable across folds: **2.12, 2.17 and 2.48 points per game**,
about 36–42 points of season-total standard deviation.

| arm | MAE | CRPS | cov80 | cov95 | PIT dev |
|---|---:|---:|---:|---:|---:|
| direct total + everything | **30.09** | **21.27** | **0.790** | **0.945** | 0.148 |
| **recursive + drift** | 31.01 | 22.18 | 0.769 | 0.907 | **0.140** |
| recursive weekly | 30.99 | 22.61 | 0.673 | 0.854 | 0.298 |
| aggregated weekly | 30.28 | 23.23 | 0.575 | 0.752 | 0.475 |

The full progression on 80% coverage against a nominal 0.80 is **0.575 → 0.673 →
0.769**, and on PIT deviation **0.475 → 0.298 → 0.140**. The drift term improves
coverage at both levels and CRPS on **3/3 folds**, and leaves MAE untouched
(30.99 → 31.01), which is what a variance correction should do.

**On distributional shape the simulator now edges the regression**: PIT deviation
0.140 against 0.148, and in weeks 1–4 it wins both PIT (0.229 against 0.237) and
80% coverage (0.725 against 0.718). Two rounds ago it covered 0.443 there.

### Simulating usage jointly: right where the horizon is long, a wash elsewhere

The drift term is a scalar: every player gets the same per-game allowance for
role uncertainty. `UsageProcess` replaces the assumption with a fitted process.
A share is bounded, so it is modelled in logit space as an AR(1) reverting toward
the player's **own** standing level rather than the population's:

```
logit(s_t) = a + b · logit(s_{t-1}) + c · L_{t-1} + e
```

Fitted per position, for the primary opportunity share (carries for backs,
targets for everyone else) and for snap share. The estimates are sensible —
persistence 0.32–0.55, reversion 0.31–0.65, both rising for backs, whose carry
share is the stickiest thing in the panel:

| share | position | persistence | reversion | innovation |
|---|---|---:|---:|---:|
| primary | RB | 0.448 | 0.496 | 1.275 |
| primary | WR | 0.317 | 0.595 | 0.918 |
| snap | RB | 0.491 | 0.435 | 1.084 |
| snap | WR | 0.505 | 0.460 | 1.249 |

The simulator carries both shares as state, steps them each played week, and the
projection moves through the same linear channel as everything else. The
mechanism is not a no-op and was checked as one: across draws the primary share
fans out from 0.061 to 0.092 standard deviations over eight weeks, moving the
projection by about 0.9 points a game.

| arm | MAE | CRPS | cov80 | PIT | **bias** | **top-24** |
|---|---:|---:|---:|---:|---:|---:|
| recursive + drift | 31.01 | **22.18** | **0.769** | **0.140** | +1.44 | 0.381 |
| recursive + usage | 31.12 | 22.24 | 0.765 | 0.142 | **+0.29** | **0.396** |

**Pooled it is a wash, and slightly negative**: CRPS 0.3% worse on 3/3 folds,
coverage and PIT fractionally worse. The falsifiable prediction failed — if the
process were generating the role uncertainty the scalar was injecting, the fitted
drift would have fallen toward zero. It moved 2.12 → 2.11.

The reason is that the fitted process is **stationary**. Persistence and
reversion put next week between last week and a standing level, so the deviation
does not accumulate; over seventeen games it contributes on the order of
`sqrt(G)` rather than `G`. Season-total uncertainty needs the *persistent* kind
of role change — a role that moves and stays moved — and a mean-reverting AR
around a slowly-updating level cannot produce much of it. The scalar drift, which
is a permanent per-game shift, is a cruder object that happens to have exactly the
right accumulation.

**Where it does pay is the long horizon, which is where the mechanism should
matter.** In weeks 1–4, projecting seventeen games, it improves CRPS on **3/3
folds** (37.11 → 36.98 pooled) and reduces the absolute bias on **3/3**
(8.63 → 6.14). And it gives the best top-24 hit rate of any arm anywhere in this
document — 0.396 pooled, 0.434 early, against the board's 0.384 and the direct
fit's 0.369 — improving on 2/3 folds with one tie.

One honest caveat on the headline bias figure: pooled it falls from +1.44 to
+0.29, but per fold the absolute bias improves only 2 of 3 (2024 moves −2.23 →
−3.55). Much of the pooled improvement is sign cancellation across folds, not a
uniformly smaller error.

So: the right mechanism, fitted honestly, and it buys ordering and long-horizon
accuracy rather than the calibration it was built for. If the aim is to generate
season-total spread from first principles rather than inject it, the missing
ingredient is a **non-stationary** level — letting a player's standing role take a
random walk rather than only deviating around one. That is a change to the
process, not to the simulator, and it is untested.

### The non-stationary level: refuted before it was built

The previous section ended by proposing one: let a player's standing role take a
random walk rather than only deviate around a slowly-updating anchor, since a
stationary process contributes about `sqrt(G)` to a `G`-game total and a
wandering one contributes `G`. That is the right diagnosis of the mechanism and
the wrong theory about the data.

A mean-reverting process and a random walk are indistinguishable one week apart
and obvious twelve weeks apart, so the horizon is the identifying variable. For an
AR(1) around a fixed level the variance of the `h`-week change has a ceiling; for
a random walk it grows linearly in `h` forever. Measured within player-seasons on
2016–2022, as a ratio to the one-week variance:

| quantity | h=1 | h=4 | h=8 | h=12 |
|---|---:|---:|---:|---:|
| **primary share** (carries/targets, logit) | 1.00 | 1.20 | 1.27 | **1.19** |
| snap share (logit) | 1.00 | 1.27 | 1.36 | 1.39 |
| **team plays** | 1.00 | 1.06 | 1.07 | **1.04** |
| **points per game** | 1.00 | 1.12 | 1.26 | **1.35** |
| points per opportunity | 1.00 | 1.03 | 1.03 | 1.15 |

**Opportunity share does not wander.** It plateaus by eight weeks and turns down
by twelve; the fitted random-walk innovation comes out at exactly 0.0000. Team
volume is flatter still, at 1.04. The proposed process does not exist to be
built, and building it would have added variance the data says is not there.

**The thing that does keep accumulating is scoring.** Points per game is the only
quantity still climbing at twelve weeks, and points per opportunity carries the
tail of it. So the accumulating uncertainty in a season total is not about how
much work a player gets — that reverts — but about what he does with it.

Which retires the standing criticism of the drift term. It was described as a
crude scalar standing in for role drift; it is nothing of the kind. It is applied
to the **points level**, which the variance ratio identifies as exactly where the
accumulation lives. The sophisticated usage process was solving a problem that
does not exist, and the blunt instrument was already pointed at the one that does.

One caveat on reading the table: only stable players survive twelve consecutive
weeks, so the long horizons are survivorship-thinned. That affects every row
equally, though, and the *contrast* is what carries the argument — points per game
rises on the same population where share falls.

`estimate_random_walk` and its test remain in the package. The test builds two
series with known answers, one mean-reverting and one wandering, and asserts the
estimator tells them apart; a measurement that decided a design question should be
demonstrably able to decide it.

### Scaling the drift with the projection: also nothing

If the accumulation is in scoring, the next refinement is obvious — a twenty-point
back and a five-point backup should not carry the same absolute allowance. The
same moment condition solves for `drift_i = a + b · mu_i` by regressing the
remaining variance on `mu²`.

The fitted slope is **0.010**. A twenty-point projection earns two tenths of a
point a game more drift than a replacement-level one, on a base of 2.02. On the
full walk-forward the two arms are identical to the fourth decimal:

| arm | MAE | CRPS | cov80 | PIT dev |
|---|---:|---:|---:|---:|
| recursive + drift | 31.011 | **22.177** | **0.769** | **0.140** |
| recursive + scaled drift | **31.005** | 22.185 | 0.765 | 0.143 |

The allowance is genuinely close to homoskedastic in absolute terms, which is not
what intuition suggests and is what the estimate says. Kept in the code behind
`drift_scales`, defaulting off, as a measured null.

### And it still loses to the regression

Even with the drift term, CRPS is 22.18 against 21.27 — **4.3% worse, on 3/3
folds**. So it does not change what ships. The gap is now entirely the mean: the
simulator's MAE is 3.1% worse (31.01 against 30.09), and CRPS pays for that even
though the simulator's distributional shape is the better of the two.

The reason is visible in what the recursion can and cannot carry. It propagates
uncertainty about a player's **scoring**, because points are what it simulates.
It cannot propagate uncertainty about his **role**, because usage shares, snap
share and the offence around him stay frozen at week `w` — simulating those means
simulating the whole team. And role change is precisely what the error
attribution identified as 36% of weekly error and mostly unannounced. The half of
season-total uncertainty that this construction was built to recover is the half
it can already see; the other half needs a different model.

Two further limitations, neither hidden by the result. The model is fitted on
real histories and fed its own, so by week ten of a draw the features it reads
were generated by itself — the standard exposure problem in recursive multi-step
forecasting, and the reason this had to be measured rather than assumed better.
And the exponential updates use the recursive form while the feature layer builds
its averages with pandas' adjusted weighting; the two agree closely once a player
has several observations behind him, which he does here by construction.

## Does aggregating the weekly model up to a season work now? (2026-08-28)

The composition test was last run with the original weekly components. Since
then the weekly model has gained the injury report, snap share, ADP, per-position
fits and a much faster decay, and is 30% better than the draft board. The
question is whether any of that carries to a season total.

`aggregated-weekly` is the hierarchical simulator given the same feature surface
as the shipped weekly model — minus the per-position fits, which are not
implemented there, and minus the single-game script terms, which are deliberately
excluded because a spread describes one game and this response spans seventeen.

Relevant population, three holdouts, n = 13,859:

| arm | MAE | CRPS | cov80 | ρ | top-24 | PIT |
|---|---:|---:|---:|---:|---:|---:|
| ADP curve | 35.08 | 24.83 | 0.668 | 0.687 | 0.384 | 0.305 |
| direct total | 31.51 | 22.33 | 0.784 | 0.763 | 0.305 | 0.160 |
| **direct total + everything** | **30.09** | **21.27** | **0.790** | **0.780** | 0.369 | **0.148** |
| aggregated weekly | 30.28 | 23.23 | 0.575 | 0.751 | **0.386** | 0.475 |
| hierarchical (original features) | 30.45 | 23.07 | 0.579 | 0.774 | 0.353 | 0.481 |

**The mean caught up. The distribution did not.** Aggregation is now within 0.6%
of the direct fit on MAE and splits the folds on it (winning 2023, losing 2024
and 2025), and it has the best top-24 hit rate of any arm. Its 80% interval
covers **0.575 against a nominal 0.80**, its CRPS is **9.2% worse**, and the
direct fit wins CRPS on 3/3 folds. Early in the season it is worse on everything:
MAE 51.85 against 48.16, CRPS 40.73 against 34.03, coverage 0.44 against 0.72.

**And almost none of the weekly gains transferred.** Against the same simulator
carrying the original feature set, the upgraded one moves MAE 30.45 → 30.28
(−0.6%) and CRPS 23.07 → 23.23 (**0.7% worse**). Everything that bought 8% on the
weekly response bought nothing here.

That is not mysterious once stated: the features that made the weekly model
better are short-horizon by construction. This week's injury status, last week's
snap share and a one-game decay describe the next game. Over seventeen games a
one-game decay is noisier than a four-game one, a Friday game status is nearly
irrelevant by November, and the useful quantity is a stable level rather than a
current state. The direct fit regresses the season total on the same columns and
learns the season-appropriate weighting from season-length data; the simulator
inherits the weekly weighting and compounds it.

Fourth time in this package: **fit the thing you are going to be scored on.** The
rank curve beat the season pipeline, composition cost nothing over projecting
points directly, the simulator lost to a regression, and now an improved
simulator still loses to the same regression by the same margin.

The one place aggregation is genuinely competitive is the top-24 hit rate
(0.386, best of any arm including the board's 0.384). If the question is "name
the best players" rather than "how many points", simulating forward is not a bad
way to ask it — and that is a narrow enough win, on a noisy metric, that it is
recorded rather than acted on.

### A stale-cache bug worth recording

The blend re-run after all these changes returned byte-identical numbers, which
looked like a clean null and was not. Two feature caches existed — a plain build
and a `+news` build — and `blend_weekly_with_market.py` still defaulted to the
plain one, so it had been scoring a pre-news, pre-snap, half-life-four frame
through an entire round of changes. There is now one canonical path,
`ffmodel.weekly.FEATURES_CACHE`, and every script reads it.

Two consequences for the numbers above the fold: the blend figures in the ADP
section were produced on that stale frame and have been re-run here, and the
error-attribution and role-signal diagnostics were run at intermediate
configurations. Their qualitative findings do not depend on the model version —
the role-signal ceiling is a fact about the feeds, and the error segmentation
reproduces — but the exact biases quoted there belong to the configuration
current when each was run, not to the shipped model.

Blend, drafted pool, on the corrected frame:

| fold | arm | MAE | CRPS | cov80 |
|---|---|---:|---:|---:|
| 2024 | curve | 36.98 | 26.75 | 0.636 |
| 2024 | model | 36.57 | 26.09 | 0.779 |
| 2024 | **blend** | **35.77** | **25.31** | **0.793** |
| 2025 | curve | 35.89 | 25.59 | 0.661 |
| 2025 | model | 34.17 | 24.12 | 0.805 |
| 2025 | **blend** | **33.34** | **23.29** | **0.825** |

It still beats both components on both folds, and in weeks 1–4 still beats the
curve where the model alone does not (2025: blend 54.14 MAE against the curve's
56.48 and the model's 57.25).

## Snaps, the decay, and what is left (2026-08-28)

Three pieces of work following the role-change diagnosis: the leading indicator
that was blocked on a join, the one parameter this layer had never tuned, and a
hunt for error sources outside role and absence.

### Snap counts, finally joined

The feed is keyed on Pro-Football-Reference ids, which nothing else here uses.
The bridge is the weekly roster, which carries `pfr_id` and `gsis_id` on the same
row; pooling it across every season rather than matching within one lands
**92.9% of played rows** (0.79 in 2016 rising to 0.99 by 2024).

Snap share is the most direct role measure available — targets say what a player
did with his field time, snaps say how much he got — and it moves first. It is
worth **−0.67% CRPS, improving on 3/3 folds**, with MAE flat. Real, consistent,
and much smaller than its billing as "the largest known gap". Being first to move
is not the same as moving far enough ahead to matter.

### The decay: one game, not four, and the sweep has a trap in it

`HISTORY_HALFLIFE` was set at four games a priori and never tuned. Selected
properly — candidates scored on 2021–2022, strictly earlier than any reported
holdout — the curve is monotone with an interior optimum at **one game**, on a
plateau from 0.5 to 1.5, worth **3.36% CRPS** against the a-priori four.

**The first run of that sweep gave the opposite answer and it was a bug worth
recording.** `relevant_population` reads
`prior_points_recent_given_played`, which is itself an average at the half-life
under test. Scoring each candidate on "its own" relevant rows compared different
populations: the eight-game arm admitted 10% more rows (11,492 against 10,425),
those extra rows were marginal players who are easier to project, and the result
was a spurious 0.93% win for the *longest* decay — the exact reverse of the
truth. Fixing the population at a reference half-life flips it.

A second check looked like a contradiction and is not. The same feature measured
on its own clearly prefers a six-game window (MAE 5.797 against 6.183 at one
game, fixed rows). Both are right: the model already carries a long-run level in
the expanding career averages, which no decay touches, and *given* that level the
most useful thing another feature can add is not a second slightly different
average but the most recent observation. Carrying both timescales explicitly
recovers about a third of the gap (3.192 → 3.168 against one-game's 3.085) and
adds nothing once the half-life is already short, which is the same statement
from the other side. The last-observation columns ship anyway: they are worth a
further **−0.38% CRPS on 3/3 folds**.

### What is left, and how much of it is anybody's to fix

`scripts/hunt_weekly_errors.py`, on the shipped model:

| segment | n | share of error | bias |
|---|---:|---:|---:|
| **offence beat its implied total by 10+** | 2,607 | 19.1% | **−2.44** |
| **offence missed its implied total by 10+** | 2,090 | 10.8% | **+2.18** |
| projected top 10% | 1,712 | 15.6% | −0.45 |
| big underdog (spread ≥ +7) | 2,238 | 13.5% | −0.41 |
| high total (≥ 48) | 3,252 | 21.0% | −0.40 |
| big favourite (spread ≤ −7) | 2,192 | 12.3% | +0.24 |
| position TE / RB | — | 14% / 26% | −0.31 / −0.27 |

**The largest remaining source is the game itself, and it is a floor rather than
a defect.** On 27% of rows a team's actual scoring misses the closing line's
implied total by ten points or more, and the model's error tracks that miss
almost one-for-one in the corresponding direction. Together those rows are ~30%
of all absolute error. The closing line is the best forecast of a game anyone
publishes and it is wrong by ten points more than a quarter of the time; nothing
in a player model recovers that.

**Touchdown regression: refuted.** Splitting on the touchdown share of a player's
recent scoring gives no monotone bias trend (−0.30, −0.33, −0.22 across the three
populated buckets; the >60% bucket holds 51 rows and says nothing). The lumpiness
of touchdowns is real but the model is not systematically fooled by it.

**A mild shrinkage signature.** The top projected decile is under-projected by
0.45 against an observed 17.4 — about 2.6% — and carries 15.6% of the error.
Small, consistent, and the population where lineup decisions are actually made.

**Game script is not fully captured even though the line is in the model.** Big
underdogs are under-projected by 0.41 and big favourites over-projected by 0.24,
which is the direction the mechanism predicts: a trailing team throws. The linear
spread term gets some of this and not all of it.

### Where that leaves the ladder

Relevant population, three holdouts, half-life one:

| arm | MAE | CRPS | ρ | top-24 |
|---|---:|---:|---:|---:|
| ADP curve | 6.127 | 4.559 | 0.399 | 0.092 |
| recency mean | 5.313 | 3.627 | 0.573 | 0.071 |
| hurdle | 5.131 | 3.459 | 0.607 | 0.090 |
| + context, per position | 5.049 | 3.423 | 0.615 | 0.104 |
| + ADP + news + snaps | 4.697 | 3.195 | 0.680 | 0.125 |
| **+ last observation (ships)** | **4.693** | **3.183** | **0.681** | **0.126** |

Against ADP on the drafted pool the shipped arm now fails **one** metric —
early-season top-24 hit rate at −4.42%, the best that number has been — and wins
everything else, CRPS by 27% on all three folds individually.

## Is role change a season boundary or a weekly one? (2026-08-27)

`scripts/decompose_role_change.py`. The two possibilities want opposite work: a
season-boundary problem is fixed by discounting last year's role and leaning on
preseason information, a weekly one is not fixed by anything at the boundary.

### Both, in almost equal measure — but only a third of role variance moves at all

Variance of a player's share of his team's carries or targets, over 2,830
player-seasons with at least six games:

| source | share of total variance |
|---|---:|
| between players — who he is | **66.0%** |
| between seasons, same player | 16.0% |
| week to week, inside a season | 18.0% |

Two thirds of role is simply who the player is, and the model has that. Of the
third that moves, the split is 47:53 between the offseason and the season —
neither dominates.

The week-to-week figure is corrected for sampling noise and that correction
decides the answer. A share measured over twenty-five carries wobbles because
the denominator is small, not because the role moved: raw within-season variance
is 0.00983, of which **0.00438 (45%) is binomial noise**. Counted uncorrected,
week-to-week would read 29% against between-season's 13% and the conclusion would
flip. `tests/test_role_decomposition.py` pins the correction with constructed
players whose true role is known.

### The season boundary is already handled about as well as it can be

Mean absolute error in predicting this week's share, by week:

| week | last season's share | this season's share | the model's blend |
|---|---:|---:|---:|
| 1 | 0.0790 | — | **0.0786** |
| 2 | 0.0801 | 0.0814 | **0.0765** |
| 3 | 0.0827 | **0.0738** | 0.0737 |
| 4 | 0.0838 | 0.0695 | 0.0707 |
| 8 | 0.0864 | 0.0712 | 0.0705 |
| 12 | 0.0895 | 0.0688 | **0.0675** |
| 14 | 0.0889 | 0.0734 | 0.0699 |

Three things fall out. **The crossover is week 3** — after two games, the current
season beats the previous one outright. **Last season decays as the year runs**,
from 0.079 to 0.089, which is roles drifting away from where they ended. And
**the model's exponentially weighted blend matches or beats the better of the two
at every single week**, including week 1, where it edges the only information
available. There is no free win in re-weighting the season boundary; the blend is
already doing it.

### And most of the error is not at the boundary anyway

First role step-ups by the week they occur — noting that "first in a season"
mechanically favours early weeks, since a player who steps up in week 2 cannot
also be counted in week 9:

| weeks | first step-ups |
|---|---:|
| 1–2 | 437 (22.7%) |
| 3–8 | 815 (42.3%) |
| 9–18 | 676 (35.1%) |

Even with that bias, **77% of first step-ups happen after week 2**. The error
attribution agrees from the other direction: the role-grew segment carries
almost identical bias in weeks 1–4 (−6.99) and weeks 5+ (−6.87), and there are
3.3 times as many rows in the second window, so roughly **77% of the role-change
error mass sits in weeks where the season boundary is irrelevant.**

**The answer is week-to-week.** Not because the offseason does not move roles —
it moves them about as much — but because the model already handles the boundary
near-optimally, and because three quarters of the error arrives in weeks when
last season is no longer the question. Combined with the news finding above, that
is a consistent picture: in-season role change is real, large, mostly
unannounced, and the largest open problem in this layer.

## Pre-game news: the injury report and the depth chart (2026-08-27)

The role-change finding above points at a leading indicator, and two cached feeds
carry one. Both were checked for timing before anything was built with them: the
injury report lands a median **28 hours before kickoff**, 99% of it at least 2.7
hours before, and 0.18% after gameday (Thursday and Monday games crossing a
timezone). The depth chart is placed by the loader against the next
regular-season game to be played. Both are legitimately available at decision
time. Coverage is stable across all ten seasons — about 8% of panel rows carry an
injury entry, 79–97% a depth rank.

### The ceiling measurement said the opposite of what was expected

`scripts/measure_role_signal.py` asks whether these feeds flag role change
*before* it happens, deliberately run before wiring anything in. They barely do:

| signal | % of rows | P(role grows) | lift | recall |
|---|---:|---:|---:|---:|
| someone ahead of him is out, and he is healthy | 5.7% | 0.162 | 1.89 | 0.095 |
| someone ahead of him is out | 6.3% | 0.150 | 1.75 | 0.097 |
| promoted on the depth chart | 3.5% | 0.145 | 1.74 | 0.059 |
| promoted last week (strictly lagged) | 3.2% | 0.129 | 1.55 | 0.050 |
| he is questionable or worse | 9.9% | 0.035 | **0.40** | 0.040 |

Lift of 1.9 on 5.7% of rows sounds useful until the recall column is read.
**Only 10–18% of role growth is flagged by anything these feeds contain**, across
all three folds. The intuition that a role change is knowable in advance as
"injuries plus news" holds for roughly one promotion in eight. The rest are
committee shifts, coaching decisions, game script, and — this is the honest
caveat on the −7.4 figure — a good deal of one-week variance, because a segment
defined by realised share exceeding lagged share is selecting on a positive
usage residual and will show a negative points residual mechanically. The
catch-up profile is the trustworthy statistic precisely because it follows the
same players forward: the genuinely persistent part of a promotion is the −1.98
at week 1, not the −7.41 at week 0.

### But the injury report is close to deterministic about availability

The last row of that table is the interesting one, and it points the other way.
A player on the report is *less* likely to see his role grow, and the model
over-projects him by +4.0. Read directly:

| report status | n | play rate | mean points |
|---|---:|---:|---:|
| **Out** | 2,218 | **0.000** | 0.01 |
| **Doubtful** | 384 | **0.010** | 0.07 |
| Questionable | 3,164 | 0.665 | 6.33 |
| not listed | 48,678 | 0.769 | 7.75 |
| did not practice | 3,251 | 0.180 | 1.87 |

Out means out, on 2,218 rows, without exception. The model was guessing 77% from
appearance history on rows where the answer had already been published. This is
the "did not play" bucket — 20–22% of all error at a bias of +3.9 — and most of
it was never a modelling problem at all. It was a missing feed.

### Result: the largest single gain since recency

The injury columns go to the availability half and the depth columns to the
magnitude half, which is the split the ceiling measurement argues for: the report
is a statement about whether a player takes the field, the depth chart about how
much work he gets once he does.

Relevant population, pooled over three holdouts:

| arm | MAE | CRPS | ρ (within pos) | top-24 |
|---|---:|---:|---:|---:|
| + context + ADP, per position | 5.010 | 3.377 | 0.588 | 0.113 |
| **+ pre-game news** | **4.609** | **3.140** | **0.668** | **0.116** |
| change | **−8.0%** | **−7.0%** | **+13.7%** | +2.5% |

Improved on **3/3 folds on every metric**. In weeks 1–4 specifically: MAE −6.7%,
CRPS −6.5%, within-position ρ +13.7%, and top-24 hit rate **+11.3%**
(0.2745 → 0.3055).

Against the ADP baseline on the drafted pool, the count of metrics that fail to
clear the bar drops from three to **one**: early-season top-24 hit rate, now
−5.9% against the board where it was −15.3%. Everything else wins, pooled CRPS by
29.7%.

A practical consequence worth noting: the p10 floors stop collapsing to zero.
Before, any player with more than a 10% chance of sitting had a tenth percentile
of exactly zero, which made the floor useless for separating two similar
projections. With the report resolving the certain cases, the floor becomes real
information again.

### On preseason hype specifically

It is already in, and that is why there is not much left. ADP *is* the aggregated
preseason news — it is what drafters believed after digesting camp reports,
depth-chart chatter and beat coverage — and it entered the model in the previous
round, cutting the early-week rest-of-season deficit from −3.05% to −0.51% MAE.
What a separate hype feed would add over the consensus that already prices it is
an open question this measurement cannot answer, and the remaining early-season
gap is now one ordering metric at −5.9%. That is a small target to spend a new
data source on.

## Every input, and what is deliberately absent (2026-08-28)

### What the shipped next-week model reads

59 magnitude features and 29 availability features, plus four position dummies
and a missing-history flag on each. (The count stood at 43/13 when this section
was first written, before draft capital and tracking efficiency were added.)

| group | n | columns |
|---|---:|---|
| player scoring history | 3 | career mean, recency mean, recency mean given played |
| player usage history | 5 | targets, carries, pass attempts, target share, carry share — all recency-weighted |
| **last observation** | 4 | points, target share, carry share, snap share — the previous played week |
| snaps | 2 | recency-weighted snap share, and its step |
| availability history | 4 | play rate, recency play rate, weeks since played, prior games |
| team context | 4 | plays, points, pass attempts, carries — recency-weighted |
| opponent, coarse | 1 | points allowed to this position |
| opponent, phase-split | 7 | carries/yards/YPC/EPA allowed rushing; targets/yards/EPA allowed receiving |
| game script | 7 | spread, total, implied team and opponent totals, own defence EPA and yards allowed |
| draft board | 4 | log rank, drafted flag, both interacted with an early-season indicator |
| pre-game news → availability | 3 | injury report status, practice status, ruled-out flag |
| pre-game news → magnitude | 6 | injury status and practice, depth rank, depth promotion, someone-ahead-out, position group out |
| pedigree | 4 | draft round, log overall pick, undrafted flag, years of experience |
| tracking efficiency | 12 | rush yards and rush % over expected, rushing efficiency; YAC above expectation, expected YAC, separation; completion % over expected, expected completion %, time to throw; three tracked flags |

### Where each of those comes from

| source | what it supplies | span |
|---|---|---|
| nflverse `player_stats` (weekly) | points, all volume, EPA; team aggregates | 1999–2025 |
| nflverse `rosters_weekly` | panel membership and the honest zeros; the PFR↔GSIS bridge; years of experience | 2011–2025 |
| nflverse `snap_counts` | snap share (92.9% of played rows) | 2014–2025 |
| nflverse `injuries` | game-status and practice reports, median 28h pre-kickoff | 2009–2025 |
| nflverse `depth_charts` | weekly positional rank | 2016–2025 |
| nflverse `schedules` | closing spread and total; actual scores for diagnostics only | all |
| nflverse `draft_picks` | round and overall pick, 99.6% GSIS coverage | all |
| nflverse `nextgen_stats` | tracking efficiency, above a volume threshold (52% of RB, 38% of WR/TE, 92% of QB played rows) | 2016–2025 |
| FantasyPros ADP CSVs (`ADP/`) | the draft board and the baseline curve | 2015–2026 |

Everything except the closing line, the two pre-game reports and the draft board
is a transform of something that already happened on a field.

### What is available and not used

Participation and personnel packages, FTN charting, contracts, combine
athleticism, Vegas *season win totals*, weather, and coaching continuity.
Red-zone usage was built from play-by-play and measured at season
level: it does not help there (+0.11% to +0.49% across four targets) because the
trait does not persist, which is a fact about the trait rather than the cadence.
Three sources previously on this list have since been joined and measured, and
the results are below: `ff_opportunity` (expected points) is a null,
play-by-play's `pass_oe` (pass rate over expected) is a null with a ceiling
argument behind it, and Next Gen Stats **ships**.

### Would the season pipeline's projections help as an input?

Probably not much, and the cheapest version of the test says so.

The projection itself is expensive: the posteriors are gitignored build outputs,
so using it means refitting three folds of a sampler that takes 700–1400s each.
And it is a **preseason** quantity competing with ADP, which this layer already
reads — while that pipeline's own headline finding is that it *loses* to the
draft board on drafted players. Importing it as a level would be importing
something weaker than a signal already present.

There is one measured reason to think it still carries information: the season
work found that about **41% of that model's disagreement with the board is
correct** (a +0.409 slope), which is exactly why blending it with ADP worked. So
the question is real rather than rhetorical.

The cheap proxy is to import the season pipeline's distinctive *inputs* instead
of its output. Two of them — draft capital and career stage — join cleanly and
are genuinely exogenous: nothing else in this feature set says what a club
thought of a player before he took a snap. The marginal relationship is strong,
9.37 points a game for first-rounders against 2.04 for the undrafted.

Conditioned on everything else, it is **−0.19% CRPS and −0.13% MAE**, improving
on 3/3 folds but below the 0.25% materiality floor this package uses. Consistent
in sign, immaterial in size. Kept, because it is free and points the right way,
and reported as a tie rather than a win.

That is evidence — not proof — that the information behind a season projection is
already priced here by ADP and by usage history. It does not settle the question:
the projection is a nonlinear combination of far more than these two columns
(coaching continuity, combine athleticism, win totals, cross-season pathways,
team-level structure), and a null on two inputs does not rule out the whole. What
it does establish is that the *obvious* channel — pedigree the box scores cannot
see — is not where the remaining error lives.

## Expected points: the better column that adds nothing (2026-08-29)

`ff_opportunity` prices every play from its context — down, distance, field
position, air yards — and reports what a week's opportunities *should* have been
worth. This layer had a specific reason to want it that the season layer did not.
The decay on player history was selected at a **one-game half-life**, so the
single most heavily weighted input to any projection is what happened last
Sunday — and last Sunday's points are substantially touchdown variance. Expected
points are the same signal with that variance removed, which is exactly the
substitution a one-game window should reward and a season average should not.

The season layer already tested this and it failed there: on 2,047 consecutive
player-season pairs, prior actual points per game beat prior expected points at
every position. That is a statement about a *year-long* average, where a season
of touchdowns is most of the way to its own expectation. Over one week it is not,
which is why the question was worth re-asking rather than inheriting.

### It joins well and it is the stronger column

The feed covers **98.5%** of relevant played rows and 90.3% of played rows over
the whole panel, 2016–2025 — better coverage than snap counts. And as a single
predictor it is clearly better than the thing it would replace:

| lagged column | correlation with next week's points |
|---|---:|
| actual points, last played week | 0.3112 |
| **expected points, last played week** | **0.3798** |

A 22% stronger raw signal, in the column the half-life sweep says the model leans
on hardest. On that basis it should have been the largest gain since the injury
report.

### In the full model it is a null

Walk-forward 2023/2024/2025, relevant population, n = 13,859, identical rows:

| rung | MAE | CRPS | ρ | top-24 |
|---|---:|---:|---:|---:|
| hurdle+everything+pedigree/position | 4.6866 | **3.1768** | 0.6817 | 0.1238 |
| hurdle+everything+**xfp**/position | 4.6883 | 3.1775 | 0.6817 | 0.1249 |

+0.04% MAE and +0.02% CRPS — marginally *worse* than the rung it sits on, and far
inside the 0.25% materiality floor. Not a small win. Nothing at all.

### Why, measured rather than asserted

The two results only look contradictory if expected points are treated as new
information. They are not. Regressing lagged expected points on the nine usage
features the model already reads — targets, carries, pass attempts, target
share, carry share, snap share — recovers most of the column. From
`scripts/probe_over_expected.py`, on relevant rows with all four columns
present (n = 42,262):

| lagged column | explained by usage (R²) | corr with next week | corr, usage removed |
|---|---:|---:|---:|
| expected points, last week | 0.805 | 0.365 | **0.071** |
| actual points, last week | 0.458 | 0.331 | **0.126** |
| expected points, recency | 0.886 | 0.415 | **0.137** |
| actual points, recency | 0.416 | 0.543 | **0.393** |

Read the last column, not the middle one. Expected points beat actual points on
the marginal correlation and lose on the residual one, in both the last-week and
the recency-weighted pairing, and lose by a wide margin in the second. Expected
points *are* a weighted sum of opportunities, and this model reads the
opportunities directly — 80% to 89% of the column is already in the design
before it is added. What is left correlates **less** with next week than the
residual of raw actual points does: the touchdown noise that expected points
strips out was carrying signal of its own, presumably about scoring role near
the goal line, and stripping it is a loss rather than a cleanup.

The same ordering holds on the narrower set of rows the ladder actually scores
(2023–25 holdouts, n = 13,859): R² = 0.815, residual correlation 0.044 for
expected points against 0.072 for actual. Different population, same verdict,
which is why the ladder shows nothing.

So the strong marginal correlation is real and the model's indifference to it is
also real, and the reconciliation is redundancy rather than either measurement
being wrong. This is the season document's method note reappearing: *a probe run
against a subset of the model's inputs measures the subset, not the model.* The
0.3798 is what expected points beat when the comparison is one column against
one column. The full feature set is not one column.

### What ships

Nothing. `src/ffmodel/weekly/expected.py` and the `EXPECTED_FEATURES` group stay
in the tree so the rung stays measurable and so the next person asking this
question gets the answer in a ladder run instead of a week of work, but
`scripts/project_week.py` does not turn it on.

The transferable claim is narrower than "expected points do not help" and worth
stating carefully: **an efficiency-stripped restatement of usage cannot beat
usage in a model that already reads usage.**

The tempting wider claim — that the other "over expected" families fail for the
same reason — was written here and then tested, and it is wrong. Next Gen Stats'
tracking metrics are barely explained by usage at all (R² 0.005–0.095 against
0.805 here) and they do move a metric. The dividing line is not the name of the
family but whether the column restates opportunity or measures a player. See
*Next Gen Stats: the first "over expected" family that is not usage* below.

## Next Gen Stats: the first "over expected" family that is not usage (2026-08-29)

Expected points and pass rate over expected both failed, and the tidy conclusion
would have been that the whole "over expected" idea is a restatement of things
the model already reads. **That conclusion is wrong, and this is the measurement
that shows it.**

Tracking data is different in kind. Separation, yards after catch above
expectation and completion percentage over expected describe an athlete and a
scheme, and a box score does not contain them. Regressed on the same nine usage
features that swallowed 80% of expected points, the tracking columns give up
almost nothing:

| lagged column | explained by usage (R²) | corr with next week | corr, usage removed |
|---|---:|---:|---:|
| completion % over expected (QB) | 0.020 | 0.160 | **0.135** |
| rush yards over expected /att (RB) | 0.032 | 0.060 | **0.060** |
| YAC above expectation (WR/TE) | 0.009 | 0.045 | **0.042** |
| *for contrast:* expected fantasy points | 0.805 | 0.365 | 0.071 |

The residual correlations are modest, but they are almost the whole of the
marginal correlation — these columns are nearly orthogonal to everything the
model has. That is the opposite of the expected-points result and the reason
this one earned a ladder run.

### Coverage is the real problem, and it is not random

The league publishes tracking summaries only above a volume threshold, so the
missingness is selection on precisely the variable that drives fantasy points:

| feed | covered rows | uncovered rows |
|---|---|---|
| rushing | 16.0 carries/wk, 15.3 pts (n=4,548) | 4.5 carries, 6.1 pts (n=4,197) |
| receiving | 7.9 targets/wk, 13.7 pts (n=10,677) | 2.7 targets, 7.7 pts (n=17,515) |
| passing | 33.3 attempts/wk, 16.6 pts (n=4,583) | 5.1 attempts, 2.7 pts (n=391) |

Half of running back weeks and nearly two thirds of receiver weeks are simply not
measured, and they are the low-volume half. The design fills a missing feature
with the training median, which reads here as "league-average efficiency for a
player nobody tracked" — defensible precisely *because* the volume that caused
the missingness is already a feature. A `_tracked` flag is carried alongside so
the fit can price the fill.

### The result is a trade, not a win and not a null

Walk-forward 2023/2024/2025, relevant population, n = 13,859:

| rung | MAE | CRPS | ρ | top-24 |
|---|---:|---:|---:|---:|
| hurdle+everything+pedigree/position | **4.6866** | **3.1768** | 0.6817 | 0.1238 |
| hurdle+everything+**ngs**/position | 4.7016 | 3.1830 | 0.6804 | **0.1403** |

MAE +0.32% and CRPS +0.20% *worse*, on 3/3 folds each. Top-24 hit rate **+13.3%
better**, on 3/3 folds and on 3/3 phases of the season (+4.8% early, +4.8% mid,
+6.5% late). Both directions are consistent, so both are real. Within-position
Spearman is flat-to-slightly-worse, which locates the effect precisely: this does
not order the position better in general, it identifies the top of it better.

### The control that makes the attribution honest

Being published by the league *is* a volume threshold, so the `_tracked` flags
alone could have produced the whole effect — and volume is something the model
could learn from carries without touching this feed at all. The ladder carries a
flags-only rung to settle it:

| rung | MAE | CRPS | top-24 |
|---|---:|---:|---:|
| pedigree (reference) | 4.6866 | 3.1768 | 0.1238 |
| **+ tracking flags only** | 4.6974 | 3.1800 | 0.1238 |
| + full tracking metrics | 4.7016 | 3.1830 | **0.1403** |

The flags on their own reproduce the reference top-24 hit rate to four decimals
and carry most of the accuracy cost. **The top-24 gain is the efficiency metrics
themselves**, not the fact of being measured.

### It ships, and here is the argument

The standing requirement for this layer is that every metric beat the naive draft
board. One metric has never met it: top-24 hit rate in weeks 1–4 on the drafted
pool, where a consensus board built by people concentrating on the players who go
early stays sharper than a model reading box scores. Tracking efficiency is the
first thing tried that moves it:

| next-week model | early top-24 vs ADP |
|---|---:|
| without tracking | 0.3098 vs 0.3328 — **−6.00%** |
| **with tracking** | 0.3240 vs 0.3328 — **−2.64%** |

More than halved, with every other metric on every other population still a win
by 8% to 200%. `scripts/project_week.py` turns it on.

The cost is stated rather than buried: this is the first rung in either ladder
accepted while making MAE and CRPS *worse*, and it is accepted because the
next-week model exists to answer start/sit, which is a top-k question, and
because it is the only lever found for the one bar the layer does not clear.
Anyone who wants the sharper point estimate instead sets `use_charting=False` and
gets 0.3% back on both loss metrics.

### What this changes about the earlier conclusions

The generalisation written after the expected-points null — that over-expected
metrics should all fail the same way — was too broad, and this is the correction.
The right rule is narrower: **a metric fails here when it is a restatement of
usage, not because it is "over expected".** Expected points are a weighted sum of
opportunities (R² 0.805 against usage) and had nothing left. Tracking metrics are
a measurement of a player (R² 0.005–0.095) and had something. The test is the
residual correlation in `scripts/probe_over_expected.py`, not the name of the
family.

## Pass rate over expected: right idea, no room to move (2026-08-29)

Expected points failed because they restate usage the model already reads. Pass
rate over expected is the one column in the "over expected" family that is not a
restatement of anything in the panel, which is why it got its own build rather
than inheriting that verdict.

The argument is specific. The panel carries a team's recent pass attempts, and
that column is two things at once: a team that threw 45 times last week either
likes throwing or was down three scores. Only the first carries to next Sunday.
nflverse prices every snap's pass probability from down, distance, field
position, score and clock; the residual, averaged over a team-week, is
play-calling identity with the game state divided out. `src/ffmodel/weekly/tendency.py`.

### At team level it is exactly what it claims to be

Over 4,883 team-weeks, 2016–2025, full coverage — every team-week in the panel
has a priced play in it:

| quantity | week-to-week correlation with itself |
|---|---:|
| team pass attempts | 0.244 |
| **pass rate over expected** | **0.400** |

The tendency is 64% more persistent than the counts it corrects. And it adds on
top of what the model already has, predicting next week's team volume:

| predictors of next week's team… | pass attempts (R²) | pass share (R²) |
|---|---:|---:|
| prior pass attempts + spread + total | 0.077 | 0.076 |
| **+ prior PROE** | **0.090** | **0.103** |
| + prior PROE + prior expected pass rate | 0.093 | 0.105 |

A 37% relative gain on pass share. On its own terms the column works.

### At player level it is nothing

Walk-forward 2023/2024/2025, relevant population, n = 13,859:

| rung | MAE | CRPS | ρ | top-24 |
|---|---:|---:|---:|---:|
| hurdle+everything+pedigree/position | 4.6866 | **3.1768** | 0.6817 | 0.1238 |
| hurdle+everything+**proe**/position | **4.6853** | 3.1781 | 0.6816 | 0.1294 |

MAE −0.03%, CRPS +0.04%. Both are two orders of magnitude inside the 0.25%
materiality floor and they disagree in sign, which is what a null looks like.
Top-24 hit rate appears to gain 4.5% pooled, and that is noise rather than
signal: split by phase of season it is +0.2% early, **−4.0% mid**, +7.8% late. A
metric that swings both ways across windows on 3,139–6,268 rows is not reporting
an effect.

### The ceiling says the channel was always too narrow

Rather than assert dilution, measure the whole channel. Give the model perfect
foreknowledge of the quantity PROE is trying to predict — this week's *actual*
team pass and rush attempts, which no model can have — and see what an oracle
would be worth (n = 37,346 relevant rows):

| magnitude design | R² on weekly points |
|---|---:|
| the model's 47 magnitude features | 0.3702 |
| + lagged PROE and expected pass rate | 0.3702 |
| **+ this week's actual team pass and rush attempts** | **0.3862** |

Perfect knowledge of team volume is worth **+0.016 R²**, total. PROE captures
about 1.4 points of team-volume variance beyond what the model has, so the share
of that oracle it could ever claim is on the order of a hundredth of it. The
feature is not diluted on its way to the player; there was never enough there.

This is the more useful half of the finding, because it retires a whole family of
ideas rather than one column. Team play-calling, pace, and pass-rate features are
a standing suggestion in fantasy modelling, and the oracle bound says the entire
category is worth 1.6 points of R² *before* anyone has to predict it. An
individual player's week is dominated by his share of the team, not the team's
volume, and his share is already read directly.

### What ships

Nothing, again, and for a different reason than expected points: not redundancy
but a ceiling. `src/ffmodel/weekly/tendency.py` and the ladder rung stay so the
measurement is reproducible; `scripts/project_week.py` does not turn them on.

## The panel, and why it starts in 2016

A stat feed contains rows for players who recorded something. Modelling on those
rows answers "how many points did he score, given he scored", which is not the
question a lineup poses — the week a starter is inactive is the outcome, and it
is the outcome that costs the most. So the panel is keyed on the **roster**: one
row per rostered skill player per team-week, carrying the stat line if there is
one and an honest zero if there is not.

Bye weeks are dropped rather than zeroed. A team-week with no game produces no
lines for anybody, which is indistinguishable from a roster full of inactives
unless the schedule is consulted.

Weekly rosters exist upstream from 2011 and **are not usable before 2016**:

| seasons | rows/season | play rate | RES play rate |
|---|---:|---:|---:|
| 2011–2015 | ~6,050 | 0.69 | **0.65–0.73** |
| 2016–2025 | ~9,600 | 0.54–0.59 | ~0.00 |

A player on injured reserve who records a stat line 70% of the time is not on
injured reserve. The panel is therefore **2016–2025, 98,225 rows**, 56.7% of them
weeks a player actually appeared. Week-`w` roster status is recorded for
diagnostics and is never a feature: whether a player was declared inactive is the
thing being predicted.

Every number below is reported on the **relevant** population — recency-weighted
average over weeks played of at least 4 points, and at least 4 prior appearances,
both read from lagged columns only. That is 49.1% of the panel. The season layer
learned the cost of not doing this: scoring on a fringe-heavy mixture flatters a
model that is good at forecasting zero.

## Leakage

On a seventeen-game series an expanding mean that forgets to shift includes the
week being predicted. The resulting model validates beautifully, cannot be used,
and nothing in the metrics says so. The lag is therefore structural: every
history column is produced by one function that applies its statistic and then
shifts it by one row within the group.

`tests/test_weekly_features.py` pins it the only way that demonstrates it —
perturb an outcome by 1000 points, assert no feature at or before that week
moves, **and assert at least one feature after it does**. A feature layer that
ignored history entirely would pass the first half alone. A second test pins the
features invariant to row order, which a grouped expanding statistic does not
give for free. A third does the same for the defence-allowed columns.

The market lines are the one exception and are deliberately unlagged: a closing
spread is published before kickoff and is legitimately known at decision time.

## Model 1: next week

Relevant population, pooled over three holdouts, n = 15,320, 800 draws.

| rung | MAE | RMSE | CRPS | cov80 | ρ (within pos) |
|---|---:|---:|---:|---:|---:|
| ADP curve | 6.046 | 7.703 | 4.468 | 0.458 | 0.405 |
| position climatology | 6.564 | 8.851 | 4.855 | 0.811 | 0.003 |
| career mean | 5.801 | 7.552 | 3.944 | 0.779 | 0.409 |
| recency-weighted mean | 5.173 | 6.905 | 3.493 | 0.792 | 0.570 |
| hurdle | 5.091 | 6.872 | 3.409 | 0.888 | 0.582 |
| + team & points-allowed matchup | 5.072 | 6.863 | 3.403 | 0.888 | 0.583 |
| + game script & phase defence | 5.049 | 6.834 | 3.391 | 0.887 | 0.587 |
| + per-position fitting | 5.031 | 6.829 | 3.385 | 0.885 | 0.587 |
| **+ ADP (ships)** | **5.010** | **6.815** | **3.377** | 0.884 | 0.588 |

Each step against the one above it, on CRPS:

| step | ΔCRPS |
|---|---:|
| climatology → career mean | −18.75% |
| career mean → **recency** | **−11.45%** |
| recency → **hurdle** | **−2.41%** |
| hurdle → team + points-allowed matchup | −0.16% |
| → **game script & phase defence** | **−0.36%** |
| → **per-position fitting** | −0.18% |
| → ADP | −0.24% |

**Recency is still the result.** A four-game half-life on a player's own history
is worth 11.5% of CRPS, more than every structural idea after it combined. The
decay was fixed a priori at four games rather than tuned, so no holdout was spent
on it.

### Game script and opponent: real, and small

The first attempt at matchup — points allowed by the opponent to this position,
recency-weighted — was worth 0.16% and exactly zero on one fold. That null did
not survive asking the question properly, but what replaced it is modest.

Three changes, together worth **−1.60% MAE and −0.93% CRPS** over the plain
hurdle:

**Phase-split defence, volume and efficiency apart.** "Good against the run" is
two claims. A defence can hold rushing yards down because it is hard to run on or
because nobody runs on it, and those point in opposite directions for a back's
workload. So carries and targets conceded are separate columns from yards per
carry and EPA per play conceded, and run is kept apart from pass — a defence is
routinely good at one and poor at the other, which a points-allowed aggregate
averages away.

**Game script from the closing line.** The spread says who is expected to lead,
and the implied totals — `total/2 ± spread/2` — say how much this offence is
expected to score and how much it will have to answer. A team favoured by ten
runs out the fourth quarter; a team down two scores throws. The sign convention
is load-bearing and tested: `spread_line` is quoted from the home team's
perspective and is re-signed per team, so positive always means *this* team is
favoured. Backwards, every game-script coefficient inverts and nothing about the
fit looks wrong.

**Per-position fitting, which is what lets the above be seen at all.** Several
script terms point in opposite directions by position: a favourite's running back
gets the fourth quarter and a favourite's receivers do not. A pooled slope
averages them toward zero and reports a real effect as a null. Fitting each
position its own design avoids the collinear-interaction pathology that sank the
season layer's ADP interactions — four separate designs, not one carrying a level
plus three deviations under a shared prior. A test asserts the RB and WR spread
slopes come out with opposite signs and the pooled slope between them.

So the mechanisms are real and they are measurable only in the right encoding.
They are also worth about a seventh of what recency is worth, and that proportion
is the honest headline. Own-defence quality (the "modern Bengals" effect — a team
that cannot get off the field throws to keep up) is in the script block and is not
separately identified from the implied opponent total, which measures the same
thing prospectively.

### The hurdle's coverage is the atom, not a defect

The hurdle reports 80% coverage of 0.888 against nominal 0.80. Central-interval
coverage is not a proper scoring rule and behaves badly on a distribution with a
point mass: if a player has a 12% chance of not playing, the 10th percentile *is*
zero and every positive outcome clears it from below.

`scripts/diagnose_weekly_calibration.py` separates the halves. Availability gets
a reliability table; magnitude is scored on the weeks he played using the
conditional predictive **without the atom** — scoring the unconditional
predictive against outcomes selected on having played would report a
correctly-sized zero mass as a downward bias, which is the mistake the diagnostic
was rewritten to avoid.

| holdout | availability gap (worst bucket) | magnitude cov80 | cov95 | PIT |
|---|---:|---:|---:|---|
| 2023 | −0.050 | 0.813 | 0.951 | flat |
| 2024 | −0.033 | 0.813 | 0.951 | flat |
| 2025 | +0.024 | 0.803 | 0.949 | flat |

Both halves are calibrated. The pooled 0.888 is the atom. The one standing
blemish is the largest availability bucket, which under-predicts the play rate by
about 2 points in all three folds — consistent in sign, small, recorded.

## Model 2: rest of season

Same folds and population. The games-remaining offset is the player's club's,
taken from the schedule at week `w`, so it cannot encode that he was about to be
cut. A player who leaves the league scores zero for the weeks he is gone and
those zeros are summed.

| estimator | MAE | CRPS | cov80 | cov95 | ρ | top-24 |
|---|---:|---:|---:|---:|---:|---:|
| ADP curve | 34.89 | 24.61 | 0.663 | 0.873 | 0.674 | 0.380 |
| direct total | 29.59 | 20.90 | 0.802 | 0.946 | 0.766 | 0.392 |
| direct total + phase | 29.60 | 20.89 | 0.802 | 0.945 | 0.766 | 0.396 |
| **+ ADP (ships, before blending)** | **29.17** | **20.56** | 0.789 | 0.945 | **0.770** | **0.433** |
| independent weeks | 29.51 | 22.48 | 0.564 | 0.733 | 0.764 | 0.365 |
| hierarchical | 29.51 | 22.18 | 0.589 | 0.763 | 0.764 | 0.365 |

### The argument for the hierarchy was right, and lost anyway

Simulating the remaining games and adding them up understates the total's
variance if the weeks are drawn independently: most of what is unknown about a
player's rest of season is unknown in all his weeks at once and does not average
out. So the hierarchical arm draws the player first — a latent availability rate
from a Beta whose concentration comes from how much realised play counts
over-disperse relative to Binomial, and a latent per-game level whose spread is
estimated by variance components from the covariance between two different weeks
of the same player — and then plays the games against that fixed player.

**That reasoning is confirmed**: the hierarchy beats independent weeks by 1.31%
CRPS on 3/3 folds and moves coverage the right way at both levels. **And it is
nowhere near enough**: 0.589 against nominal 0.80 is still severely
over-confident, and a plain ridge on the total beats it by 5.80% CRPS on 3/3
folds while covering 0.802.

The direct fit is trained on realised season totals, so it learns their spread
from data. The simulator assembles that spread from parts, and every part it gets
slightly wrong compounds across seventeen games — including the one it has no term
for: that a role can simply end. Its bias is +2.97 against the direct fit's −1.36.

This is the third time this package has run that comparison and got the same
answer, after the rank curve and the composition test. **Fit the thing you are
going to be scored on.**

Game script is deliberately absent from this response even though it helps the
weekly one. A spread is published for one game; this response spans up to
seventeen, and this week's line says nothing about week twelve's.

## What ships

`scripts/project_week.py`, fitted on seasons strictly before the one requested.

- **Next week** — the hurdle with team, matchup, phase defence, game script, ADP
  and the pre-game injury/depth feeds, fitted per position. Output carries `p_plays` alongside quantiles: a p10
  of zero means "he might not play", a different call from a low projection for
  someone certain to suit up.
- **Rest of season** — the direct total with phase and ADP, blended with the rank
  curve at a per-horizon weight. The weight is estimated inside the training
  window by holding out its most recent season, since a live projection has no
  later season to borrow from; on 2025 that gives 0.41 early, 0.80 mid, 1.00
  late, consistent with the walk-forward weights. `--no-blend` disables it.

Shipping the better-motivated model over the better-calibrated one would be
choosing the story over the evidence, which is why the simulator does not ship.

## What is not established

- **Three folds, two of them scorable for the blend.** 2023 has no earlier
  holdout to take a weight from.
- **The early-season ordering gap is real and unfixed.** Both remedies narrowed
  it and neither closed it. A model that reads the board still picks the top 24
  worse than the board does in weeks 1–4.
- **Snap counts are not in the panel.** They exist upstream from 2014 but join on
  name and PFR id rather than the gsis id everything else uses. Snap share is the
  most direct measure of role available and remains the largest known gap.
- **Role change is still mostly unpredicted.** The news feeds flag 10–18% of it.
  The remaining 82–90% is committee shifts, coaching decisions and game script,
  and nothing currently cached speaks to it.
- **The news feeds are only in the next-week model.** Adding them to the
  rest-of-season response is untested: a Friday game status is a statement about
  one game, and its value over a seventeen-week horizon is a different question
  that has not been measured.
- **Depth-chart coverage jumps in 2025** (0.79 to 0.97) with the upstream schema
  change. That is more coverage in the final holdout than in training, which
  could flatter 2025 slightly; the gain holds on 2023 and 2024 regardless.
- **Game script is measured through the closing line only.** No pace, no
  personnel-grouping tendency, no coverage-specific matchup. The line prices what
  the market expects, which is not the same as what the offence will do.
- **The panel is PPR only.** The scoring rules are parameterised and the panel
  builder takes a format, but nothing has been validated in standard or half-PPR.
