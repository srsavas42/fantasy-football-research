# Mean reversion, injury games, and the availability checkpoint (2026-08-26)

Five questions, asked of the pipeline end to end. Every number below was
measured on this checkout against nflverse 1999–2025 and is reproducible with
`scripts/measure_efficiency_reversion.py`,
`scripts/measure_post_injury_efficiency.py`,
`scripts/measure_history_depth.py` and
`scripts/measure_availability_signal.py`.

Short answers:

1. **Touchdown reversion is applied, but by a hand-set constant that is roughly
   half of what the data asks for.** Receiving and rushing touchdown rate ship a
   conditional mean equal to the lagged feature itself. The data wants a slope
   of 0.59 and 0.44 on that feature. The cost is a monotone bias across the
   board — the top quintile of last year's touchdown rate is projected about 5
   PPR points high at receiver and 8 high at running back, the bottom quintile
   the same distance low.
2. **Injury games do not contaminate the efficiency layer.** The per-touch
   decomposition plus the exposure-weighted shrinkage handles it. Measured, the
   efficiency prior's residual bias and its dispersion are both flat across
   prior-season availability. The one channel it does not close is playing
   hurt, which is untested rather than ruled out.
3. **A post-injury recovery curve is not a feature the model is missing, it is a
   shape the feature contract cannot express.** Efficiency is lag-1 throughout —
   there is no `prior2_*` anywhere in the package — so "worse in year one, better
   in year two" has nowhere to live. Measured, though, the year-one penalty is
   −1.3% of points per touch with a 95% interval of [−5.7%, +3.0%], and on a
   balanced panel the year-two rebound does not exist — its point estimate is
   negative. Most of the apparent rebound in the folk version is survivorship:
   only 22.9% of players who lose a season hold a real role two years later,
   against 51.3% of healthy peers. The real
   post-injury effect is on availability and role, not on efficiency.
4. **The pipeline is not uniform about how many prior seasons it reads, and the
   inconsistency is undocumented.** One layer — the snap model — carries a career
   EWMA and a one-season trend. Availability, the role allocators and every
   efficiency response read exactly one lagged season. Measured, the missing
   depth is worth **−2.19% availability MAE on 5 folds of 5** from a column
   (`prior_availability_3yr`) that is already built and populated in every frame
   and simply not listed in `AVAILABILITY_FEATURES`. For efficiency it is worth
   almost nothing.
5. **Prior-season injury and recovery time add essentially nothing** beyond
   `prior_availability`, which the layer already carries — this repo measured
   that expensively and inconclusively, and the mechanism is now visible.
   **Current** injury state is a different matter, and it is where the
   Charbonnet case actually breaks: not a missing feature, but a status code
   that means two different things in July and September, and no training data
   at a July cutoff at all.

---

## 0. The pipeline, end to end, and where each answer lives

`scripts/project_season.py` is the shipping path. It reads a cached
`SeasonAverageData` (team rows + player rows), fits
`SeasonAverageScoringPipeline` on every season before the target, predicts the
target season, and blends the result with a rank curve fitted on ADP. The 2026
file is 907 players.

Worth stating plainly, because it scales everything below: `BLEND_WEIGHT = 0.316`
is *the model's* share, and `blend_samples` is a draw-level mixture rather than a
weighted average — each posterior draw is taken from the model with probability
0.316 and from the ADP rank curve otherwise. The published `projection` column is
therefore about two-thirds draft board. `model_only` in the same file is the
unblended pipeline, and that is the column any finding here moves at full size.

```
preseason roster snapshot  ──►  AVAILABILITY  ──►  SNAP SHARE  ──►  ROLE ALLOCATION  ──►  EFFICIENCY  ──►  SIMULATOR  ──►  MARKET BLEND
 (nflverse wk-1 rosters,        Bernoulli        conditional        Multinomial            10 posterior      binomial /       draw-level
  or a Sleeper capture)         hurdle +         on activity        softmax over the       marginals,        multinomial      0.316 model,
                                Beta-Binomial    (games as the      active room, each      exposure-aware    draws at the     mixture with
                                games            exposure           team-season sums                        simulated        the ADP curve
                                                 offset)            to one                                  opportunities
```

Three properties matter for the questions asked:

- **Every layer is a rate, and exposure is carried separately.** Availability
  draws become the exposure offset for the snap model, the QB workload softmax
  and the target/carry allocators. A player projected for fewer games gets
  proportionally fewer opportunities, and his per-touch rates are untouched.
  This is why question 2's answer is mostly "no".
- **The scoring layer draws counts, not points.** `simulate_season_scoring`
  takes integer opportunities from the volume draws and a rate from the
  efficiency draws and draws `rng.binomial(carries, rush_td_rate)`. A biased
  rate therefore becomes a biased touchdown count one-for-one, with no
  compensating mechanism anywhere downstream. This is why question 1's answer
  matters more than its 2.3% share of points variance suggests.
- **The roster snapshot is the only place current injury state can enter.**
  `AVAILABILITY_FEATURES` contains `roster_active`, `roster_reserve` and
  `depth_rank` and nothing else about health. The eleven injury features exist
  but `SeasonAvailabilityModel.extra_features` is `()` in the shipping
  configuration. This is where question 3 lands.

The layers themselves are in good shape and the acceptance discipline around
them is unusually strong — the production fit records zero divergences and max
R-hat 1.005 across eighteen models, count conservation is asserted per draw,
and the docs record several negative results in more detail than most projects
record their positive ones. Nothing below is a claim that the architecture is
wrong. All three findings are about single constants and single columns.

## 1. Efficiency and touchdown reversion

### What the pipeline does

Reversion happens in exactly one place for most responses:
`features/season_efficiency.py` partially pools each observed season rate toward
the same-season position mean with a fixed pseudo-count.

```python
shrunk = (numerator + K * pooled_position_mean) / (denominator + K)
```

`K` comes from `EFFICIENCY_SPECS` — 120 targets for receiving touchdown rate,
120 carries for rushing, 200 attempts for passing, 40 targets for catch rate.
That gives the player's own history an effective weight of `den / (den + K)`.

Then `POSTERIOR_MEAN_MODE` in `models/efficiency_season_average.py` decides what
becomes the conditional mean:

| response | mean mode | share of points variance* |
|---|---|---:|
| rec_catch_rate | `prior` | 8.39% |
| pass_td_rate | `ridge` | 7.03% |
| rec_td_rate | **`prior`** | 2.60% |
| rush_td_rate | **`prior`** | 2.33% |
| pass_int_rate | `ridge` | 0.39% |
| fumble_lost_rate | `prior` | 0.05% |
| rec_yards_per_target | `posterior` (persistence ~ N(0.80, 0.25)) | — |
| pass_yards_per_attempt, rush_yards_per_carry, pass_completion_rate | `ridge` | — |

\* from the `CONCENTRATION_PRIOR_SIGMA` comment in the same module.

`prior` mode is an identity map. `_prior_signal` logit-links the lagged feature
and subtracts a centre; `_prior_mean` re-adds that centre and inverts the link.
The mean handed to the simulator **is** `prior_<response>` exactly. The
Beta-Binomial around it estimates a concentration, which moves the spread and
never the location.

So for the two touchdown responses, the pipeline's entire regression to the mean
is the choice of `K = 120`, and its implied persistence coefficient on the
shrunk feature is exactly **1.0**.

### What the data asks for

Walk-forward, one fold per season transition, fitting an intercept and a slope on
the pipeline's own `shrunk_*` column and scoring at realized next-season
exposure so the volume layer is held out of it:

| response | mode | effective persistence on the raw lagged rate | slope the data wants on the shipped feature | held-out event MAE | folds |
|---|---|---:|---:|---:|---|
| rec_td_rate | `prior` | 0.413 | **0.591** | **−3.15%** | 12/15 |
| rush_td_rate | `prior` | 0.572 | **0.441** | **−5.42%** | 12/18 |
| rec_catch_rate | `prior` | 0.671 | 0.795 | −2.38% | 13/15 |
| pass_td_rate | `ridge` (reference) | 0.676 | 0.493 | −11.95% | 4/4 |

The shipped policy asserts that column-four slope is 1.000. For catch rate the
assertion is nearly true — 0.795, and catch rate is genuinely sticky, so `K = 40`
happens to land close. For the two touchdown rates it is not: the pipeline keeps
about twice as much of last season as the following season repays.

The `pass_td_rate` row is a reference, not an indictment — that response's mean
is a fitted ridge, so the "shipped" arm there is the lagged feature alone rather
than what actually ships. It is included because it shows the same shape at the
response where the stakes are largest.

Every one of these clears the package's 0.25% materiality floor by an order of
magnitude.

### Where the error lands

Pooled MAE understates this badly, because the error is a location bias that
changes sign across the distribution. By quintile of the lagged feature, at
realized next-season exposure:

**Receiving touchdowns** (n = 2038, `prior` mode — this is the shipped policy)

| quintile | lagged rate | realized rate | projected TD | actual TD | gap | PPR points |
|---|---:|---:|---:|---:|---:|---:|
| Q1 low | 0.027 | 0.033 | 2.04 | 2.62 | −0.57 | **−3.4** |
| Q2 | 0.039 | 0.046 | 3.43 | 4.08 | −0.65 | −3.9 |
| Q3 | 0.046 | 0.046 | 4.29 | 4.36 | −0.08 | −0.5 |
| Q4 | 0.053 | 0.050 | 4.90 | 4.56 | +0.34 | +2.0 |
| Q5 high | 0.067 | 0.057 | 6.42 | 5.56 | +0.85 | **+5.1** |

**Rushing touchdowns** (n = 1155, `prior` mode)

| quintile | lagged rate | realized rate | projected TD | actual TD | gap | PPR points |
|---|---:|---:|---:|---:|---:|---:|
| Q1 low | 0.019 | 0.025 | 3.26 | 4.64 | −1.38 | **−8.3** |
| Q2 | 0.025 | 0.029 | 4.27 | 5.07 | −0.80 | −4.8 |
| Q3 | 0.029 | 0.028 | 5.20 | 5.20 | −0.00 | −0.0 |
| Q4 | 0.035 | 0.032 | 6.38 | 5.65 | +0.72 | +4.3 |
| Q5 high | 0.049 | 0.041 | 8.44 | 7.11 | +1.33 | **+8.0** |

A 13-point spread between the tails at running back, and it is systematic rather
than noise: the sign flips at Q3 in both tables, on 15 and 18 folds.

### Why the gate never caught it

Three reasons, and they compound.

**The alternative that was tested is not the alternative that is needed.**
`docs/efficiency-v2-validation.md` records receiving touchdown rate's flexible
posterior mean losing 0/3 folds and rushing touchdown rate 1/3. But that
challenger is the full posterior regression: fitted persistence *plus* the whole
base-feature block, the advanced-efficiency block and the projected-volume
covariate, all admitted at once on ~2,000 rows. There is no mode between "use
the feature raw" and "regress on everything". The two-parameter version — an
intercept and a slope on the feature the model already builds — was never on the
menu.

**The concentration absorbs the mean's error, which protects CRPS.** With a
biased mean, the Beta-Binomial fits a smaller concentration and the intervals
widen. Measured, the mean's own error is 4.9% of the residual variance for
receiving touchdowns and 11.6% for rushing — about 2.5% and 6.4% wider intervals
than a fitted mean would need. That is small enough that CRPS barely moves and
large enough to keep coverage looking healthy while the location is wrong. It is
consistent with the two most over-persistent responses being the two most
over-covered at the 80% level in the v2 table (0.899 and 0.917 against a nominal
0.800), though sparse discrete support explains part of that too.

**The stakes are small in pooled variance and large in ranking.** Receiving and
rushing touchdown rate are 2.6% and 2.3% of points variance. A pooled MAE gate on
total points is nearly blind to them. A drafter reading position ranks is not:
the players in Q5 are exactly the ones near the top of the board.

### One standing claim needs qualifying

`docs/xfp-assessment-2026-08.md` concludes that expected fantasy points adds
nothing because "the pipeline already regresses touchdown luck, through
`shrunk_pass_td_rate`, `shrunk_rec_td_rate` and `shrunk_rush_td_rate`". That is
half right. The pipeline regresses touchdown luck **partially**, by a constant,
and on the two responses xFP is usually invoked for it applies about half the
correction the data supports. The xFP result stands — expected points was not a
better route — but "the correction is already applied" is not why it failed.

### Recommendation

Add a fourth `POSTERIOR_MEAN_MODE`: the shrunk prior with a fitted intercept and
persistence slope, and nothing else. It is two parameters, it reuses the feature
already built, and it is a strict generalisation of `prior` (slope 1, intercept 0
recovers today's behaviour exactly), so the arm cannot lose by more than sampling
noise. Gate it on total points as usual, but score the tails explicitly — a
pooled metric is the instrument that missed this, so it is not the instrument to
confirm the fix with.

The same architectural pattern appears one layer up and deserves its own look
rather than a claim here: `SeasonRosterShareModel.fit` puts `log(role_prior)` in
the linear predictor as a fixed offset, so the lagged role also carries an
implicit coefficient of 1.0, with all the reversion done by the
`innovation_cap`. The code comment notes that cap binds on every fit against
measured dispersion of 1.43 and 2.00, so the same shape — a hand-set constant
standing in for a fitted coefficient — is doing the work there too. Role is far
stickier than touchdown rate (snap share persists at r ≈ 0.76), so the offset is
much more defensible; that is a reason to check it, not to assume it.

---

## 2. Injury games and per-touch efficiency

**The intuition in the question is right, and it is measurable.** Because every
efficiency response is a ratio of season totals over opportunities, a shortened
season shrinks the numerator and the denominator together. What is left is a
noisier estimate of the same quantity, not a biased one — and the exposure term
in the shrinkage is exactly the right correction for extra noise. The
availability layer, meanwhile, models games separately, so the missed games are
not double-counted either.

Residual of the realized next-season rate around the shipped `shrunk_*` prior,
split by what fraction of season Y the player was available for:

Bias is the mean residual as a percentage of the response's mean; sd is the raw
residual standard deviation. nflverse 1999–2025, 40+ targets or 60+ carries in
both seasons.

| prior-season availability | n (rec) | rec TD bias | rec TD resid sd | rec yds/tgt bias | rec catch bias | n (rush) | rush TD bias | rush YPC bias |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| under 55% | 75 | −8.6% | 0.0266 | +1.0% | −0.8% | 72 | −6.9% | +1.5% |
| 55–75% | 248 | +6.7% | 0.0303 | +1.5% | +1.6% | 184 | +5.6% | −0.3% |
| 75–90% | 815 | +2.7% | 0.0291 | −0.6% | +0.2% | 444 | −0.1% | −2.9% |
| 90%+ | 900 | −3.9% | 0.0269 | −1.7% | +0.6% | 455 | −2.3% | −1.4% |

No monotone gradient in any column, and the residual dispersion is flat — the
injury-shortened bucket is if anything the narrowest, not the widest. The
touchdown columns bounce ±7% on 72–75 rows, which is what noise at that cell
count looks like; the well-populated buckets sit within a couple of percent of
zero. The `den / (den + K)` weight is doing its job. Three things still deserve
flagging.

**The exposure floor admits seasons that carry almost no information.**
`efficiency_exposure_floor = 5` (promoted, and it improved efficiency MAE on all
three holdouts) means a five-target season becomes a training row and a lagged
feature. At five targets the shrinkage weight is 4%, so the prior is
approximately the position mean. That is harmless for the mean and it is the
right answer for uncertainty, but it does mean the feature cannot distinguish
"lost the year to injury" from "was genuinely a backup" — the two collapse to the
same value.

**Playing hurt is invisible.** A player who takes the field at 80% for fourteen
weeks is a full-availability row with depressed efficiency, and the pipeline
records that as talent and projects it forward. This is the one contamination
channel the per-touch decomposition does *not* close.
`docs/partial-games-2026-08.md` measured the adjacent version of this at the
snap-share layer and found cleaning made prediction worse, with a good argument:
attrition persists, so the contamination is signal. That argument is about
volume. For efficiency it does not obviously carry — "played hurt last year and
is healthy now" is a reverting condition, not a durable trait — and the injury
feature block already built in `features/season_injury.py` is scoped to the
availability layer only, so nothing tests it. It is untested rather than ruled
out.

**One caveat on the table above.** It requires 40+ targets in both seasons, so a
season genuinely lost to injury is excluded from it by construction. The
measurement says the efficiency prior is unbiased for players who kept playing;
it does not say anything about the five-target rows.

---

## 2b. Post-injury recovery curves

A separate question from the one above, and a sharper one: the claim is not that
an injury season's rate is mismeasured, it is that the *next* season's rate is
genuinely depressed and then recovers.

### The model cannot represent this, and the reason is structural

`lagged_efficiency_rows` shifts season `Y` onto `Y+1`. There is no `prior2_*`
column anywhere in the package. Every efficiency feature is exactly one season
deep, so the model has no way to know a player was hurt two years ago, and no way
to see his pre-injury level once the injury season becomes his prior. A two-year
recovery curve needs a two-year window; the feature contract has a one-year one.

What does exist is thinner than it looks. `prior_availability` sits in
`BASE_EFFICIENCY_FEATURES`, so five responses carry a linear "how much did he
play last year" term through their ridge mean, and `rec_yards_per_target` carries
it through its posterior regression. The four `prior`-mode responses —
receiving TD, rushing TD, catch rate, fumble rate — carry **no covariates at
all**, so for those the model does not know whether last season was sixteen games
or four. Nothing anywhere is conditioned on injury *type*.

### Measured, the penalty is small and the rebound is absent

Event study on a balanced panel: healthy baseline in `Y-1` (85%+ availability,
50+ opportunities), season `Y` classified lost (65% availability or less) or a
healthy control, and **both** `Y+1` and `Y+2` required to qualify, so the two
columns describe the same players. Outcome is PPR points per opportunity over
targets plus carries; the figure is a difference in differences against the
control, which nets out the ageing and mean-reversion drift both groups share.

| group | n | availability in Y | career year | baseline | year 1 | year 2 |
|---|---:|---:|---:|---:|---:|---:|
| healthy control | 554 | 0.926 | 4.83 | 1.384 | 1.359 | 1.341 |
| lost season | 116 | 0.469 | 4.67 | 1.310 | 1.268 | 1.231 |

| | difference in differences | 95% bootstrap | as % of baseline |
|---|---:|---|---:|
| year 1 after | −0.0179 | [−0.0781, +0.0416] | **−1.29%** [−5.65%, +3.00%] |
| year 2 after | −0.0364 | [−0.0965, +0.0272] | **−2.63%** [−6.97%, +1.96%] |
| the rebound (year 2 − year 1, same players) | −0.0185 | [−0.0701, +0.0346] | wrong sign |

Both halves of the claim come back weak. The year-one penalty is the right sign
and about a point and a half of per-touch efficiency, with an interval that
comfortably contains zero at n = 116. The rebound is not merely absent, its point
estimate is negative: on the players who survive to be measured twice, year two
is slightly *worse* than year one, not better. The two groups are matched on
career year (4.67 against 4.83), so this is not an ageing artifact, and the lost
group starts from a slightly lower baseline, which if anything should pull it up
rather than down.

### Where the recovery story actually comes from

| from a healthy baseline into season Y | n | back with a real role in year 1 | still there in year 2 |
|---|---:|---:|---:|
| healthy control | 1079 | 71.3% | 51.3% |
| lost season | 506 | **41.7%** | **22.9%** |

A player who loses a season is barely more than half as likely to hold a 50+
opportunity role two years later. Score year two on whoever is still playing and
you have selected for the players who recovered — which is exactly how "they
bounce back in year two" gets manufactured. The balanced panel above exists to
avoid that, and once you avoid it the rebound goes away.

This is the real answer to the question, and it is not an efficiency finding: the
post-injury effect is overwhelmingly about **availability and role**, both of
which the pipeline already models, rather than about per-touch efficiency, which
it does not condition on injury at all. That is a defensible place for the
package to be.

### "It depends on the injury" — the right cut, at sample sizes that cannot carry it

Year one only, which roughly doubles the lost-season sample, joined to the
nflverse injury reports from 2009 and bucketed by the repo's own
`_injury_body_group`:

| primary body group | n | difference vs control | 95% bootstrap |
|---|---:|---:|---|
| no severe report | 74 | −0.7% | [−6.7%, +5.1%] |
| lower_body | 69 | −2.2% | [−7.8%, +3.3%] |
| head_neck | 12 | −11.0% | [−22.5%, −0.4%] |
| upper_body | 9 | −9.9% | [−26.3%, +6.3%] |
| core | 7 | +3.2% | [−15.1%, +20.1%] |

Two things worth saying about this table and neither is the obvious one.

**The best-powered cell is the classic narrative, and it is the smallest effect.**
Lower body — knees, ankles, hamstrings, the injuries the "never the same again"
story is usually told about — has 69 rows and a −2.2% point estimate whose
interval spans zero.

**The two eye-catching rows have 12 and 9 rows.** `head_neck` at −11% is the only
interval that excludes zero, barely, on twelve observations, from the highest
baseline in the table, where regression to the mean is working in the same
direction. It is a hypothesis worth a real study, not a coefficient.

And 74 of the 219 lost seasons carry **no severe injury report at all**, which is
the honest limit of an availability proxy: a third of what this measurement calls
an injury is a benching, a holdout, or a lost job.

### Recommendation

**Do not add a post-injury efficiency feature on this evidence.** The effect is
small, the interval spans zero, and the shape the question is really asking about
— a two-year curve — would require extending the efficiency feature contract to
lag 2, which is a much larger change than the finding supports.

Two cheaper things are worth doing instead:

1. **Give the four `prior`-mode responses their `prior_availability` term.** They
   currently have no covariates whatsoever, so receiving and rushing touchdown
   rate do not know whether the prior season was sixteen games or four. That is
   a gap regardless of any recovery curve, and it falls out of the fitted-mean
   mode recommended in section 1 at no extra cost — the same two-parameter
   change that adds a slope can add this one column.
2. **Trust the availability layer with this, because that is where the effect
   is.** A 41.7% return-to-role rate against 71.3% is a large, well-identified
   effect on exactly the quantity the availability model exists to predict.
   Note the tension with section 3, though: the controlled table there shows
   prior-season injury adding nothing *conditional on prior availability*, and
   these two facts are consistent — the attrition is visible in the availability
   number itself, which is why a separate injury feature keeps coming back null.

---

## 2c. How many prior seasons does each layer actually read?

Not a uniform answer, and the inconsistency is not written down anywhere.

### What exists, and who reads it

`features/season_pathways.py` builds a real multi-season state for every player:
a career exponentially-weighted mean at `HISTORY_ALPHA = 0.50` — last season
half the weight, the one before a quarter — plus a one-season `_trend`
difference, over eight production inputs and five efficiency ones, grouped by
`player_key` so the history follows a player through a trade.
`build_season_average_data` attaches all of them to every frame. Verified on a
2018–2024 build: `prior_snap_share_3yr` is populated on 3,384 of 4,268 rows and
`prior_snap_share_trend` on 2,176, which is every row whose player has two
consecutive prior seasons.

Exactly one model reads any of it.

| layer | history depth in the shipping configuration |
|---|---|
| snap share | **career EWMA + trend** — `SeasonSnapShareModel.extra_features` defaults to `SNAP_HISTORY_FEATURES` |
| availability | one lagged season. `prior_availability_3yr` is in the frame and absent from `AVAILABILITY_FEATURES` |
| role allocators | one lagged season, as the `log(role_prior)` offset |
| efficiency, all ten responses | one lagged season. No `prior2_*` column exists anywhere in the package |
| team layer | one lagged season **plus** a per-team intercept pooled across every training season, and a linear era term |
| injury features (off) | three-year windows — `prior_injury_*_3yr` — the only other multi-year contract in the codebase |

Two structural notes worth separating from the feature question. The team layer's
franchise intercept is genuine multi-season information, but it is a constant per
team, not a trajectory. And `SeasonRosterShareModel._design` computes a
`player_idx` and returns it in the design dictionary, but `fit` never uses it —
there is no player-level random effect in the role allocators, so nothing pools a
player's own seasons there either.

### What the missing depth is worth

Walk-forward ridge, one fold per season, each fitted only on earlier ones.
Challengers are the cheapest available: the same lag-1 feature plus a career
EWMA, plus a trend, or plus an explicit second lag. Folds whose training rows
cannot support an arm are dropped from every arm rather than scored — a silently
NaN arm reads as a large improvement once the weighted mean skips it.

| layer / response | lag-1 MAE | best challenger | change | folds |
|---|---:|---|---:|---|
| **availability** | 0.25836 | + career EWMA | **−2.19%** | **5/5** |
| **snap share** | 0.15618 | + EWMA + trend | −1.22% | 3/4 |
| rush_yards_per_carry | 0.50520 | + career EWMA | −1.10% | 10/15 |
| rush_td_rate | 0.01274 | + explicit lag-2 | −0.87% | 11/15 |
| rec_td_rate | 0.02159 | + explicit lag-2 | −0.07% | 13/15 |
| rec_catch_rate | 0.05577 | + explicit lag-2 | −0.01% | 8/15 |
| rec_yards_per_target | 1.11858 | + explicit lag-2 | +0.00% | 2/15 |

### Reading it

**Availability is the finding.** −2.19% on five folds of five, from a column that
already exists, is already populated, is already leakage-safe, and is one entry
short of being read. It is nearly nine times the package's 0.25% materiality
floor and unanimous. It also fits what section 4 establishes independently:
availability is the layer with a documented resolution problem and the weakest
year-over-year signal, so it is exactly where a longer window should help most —
a single noisy season is a poor estimate of a durability trait, and averaging
three of them is a better one.

**The snap layer's −1.22% is a validation, not a proposal.** That arm already
ships. It is here because it demonstrates the mechanism works on this frame,
which is what makes the availability row worth acting on.

**Efficiency barely moves, and that is the honest answer to "would sequential
input help".** The two rushing responses clear the floor; the three receiving
ones do not, and receiving yards per target is worse with more history on 13 of
15 folds. Part of this is by construction: the lag-1 efficiency feature is
already `shrunk_*`, partially pooled toward the position mean, so it has absorbed
some of what an EWMA would do. Whatever the reason, the numbers are an order of
magnitude below what section 1 found by fixing the *slope* on the one season the
model already has — −3.15% and −5.42% there against −0.07% and −0.87% here. The
binding constraint on the efficiency layer is not how much history it reads. It
is what it does with the season it has.

### Recommendation

1. **Add `prior_availability_3yr` to `AVAILABILITY_FEATURES`.** One line, a
   column that already exists, −2.19% MAE unanimous on five folds. This is the
   cheapest well-evidenced change in this document and it should go to the
   scoring gate first.
2. **Consider `prior_rush_yards_per_carry` and `prior_rush_td_rate` career
   EWMAs**, but only after the fitted-mean change in section 1, and measured
   against it rather than against today's baseline. Both are RB-only responses
   on the smallest samples here, and a −1% arm measured against a mean that is
   itself mis-specified is not a clean read.
3. **Leave the receiving responses at lag 1.** More history makes them worse.
4. **Either use `player_idx` in the role allocators or drop it from the
   design.** A key the design computes, carries and never reads is either a
   missing player random effect or dead weight, and the code does not say which
   was intended.

---

## 3. Availability, PUP, and the Charbonnet number

The shipped 2026 projection gives Zach Charbonnet **14.636** projected games,
against a running-back mean of 13.32 and a whole-file maximum of 15.34. He is
projected *above* average availability. Nothing in the 907-row file is shaded for
injury: the lowest running back is 11.24 and the only single-digit numbers belong
to fourth-string rookie quarterbacks that the any-appearance hurdle is
suppressing.

### The half of the proposal that is already answered

Prior-season injuries plus recovery time is built. `features/season_injury.py`
ships eleven contracted features including `prior_injury_episode_count_3yr` and
an empirical-Bayes `current_injury_expected_recovery_weeks` pooled by body group
and severity — essentially the design in the question. It was screened at the
availability layer (CRPS −2.39% on three folds of three), passed the scoring gate
in-window, then went flat on the reserved 2025 season and halved across six
holdouts with the sign varying. `injury_availability_features` is off.
`docs/injury-availability-2026-08.md` has the full record.

The mechanism is now visible, and it explains the null. Split next-season active
weeks on whether the player finished season Y on a reserve list, **holding
season-Y active weeks fixed**:

| prior-season active weeks | finished healthy | finished on reserve | increment |
|---|---:|---:|---:|
| 4–8 | 6.09 | 7.75 | **+1.65** |
| 9–12 | 8.73 | 9.44 | +0.71 |
| 13–15 | 11.04 | 11.00 | −0.04 |
| 16+ | 12.61 | 13.32 | +0.71 |

Uncontrolled, the reserve group looks worse — 9.46 next-season active weeks
against 11.20. Controlled, the sign flips: a player who played eight weeks and then went
on IR is a *better* bet than one who played eight weeks without one, presumably
because landing on IR is evidence he was starting when healthy. The entire raw
difference is prior availability, which `AVAILABILITY_FEATURES` already carries.

A prior-injury feature has to fight for signal that `prior_availability` has
already taken. That is what six inconclusive holdouts look like from the inside.

### The half that is genuinely broken

The information that would move Charbonnet is not his history. It is that he is
on the PUP list right now. That path has three separate defects.

**(a) `PUP` is dropped from the roster snapshot, contrary to its own docstring.**
`preseason_roster_snapshot` documents that "active, inactive, reserve/PUP, and
exempt players remain because they are valid season-long availability outcomes",
but `ROSTER_STATUSES = {"ACT", "RES", "INA", "EXE"}` and nflverse emits a
distinct `PUP` code. Week-1 status against active weeks that season, model
positions, 2015–2024:

| Week-1 status | n | mean active weeks | never active | kept by the snapshot |
|---|---:|---:|---:|---|
| ACT | 4903 | 13.44 | 0.0% | yes |
| INA | 313 | 8.03 | 8.6% | yes |
| RES | 704 | **1.63** | 77.4% | yes |
| **PUP** | **29** | **2.90** | **51.7%** | **no** |
| SUS | 37 | 6.27 | 32.4% | no |
| DEV | 1296 | 2.78 | 43.8% | no |

Twenty-nine rows over ten seasons, so the effect is small — but the failure mode
is not "shaded down", it is "removed from the roster", which sends the player's
volume to the synthetic replacement bucket instead. The docstring says otherwise,
and the fix is one entry in a frozenset.

**(b) The status code means two different things in July and September, and the
live path collapses them.** `RES` at Week 1 is a post-cutdown reserve
designation carrying a mandatory multi-game absence — hence 1.63 mean active
weeks. `PUP` in late August is *active*/PUP: reversible any day up to cutdowns,
and most players come off it. `release/roster.py::_projection_roster_status` maps
every status that is not ACT/INA/EXE — `PUP` and `NFI` included — to `RES`, and
`roster_reserve` is then scored against a coefficient fitted entirely on Week-1
semantics. A live August snapshot naming a player PUP would therefore assert
something worth roughly eleven games, which is far too harsh for a designation
that is usually lifted before the opener.

Charbonnet's 14.6 says this did not fire — so whatever snapshot built
`.cache/ffmodel-2026`, it carried him as ACT. Worth confirming directly:

```python
import pandas as pd
rows = pd.read_pickle(".cache/ffmodel-2026/player_rows.pkl")
current = rows[rows.season == 2026]
print(current.roster_status.value_counts())
print(current[current.player_name.str.contains("Charbonnet", na=False)]
      [["player_name", "roster_status", "roster_active", "roster_reserve",
        "prior_availability"]])
```

If `roster_status` is `ACT` for everyone, the availability layer had no injury
information about any 2026 player at all, and the eleven injury features would
not have helped even if they were switched on.

**(c) There is no preseason training data, so this cannot be learned.**
nflverse weekly rosters contain no `PRE` rows — the earliest snapshot in the feed
is regular-season Week 1. The historical injury path has the same shape:
`_current_snapshot` reads `reports[week == 1]`, the official Week-1 game-status
report. Live projections serve from an archived August Sleeper snapshot through
the same columns. That is a train/serve mismatch of exactly the kind this branch
has already fixed twice, and no amount of feature engineering closes it, because
the training set has no August in it.

### What is worth doing, in order

1. **Add `PUP` and `NFI` to `ROSTER_STATUSES`**, matching what the docstring
   already claims. Small, correct, and independent of everything else.
2. **Stop collapsing preseason designations onto the Week-1 `RES` coefficient.**
   Either give a live snapshot its own column, or map August PUP to a separate
   level. What must not happen is the current arrangement, where a July
   designation silently inherits a September coefficient.
3. **Handle reserve/PUP as an arithmetic constraint, not a learned effect.** A
   player who is on reserve/PUP at Week 1 is ineligible for the first four games
   by rule, so his games ceiling is `team_games − 4`. A projection of 14.6 is not
   merely optimistic in that state, it is impossible. A constraint applied at
   the point the roster resolves is more honest and more robust than asking a
   logistic regression to rediscover a league rule from twenty-nine rows.
4. **Archive Sleeper snapshots weekly, starting now.** The reason this cannot be
   modelled is that nobody has been storing the preseason state. `load_players`
   already refuses to fabricate history, which is right and which also means the
   only way to get a preseason training set is to start keeping one. Three
   seasons of archived August snapshots make the question answerable in 2029;
   nothing else does.
5. **Leave `injury_availability_features` off.** The controlled table above says
   the prior-burden half of that block is competing with a feature that already
   has the signal. That is consistent with what six holdouts found and it is not
   worth re-litigating without new data.

### On the availability layer's spread generally

Separately from injuries, the layer does have headroom. On returning players with
a prior-season role, the ceiling on a history-only forecast's cross-player spread
is about 2.27 games pooled (`r = 0.365`, realized sd 6.22); the shipped 2026
projection spreads its per-player means 1.47. The populations are not identical,
so that is indicative rather than a bound — but it points the same way as
`docs/availability-resolution-2026-08.md`, which found the layer biased −6.7% on
drafted players and +5.3% on undrafted ones *in sample*. That is a resolution
problem, and it is the same problem whether or not anyone is hurt.

---

## Reproducing

Each script caches its frames on first run and is offline thereafter.

```bash
pip install -e ".[dev]"

# Sections 1 and 2: shrinkage, the fitted slope, quintile bias, and the
# injury-contamination check on the efficiency prior.
python scripts/measure_efficiency_reversion.py

# Section 2b: the post-injury event study, its survivorship table, and the
# injury body-group split.
python scripts/measure_post_injury_efficiency.py

# Section 3: the forecast ceiling, the controlled injury increment, and
# Week-1 roster status against active weeks.
python scripts/measure_availability_signal.py
```

No source file is changed by any of this. The scripts and this document are
additive, so nothing here alters a shipped projection until a gate says it
should. Fast suite at the time of writing: 470 passed, 11 skipped.
