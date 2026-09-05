# What a better projection is worth, in wins

*September 2026. Code: `src/ffmodel/league/`, `scripts/validate_league.py`,
`scripts/validate_weekly_baselines.py`. Numbers: `reports/league_baselines.json`,
`reports/weekly_baselines.json`.*

Two questions, answered in order, because the second one is only interesting if
the first one has a good answer.

1. Does the weekly model actually beat "average his last few weeks"?
2. If it does, how much is that worth to a fantasy manager?

They have very different answers.

---

## 1. The model against every naive average

The weekly ladder already carried two heuristic rungs, but neither answered the
sceptic's version of the question: *which* average, and would a differently-tuned
one have closed the gap on its own? So `validate_weekly_baselines.py` scores a
grid — last week's points, simple moving averages over 3/5/8 rostered weeks, the
same windows over *played* weeks only, exponentially weighted averages at
half-lives 1/2/4/8, and season-to-date — each wrapped in the same `HistoryMean`
estimator the ladder uses, so every baseline gets an honest predictive
distribution rather than a bare point estimate. Scoring a point forecast against
CRPS would have handed the model a win it did not earn.

Walk-forward, holdouts 2023/2024/2025, relevant population:

| estimator | CRPS | vs. best naive | within-position Spearman |
|---|---|---|---|
| shipped model | **3.1854** | **−11.6%** | 0.680 |
| ewma, half-life 2 | 3.6049 | — (best naive) | 0.575 |
| ewma, half-life 1 | 3.6299 | +0.7% | 0.572 |
| ewma, half-life 4 | 3.6712 | +1.8% | 0.549 |
| moving average, 3 weeks | 3.7105 | +2.9% | 0.544 |
| moving average, 5 weeks | 3.7262 | +3.4% | 0.531 |
| ewma, half-life 8 | 3.7684 | +4.5% | 0.514 |
| moving average, 8 weeks | 3.7780 | +4.8% | 0.512 |
| moving average, 3 weeks, played only | 3.8242 | +6.1% | 0.491 |
| moving average, 5 weeks, played only | 3.8790 | +7.6% | 0.467 |
| season-to-date | 3.9108 | +8.5% | 0.484 |
| moving average, 8 weeks, played only | 3.9624 | +9.9% | 0.440 |
| career mean | 4.0936 | +13.6% | 0.404 |
| last week | 4.4142 | +22.4% | 0.293 |

13,859 scored rows. Three things worth keeping:

- **The model wins by 11.6% CRPS, on 3 of 3 folds** (−11.4% / −10.8% / −12.8%
  on 2023 / 2024 / 2025). That is forty-six times the repo's 0.25% materiality
  floor, and it does not rest on one kind season. Within-position Spearman is
  0.680 against the best naive 0.575 — the *ordering* is 18% better, not just
  the calibration, and ordering is the half a start/sit decision actually uses.
- **Exponential beats flat, and the half-life is not load-bearing over a sensible
  range.** Half-lives 1, 2 and 4 land within 1.8% of each other; only half-life 8
  falls away, and by then it is behind a flat 3-week window anyway. The feature
  layer's choice of 1 is close enough to optimal that retuning it is not worth a
  run.
- **Averaging only the weeks he played is worse at every window** — +3.1% CRPS at
  3 weeks, +4.1% at 5, +4.9% at 8, the penalty growing with the window because a
  longer played-only window is a longer memory of a player at his healthiest. It
  is the number people quote — "he's averaging twelve when he suits up" — and it
  answers a different question than a lineup decision asks. Absence risk is part
  of next week's expectation, and dropping the zeros deletes exactly that.
- **Shorter flat windows beat longer ones** (3 > 5 > 8), which is the same
  finding the EWMA half-lives give: recency is most of what a naive average has.

## 2. The same policies, run as a season

CRPS is not a currency anyone is paid in. The league environment
(`ffmodel.league`) plays real seasons: 12 teams, ESPN-style roster (1/2/2/1/1/1/1
plus 6 bench), snake draft off the FantasyPros board, round-robin schedule,
weeks 1–14. Eleven opponents all play the same strategy — the board for three
weeks, then recent form — and the agent's seat is drawn at random with the draft
order and schedule. 3 seasons × 20 seeds = 60 episodes per policy.

| policy | wins (of 14) | points | rank | title rate |
|---|---|---|---|---|
| oracle (perfect start/sit) | **9.80** | 1645.0 | 2.13 | 61.7% |
| ewma | 6.88 | 1373.4 | 6.52 | 8.3% |
| adp-then-ewma (the field) | 6.85 | 1369.2 | 6.68 | 6.7% |
| adp all season | 6.30 | 1288.1 | 7.95 | 5.0% |

Read against the field's own strategy, paired on seed — every policy plays the
same draft order and the same schedule, so the difference is the policy and not
the luck:

| vs. the field | wins | SE | t |
|---|---|---|---|
| oracle | +2.95 | 0.19 | +15.6 |
| ewma | +0.03 | 0.08 | +0.4 |
| adp all season | −0.55 | 0.16 | −3.4 |

- **Never updating costs 0.55 wins and 81 points**, and it is a real effect
  (t = −3.4), not a seed. Playing the preseason board all season is a measurable
  mistake — but a smaller one than it feels like, because the board is not
  stupid, it is just stale.
- **Switching to recent form on week 1 instead of week 4 is worth +0.03 wins**
  (t = 0.4). Nothing. Which retires the splice week as a tuning knob: through the
  first three weeks the board and a one-or-two-game average are the same bet, and
  the environment's default of 4 could be 1 without changing any result here.
- **A perfect start/sit is worth +2.95 wins and +276 points**, and turns a 6.7%
  title rate into 61.7%.

That last row is the number the RL work is aimed at, so it is worth being precise
about what it is and is not.

### The ceiling is 2.95 wins, and it is a ceiling

The oracle sees each week's actual scores and starts the best legal lineup. It
still only wins 9.8 games of 14, not 14, because a perfect lineup cannot fix a
bad draft and cannot stop an opponent from scoring 140. **Everything an RL
start/sit agent could ever earn lives in a 2.95-win band**, and it will not get
all of it — the oracle's information is unobtainable by construction.

For scale: the entire gap between the field's strategy and never updating at all
— the difference between a manager who sets his lineup every week and one who
never logs in again after the draft — is 0.55 wins, 19% of the oracle band. So a
good agent is playing for something between a few hundredths and a few tenths of
a win, which brings us to the reason every number above is paired and averaged
over twenty seeds.

### Season noise is ±3 wins, and it swamps everything

| policy | win std | min | max |
|---|---|---|---|
| adp | 2.15 | 2 | 12 |
| adp-then-ewma | 2.09 | 2 | 11 |
| ewma | 2.03 | 2 | 10 |
| oracle | 1.76 | 6 | 13 |

A policy playing the field's *own* strategy — a seat with zero edge by
construction — finishes anywhere from 2 wins to 11. One season is roughly ±3
wins of pure luck from the draft slot and the schedule.

This is the single most important property of the environment for the RL work
that follows. One standard deviation of seed luck is 2.1 wins — most of the
entire 2.95-win oracle band, and four times the 0.55 wins that separate the field
from never logging in. **A single-season comparison of two policies is worth
nothing**, and an agent that "wins its league" in one simulated season has shown
nothing at all.

What rescues the comparison is pairing. Run both policies on the *same* seeds and
the draft slot and schedule cancel: the standard error on the ewma-vs-field
difference falls from 0.27 unpaired to 0.075 paired, which is what makes a
0.03-win null distinguishable from a 0.3-win edge at sixty episodes instead of
several thousand. Every claim about an agent has to be a paired average over
shared seeds. An unpaired comparison of two RL checkpoints will mostly measure
which one drew the better draft slots.

Note also that the oracle's spread is *narrower* (1.76 vs 2.09) but not close to
zero. Even perfect information leaves most of the variance in place, because most
of it is not in the agent's lineup at all — it is in who they drafted and who
they played.

---

## What this says about where to look next

- **Weekly projection quality is in good shape and is not the binding
  constraint.** The model beats every naive average by a wide margin, and the
  entire start/sit decision is worth at most 2.95 wins.
- **The draft plausibly matters more than the lineup, and this run does not
  prove it.** The oracle loses 4.2 games a season with a perfect card, so those
  losses are the roster it was handed plus weeks an opponent simply outscored it
  — but nothing here separates those two, and opponent variance alone could
  account for a good share. The way to settle it is the same environment with the
  lineup policy held fixed and the draft varied, which is also the measurement
  the draft RL model will need.
- **Waivers are unmeasured here.** The environment supports a claim per week and
  no baseline policy uses one, so the free-agent pool's value is currently
  counted as zero. It is not zero.
