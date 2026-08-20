# Cleaning partial games makes the role signal worse (2026-08-20)

## What the model actually targets

Worth stating first, because the proposal assumed otherwise: the pipeline does
not regress season points. It is already a rate-times-exposure decomposition.
`observed_availability = games / team_games` is what the availability model
fits, and the snap model fits `snap_share / observed_availability`, a
*conditional* share. Volume and efficiency are per-opportunity rates throughout.

Season totals are the **evaluation** target, not the regression target.

## Where the proposal is right

`games` counts a game as one whether the player took three snaps or seventy. So
a partial game inflates exposure and deflates the per-game rate, and the two
errors partly cancel in the season total while corrupting both components.

That is real, and it is not rare. Over 2014-2024, taking a partial game to be
one under half the player's own median snap share that season:

- **9.9%** of played games are partial
- **61.4%** of player-seasons contain at least one
- Cleaning them lifts mean snap share from 0.500 to 0.536, **+7.3%**

## Where it is wrong

Cleaning does not improve prediction. It makes it worse.

| | year-over-year r |
|---|---:|
| raw share → next raw share | **0.7626** |
| clean share → next raw share | 0.7492 |
| clean share → next *clean* share | 0.7401 |

Walk-forward 2019-2024, predicting next season's raw snap share:

| features | MAE | vs raw only |
|---|---:|---:|
| raw only | **0.13107** | — |
| clean only | 0.13355 | +1.90% |
| raw + clean | 0.13116 | +0.07% |
| raw + clean + partial rate | 0.13080 | −0.21% |

Nothing clears the 0.25% floor, and the cleaned quantity alone is materially
worse.

## Why, and it is two things

**The contamination is signal.** A player's raw share includes his attrition,
and attrition persists — injury-prone players keep getting hurt. That is the
same effect the injury-history features just picked up at the availability
layer. Cleaning deletes a durability signal and keeps only the healthy-day role.

**Cleaning trades bias for variance, and loses.** The clean mean is computed on
fewer games — 9.9% fewer on average, far more for the players it is supposed to
help most. The third row above is the tell: the clean quantity predicts *itself*
next season worse (0.7401) than the raw quantity predicts itself (0.7626). A
quantity that is intrinsically less self-predictable is noisier, not cleaner.

## What is worth doing instead

The evaluation, not the regression. Roughly 23.9% of relative MAE on the drafted
pool is missed games that no preseason input can forecast (see the availability
oracle in [final-week-2026-08.md](final-week-2026-08.md) and the ADP work). Every
gate this session has been reading modelling changes of 1-2% against a metric
that is a quarter irreducible noise.

Scoring a **full-season subpopulation** alongside the existing ones would
separate "did we model the player" from "did we model his availability", at no
sampling cost — `score_fantasy_points_posterior` already takes a `subset`, added
earlier today for the drafted pool. That makes future gates more sensitive
without changing what the package optimises, which should stay season totals
because that is what a drafter buys.
