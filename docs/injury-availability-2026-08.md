# Injury features: built, contracted, and never measured (2026-08-20)

`ffmodel.features.season_injury` builds eleven leakage-safe injury covariates —
three years of report weeks, out weeks, episode counts, mean recovery weeks and
weeks since the last episode, plus a preseason snapshot of current injury status,
practice severity and expected recovery. They are listed as a formal contract in
`INJURY_AVAILABILITY_FEATURES`, they are present in every built frame at 100%
coverage, and `scripts/validate_injury_availability.py` exists specifically to
screen them.

Before today that script had never successfully run. `SeasonAvailabilityModel`
still had `extra_features = ()`.

## Why it never ran

The script defaulted to `--nuts-sampler nutpie`. nutpie is not installed here and
is not a declared dependency, so every invocation died on an `ImportError` inside
PyMC's external-sampler dispatch before reaching a single fit. Because it never
produced output, nothing recorded that it had never succeeded — an absent report
looks the same as a report nobody asked for.

It now defaults to `pymc`, which every other validation on this branch uses, and
takes `--cache-dir` so it screens the same frames as everything else instead of
building a private one.

## The screen

2022–2024, 1000 draws, 1000 tune, 4 chains, `.cache/ffmodel-wf-2025-adp2`.

| population | Δ MAE | Δ CRPS |
|---|---:|---:|
| all rostered | −0.65% (2/3 folds) | **−2.39% (3/3 folds)** |
| injury-exposed | −1.29% (2/3 folds) | **−5.15% (3/3 folds)** |

Per fold, pooled:

| fold | base MAE | challenger | base CRPS | challenger |
|---|---:|---:|---:|---:|
| 2022 | 0.22105 | 0.21721 (−1.74%) | 0.15288 | 0.14753 (−3.50%) |
| 2023 | 0.20506 | 0.20428 (−0.38%) | 0.12937 | 0.12661 (−2.14%) |
| 2024 | 0.20081 | 0.20146 (**+0.32%**) | 0.12781 | 0.12620 (−1.26%) |

Nine of the eleven contracted features survive the design matrix's variance
filter. The two that do not — `injury_history_available` and
`current_injury_snapshot_available` — are always-on flags with sd 0.000 in this
frame, so they carry no information by construction.

## Reading it

CRPS moves three to four times as far as MAE, and unanimously where MAE does not.
That is the same signature the ADP feature showed, and it means the same thing:
injury history tells the model **how uncertain to be** about a player's games
rather than where to put the mean. A player with three years of soft-tissue
episodes does not have a predictably lower games total so much as a wider one,
and a metric that only scores the centre cannot see that.

2024 regressing 0.32% on pooled MAE is recorded rather than rounded away. It is
within the fold-to-fold spread, but it is the one fold that disagrees.

## What this does not yet establish

The screen is at the availability layer, which is deliberate — it is cheap and it
isolates the question. But availability feeds exposure, exposure feeds every
volume stream, and every stream feeds points. A gain here is a gain in projected
games, not in projected fantasy points, and the two are not the same claim.

`injury_availability_features` is wired on `SeasonAverageVolumePipeline`, off by
default, scoped to the availability model only (verified: no injury column
reaches the snap or allocation designs), guarded against frames that predate the
columns. The scoring walk-forward decides whether it ships.

## The 2025 confirmation does not confirm (2026-08-20)

The scoring gate accepted on 2022-2024 with every accuracy metric improving on
three folds of three. On 2025 — the season no choice in this package has seen —
the effect is not there.

| metric | 2025 | 2022-24 pooled |
|---|---:|---:|
| ppr/mae | −0.12% | −1.35% |
| ppr/crps | −0.12% | −1.55% |
| ppr/rmse | **+0.02%** | −1.43% |
| drafted/mae | −0.18% | −1.82% |
| drafted/crps | −0.12% | −2.13% |
| drafted/rmse | **+0.15%** | −1.73% |

Every number is inside ±0.2%, against a 0.25% materiality floor. Two are the
wrong sign. The challenger fitted its nine injury columns and the control fitted
none, so this is not a mis-wired arm; zero divergences in both.

Per fold the effect declines monotonically across the window:

| fold | points ΔMAE |
|---|---:|
| 2022 | −1.59% |
| 2023 | −1.85% |
| 2024 | −0.61% |
| **2025** | **−0.12%** |

### What to make of it

This is not the usual overfitting story. Nothing was tuned on 2022-2024: the
eleven features were built before this session, the contract was fixed, and the
screen and the gate were each run once. There is no search here to have
overfitted.

Three readings, and the data does not separate them:

- **Year-to-year variation.** Three folds is not many, and 2024 was already the
  weak one. 2025 may simply be another weak year.
- **A real decline.** Injury-report practice and roster rules have changed
  across this window, and a feature built on reporting conventions can decay.
- **Chance in the other direction** — that 2022 and 2023 were the unusual folds
  and the true effect was always small.

### Recommendation, revised

**Do not promote on this evidence.** An in-window gate pass with a flat
confirmation is exactly the pattern the reserved season exists to catch, and the
honest description is "accepted in-window, unconfirmed out-of-sample" rather than
"clears the gate".

What would settle it is more holdouts rather than more argument: the walk-forward
window can extend back before 2022, and if the effect is real it should appear
in 2019-2021 at a size like the 2022-2023 one. That is six more fits and it is
the cheapest thing that would move the question.

The availability-layer screen still stands on its own — CRPS −2.39% on three
folds of three, −5.15% on the injury-exposed half. What has not survived is the
claim that it reaches fantasy points.

## Six holdouts, on a window that finally samples cleanly (2026-08-21)

The three-fold gate accepted with every accuracy metric improving 3/3. Six
holdouts, all twelve fits on one configuration with **zero divergences in either
arm**, say something weaker:

| metric | six folds | three folds |
|---|---:|---:|
| ppr_drafted/crps | −0.94% ±0.59%, 3/6 | −2.13%, 3/3 |
| ppr/crps | −0.73% ±0.39%, 4/6 | −1.55%, 3/3 |
| ppr/mae | −0.55% ±0.39%, 3/6 | −1.35%, 3/3 |

Every one is marked *sign varies*. The effect is roughly half what the narrow
window reported, and it does not hold its direction across folds. Together with
the flat 2025 confirmation, six holdouts say **inconclusive**, not established.

`injury_availability_features` stays off.

## The full-season population is the wrong instrument for this question

The gate's only remaining blocker is the population added earlier the same day:
`ppr_full_season/crps` regresses +0.39%, and its siblings with it. That is not
evidence against the feature. It is a defect in the diagnostic.

The population selects on **games actually played**. An availability feature
earns its keep by shading down players who will miss time; restricting the
scoring to players who did *not* miss time removes precisely the rows it gets
right and keeps the ones where its adjustment was wrong. A feature that is
correct on average must look worse inside that group.

The per-fold pattern is the signature:

| fold | full season ΔCRPS | drafted ΔCRPS | all rostered ΔCRPS |
|---|---:|---:|---:|
| 2022 | **+1.07%** | −1.40% | −1.32% |
| 2023 | −0.19% | **−3.27%** | −2.16% |
| 2024 | **+0.15%** | −1.69% | −1.15% |

On the three folds where the feature helps most, the full-season population
shows no benefit or a regression. A genuinely harmful feature would hurt every
population; this one hurts only the one that conditions on the outcome it exists
to predict.

The comment added with the population says it "conditions on an outcome,
deliberately" and that "a model that projects availability badly is partly
excused here". Both true, and both understated the problem: the population is
not merely lenient toward bad availability modelling, it is **actively biased
against good availability modelling**, and it should not gate any change to the
availability layer.

It remains a reasonable instrument for its intended purpose — isolating rate
accuracy from availability accuracy, which is why it was built — but that
purpose excludes exactly the feature it was first used on.
