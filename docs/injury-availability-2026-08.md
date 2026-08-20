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
