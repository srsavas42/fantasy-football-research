# What a forecast actually delivers, and when to ask for it (2026-09-05)

Every weather number in [specialists & weather](specialists-and-weather-2026-09.md)
used the conditions nflverse recorded **at** the game. That is a ceiling: what
perfect foreknowledge would have been worth. This document is about the two
questions that stand between that ceiling and something shippable — what
Open-Meteo actually serves, and which forecast a decision is entitled to.

Pulled by `scripts/fetch_stadium_weather.py` on a GitHub runner
(`.github/workflows/pull-stadium-weather.yml`), because the research sandbox's
egress policy denies Open-Meteo. `--probe DATE` makes one tiny request per
source and writes `data/weather/probe.md`; run it before any backfill.

## Precipitation: yes, and rain is separable from snow

All eleven requested variables are served, at valid time **and** at forecast
lead times: `temperature_2m`, `apparent_temperature`, `precipitation`, `rain`,
`snowfall`, `weather_code`, `wind_speed_10m`, `wind_gusts_10m`,
`wind_direction_10m`, `cloud_cover`, `relative_humidity_2m`.

`precipitation` alone would not have answered the question. It is
water-equivalent, so an inch of snow and a light shower can carry the same
number; `snowfall` and `weather_code` are what separate them, and they are
requested for that reason rather than inferred from temperature. Gusts are
carried alongside mean wind because a kicked or thrown ball meets the gust.

Units are requested in Fahrenheit, mph and inches, so the result is directly
comparable to the nflverse columns rather than a conversion away from it.

## The coverage trap: the columns are present before the data is

**A variable being "served" is not the same as it having values.** The
previous-runs archive returns the full column set for any date it accepts, and
fills only some of it. Two spot probes on Buffalo first suggested the boundary
sat somewhere between 2022 and 2025; the full backfill below pins it exactly.

### The exact boundary (2016-2025 backfill, 7,917 rows, all 42 stadiums)

| source | variable | 2016-2020 | 2021 | 2022 | 2023 | 2024-2025 |
|---|---|---:|---:|---:|---:|---:|
| `observed` | temperature | 0%\* | 100% | 100% | 100% | 100% |
| `observed` | wind, precip, snowfall | 0%\* | 100% | 100% | 100% | 100% |
| `lead_1` / `lead_4` | temperature | 0% | 99.3% | 98.9% | 87.5% | 100% |
| **`lead_1` / `lead_4`** | **wind, precip, snowfall, gusts** | **0%** | **0%** | **0%** | **0%** | **100%** |

\*`observed` is effectively unusable before 2018 too -- 2016 is 0%, and 2017 is
2% (five international-site games only; every domestic 2017 game is null). That
does not affect anything already published, since the ceiling numbers in
[specialists & weather](specialists-and-weather-2026-09.md) read temperature and
wind from **nflverse's own schedule columns**, not from Open-Meteo -- this
backfill is a separate pull built to test the forecast, and the two have never
been joined.

**The forecast archive's non-temperature variables exist only for 2024 and
2025.** Temperature reaches back to 2021 on its own, but wind, precipitation,
snowfall and gusts -- everything the gated wind hinge and the precipitation
question actually need -- are entirely null before 2024 at every lead time
tested. This is not a partial-coverage inconvenience; it is a hard floor. Any
forecast-lead measurement of the wind feature has **two usable seasons**, not
ten.

That collapses the backtest this package can run on it. Everywhere else in this
package, a claim is walked forward across three holdout seasons specifically
because one is not enough to trust -- the fold-by-fold tables throughout
[specialists & weather](specialists-and-weather-2026-09.md) exist for that
reason. A forecast-lead version of the gated wind rung has exactly one possible
split (train on 2024, test on 2025), and the population it would be scored on --
rows above 15 mph, already only 5.6% of the panel, now confined to a single
season -- is small enough that a null and a real effect would look similar. A
single-fold number here would not carry the weight a reader familiar with the
rest of this package would reasonably assign it, so it is not run rather than
run and mislabeled. The honest state of this question is **not "measured
null," but "not enough forecast history exists yet to measure."** It becomes
answerable, on the same one-fold basis, once 2026 is in hand, and on the
package's usual three-fold basis in 2028.

This nearly went unnoticed even at the single-stadium probe scale. The script's
per-source progress line reports the share of kickoff hours matched, and it
computes that from `temperature_2m` -- the one variable that is populated
everywhere. An early run printed "lead_1: 100% of kickoff hours matched" for a
season whose wind column was entirely null. Printing a sample row in probe mode
is what caught it there; the full backfill's per-variable coverage table above
is what turned it into an exact date rather than a suspicion.

## Which forecast, and what it costs

The lead time a decision is entitled to is set by its deadline, and those
differ. Measured against the 2023-2025 schedule from a Wednesday 16:00 waiver
cutoff:

| game day | lead (days) | share of slate |
|---|---:|---:|
| Thursday | 1.18 | 7.2% |
| Saturday | 3.02 | 2.6% |
| Sunday | 3.88 | 81.6% |
| Monday | 5.18 | 7.7% |

Waiver claims must be in by Wednesday, so they get roughly a four-day forecast.
A start/sit call can wait until Sunday morning — a four-*hour* horizon — at no
cost. A single fixed lead is wrong for about a fifth of the slate, so the join
is per game; Open-Meteo archives offsets 1-7, which covers all of them.

### The tail is where the signal is, and where forecasts are worst

Two probes are not a skill curve, but they frame the problem sharply.

A benign Sunday (Buffalo, 2025-11-16) is forecast well at both horizons:

| | temperature | wind |
|---|---:|---:|
| observed | 38.9 | 14.5 |
| lead 1 | 37.9 | 15.0 |
| lead 4 | 37.9 | 12.3 |

An extreme one (Buffalo, 2022-12-24 — the blizzard game) is not:

| | temperature |
|---|---:|
| observed | **1.9** |
| lead 1 | 5.4 |
| lead 2 | 9.1 |
| lead 4 | **20.6** |
| lead 7 | 20.1 |

Four days out, the forecast was **19 degrees warm** on the coldest game in the
sample. That is the shape of forecast error in general — skill collapses fastest
on extremes — and it lands precisely where this package measured the effect.
The weather signal is a tail effect: −0.60% CRPS on the 5.6% of rows above
15 mph, nothing anywhere else. A forecast that is good on calm days and poor on
extreme ones is good exactly where the feature does not pay.

That is an argument, not a measurement, and it points one way: **for start/sit,
take the Sunday-morning forecast, because waiting is free and the decision-relevant
cases are the ones a Wednesday forecast gets worst.** For waivers there is no
choice, and the honest expectation is that a lead-4 wind feature retains
noticeably less than the −0.60% ceiling — possibly little enough to leave the
whole thing below the promotion floor.

That was the plan, and the backfill overturned it before it could run. The
coverage boundary above means `lead_1`/`lead_4` wind is null before 2024, so the
"re-run the gated wind ladder against forecasts" measurement has exactly one
usable holdout (train 2024, test 2025) instead of the three every other claim in
this document was walked forward across. A single-fold number on a feature
that's already only 5.6% of rows, confined to one season, would not carry the
weight a table like the ones above implies -- so it is not run and reported as a
number here. It becomes a real measurement, on the package's usual basis, once
2028 supplies a third holdout; a provisional one-fold read is possible in 2027
with 2026 added. What this section settles instead is the two questions that
motivated the backfill: precipitation is available and separable from snow
(confirmed), and Sunday-morning beats Wednesday for start/sit specifically
*because* forecast skill is worst on the extremes the feature depends on
(argued from the two probes above, not yet from a walk-forward -- the coverage
floor is exactly why that argument can't yet be upgraded to one).
