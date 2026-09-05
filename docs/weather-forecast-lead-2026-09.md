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
fills only some of it. Probing Buffalo on two dates:

| date | source | temperature | wind, precipitation, snowfall, … |
|---|---|---|---|
| 2022-12-24 | observed | populated | populated |
| 2022-12-24 | lead 1, 2, 4, 7 | populated | **all null** |
| 2025-11-16 | observed | populated | populated |
| 2025-11-16 | lead 1, 4 | populated | populated |

So the forecast archive's non-temperature variables begin somewhere between
2022 and 2025, while temperature reaches further back. The practical
consequence is that **the forecast backtest cannot span as many seasons as the
ceiling measurement did**, and the number of usable seasons is a fact to be
measured by the coverage table, not assumed.

This nearly went unnoticed. The script's per-source progress line reports the
share of kickoff hours matched, and it computes that from `temperature_2m` —
the one variable that is populated everywhere. An early run printed
"lead_1: 100% of kickoff hours matched" for a season whose wind column was
entirely null. Printing a sample row in probe mode is what caught it, and the
coverage check should be widened to the variable a model will actually read.

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

The measurement that settles it is the one the backfill enables: re-run the
gated wind ladder against lead-1 and lead-4 forecasts and compare both to the
recorded-conditions ceiling. The gap between those three numbers is the price of
not being able to see the future, and it is the last open question here.
