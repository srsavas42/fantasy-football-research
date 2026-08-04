# Data-quality notes from the tail-coverage review (2026-08-04)

Checks run against the cached frames while reviewing the cold-role work. These
are about the data rather than the code, so they do not show up in a diff.

## 2016 carries about 280 rows no other season has

| season | rows | named | snaps > 0 | prior_snap_share NaN |
|---|---:|---:|---:|---:|
| 2015 | 619 | 491 | 537 | 126 |
| **2016** | **986** | **858** | 573 | **378** |
| 2017 | 668 | 548 | 519 | 166 |
| 2018–2025 | 692–728 | 564–600 | 559–599 | 148–192 |

The excess is entirely in players with **no snaps and no prior snap share**, and
it is spread across every position — named rows go QB 82 → 114, RB 137 → 221,
WR 171 → 347, TE 101 → 176 between 2015 and 2016, then back to 81/153/204/110
in 2017. Snap coverage itself is normal at 573. So 2016 admits roughly 280 extra
fringe players who recorded a stat line and never appear in the snap feed.

Whether that is broader source coverage for that season or an ingest artifact is
not settled here. What matters for this branch is that 2016 contributes about
50% more rows than its neighbours to every fit that trains on it, and that
almost all of the excess lands in the cold population.

### It does not move the number the feature is sized by

| stream | window | cold rms | warm rms | ratio |
|---|---|---:|---:|---:|
| carry | all training seasons | 2.6783 | 1.9363 | 1.383 |
| carry | excluding 2016 | 2.6733 | 1.8716 | 1.428 |
| target | all training seasons | 1.8778 | 1.1475 | 1.636 |
| target | excluding 2016 | 1.8986 | 1.1108 | 1.709 |

Dropping 2016 moves the cold–warm ratio by 3 to 4%, in the direction of a
*wider* cold scale. Under the promoted `measured` mode the multiplier caps at 6
either way, so nothing served today depends on this.

It would matter if the cap came down. `cold_role_multiplier_cap` is the open
item in that feature (task 36), and any inner-fold selection of it should decide
deliberately whether 2016 belongs in the estimate rather than inheriting it.

## The cold population is otherwise stable

| | share of rows cold | rookie share of cold |
|---|---:|---:|
| 2015 | 30.4% | 72.9% |
| **2016** | **44.8%** | **50.7%** |
| 2017–2025 | 28.8%–36.7% | 64.7%–74.9% |

Every season but 2016 sits in a narrow band on both measures, which is what a
multiplier fitted on history and applied to a new season needs. 2016 is the only
year where the cold group is both much larger and much less rookie-dominated —
the same 280 rows, seen from the other side.
