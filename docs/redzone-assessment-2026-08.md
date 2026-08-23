# Red-zone role: built, measured, and it is mostly noise (2026-08-20)

The frame carries EPA, air yards and shrunk touchdown rates but nothing about
*where on the field* a player's opportunities came from, and play-by-play was
pulled but never used to build features. That looked like the clearest genuinely
absent signal in the package.

`ffmodel.features.redzone` now builds it from play-by-play: per player-season
shares of team carries and targets inside the twenty and inside the five, and —
the quantity that matters — the **differential** against the player's ordinary
share. A raw red-zone share is mostly a restatement of overall volume; the
differential is the part that would encode "goal-line back" as a trait.

## It does not help

Walk-forward 2020-2024, every probe run with the frame's existing predictors
present, targets ordered before running:

| target | rank | base MAE | + zone | change |
|---|---|---:|---:|---:|
| rush_td_rate | primary | 0.04190 | 0.04195 | +0.11% |
| rec_td_rate | primary | 0.02412 | 0.02417 | +0.20% |
| carry_share | secondary | 0.04108 | 0.04128 | +0.49% |
| target_share | secondary | 0.03274 | 0.03280 | +0.20% |

Restricting to rows where the feature actually exists rather than median-filling
(n=1741 and n=2425) does not rescue it: +0.29% and +0.26%.

## Why: the trait does not persist

Year over year, same player, consecutive seasons with 20+ opportunities:

| quantity | year-over-year r |
|---|---:|
| overall carry share | 0.777 |
| overall target share | 0.724 |
| red-zone carry share | 0.744 |
| goal-line carry share | 0.680 |
| **red-zone carry differential** | **0.195** |
| **red-zone target differential** | **0.194** |
| **goal-line carry differential** | **0.179** |
| **goal-line target differential** | **0.202** |

Red-zone *usage* persists strongly — because volume persists, and the frame
already has volume. The *differential* persists at r ≈ 0.19, which is about 4%
of variance. The goal-line back is largely a story told after the fact about one
season's carry distribution, not a stable property a forecast can lean on.

That is the whole explanation. The feature is not badly encoded and it is not
being shrunk; there is very little there to find.

## A bug worth recording

The first build reported 48-53% coverage on the goal-line differentials, which
should have been near-universal. Team denominators were being taken from the
player's own row, so a back with no goal-line carries got a missing share
instead of a share of zero — a different claim, and one that silently dropped
half the population. Team totals now join on the team. The check that caught it
is that shares must sum to one across a team: they now do, to three decimals,
and the differentials centre on zero by construction.

## Scope

Season-level. This does not test in-season use, where field position is
observed rather than forecast and where the same information is worth much more.
It also does not test the per-play expected-touchdown weighting that
`ff_opportunity` already computes, which is a different construction — though
that source was separately measured as null in
[xfp-assessment-2026-08.md](xfp-assessment-2026-08.md).
