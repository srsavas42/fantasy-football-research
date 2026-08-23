# Vegas win totals in the team layer: a null, and a legible one (2026-08-23)

## Why this layer

Preseason win totals are the only market input in this package that is an
opinion about *teams* rather than players, and the team layer is the one the ADP
work never reached. Every other stream had already been tested against the
board; this had not been tested against anything.

## The encoding, which is the load-bearing choice

`market_win_total` is the win total **standardized within season**, not pooled.

The raw line is not comparable across eras. The schedule went from sixteen games
to seventeen in 2021 and the average line went with it, 8.15 to 8.56, leaving
the raw number correlated **0.095** with the season index. The team model already
carries an `era` coefficient in every stream for exactly that drift, so a pooled
standardization hands it a second, partly-collinear copy — the same shape of
mistake the rejected ADP interaction arm made. Within-season standardizing takes
that correlation to **0.000** and leaves the feature saying only what it should:
where a team sits against the rest of its own league-year.

The coefficient enters all four streams (plays, pass rate, sack rate, target
rate) with a `Normal(0, 0.15)` prior, matching the scale of the `era` terms it
sits beside.

## The gate

Paired 2022–2024, both arms on a cache differing only in the two market columns,
zero divergences in both:

| population | MAE | folds | CRPS | folds |
|---|---:|---|---:|---|
| all rostered | −0.15% | 2/3 | −0.13% | 3/3 |
| drafted | −0.26% | 2/3 | −0.25% | 3/3 |
| undrafted | +0.04% | 1/3 | +0.09% | 0/3 |

**Nothing clears the 0.25% materiality floor.** Drafted-pool MAE at −0.26% and
CRPS at −0.25% sit exactly on it, on two and three folds respectively. That is
not a pass, and the floor exists precisely to stop an effect this size being
read as a finding — as does this package's own rule that a consistent *sign* is
not a result.

## Why it is null, which is the interesting part

The market does know something the play-by-play history does not. It is just
very far upstream of a fantasy point.

Regressing each same-season team rate on its prior-season value, then adding the
market covariate, over 2016–2025:

| team rate | extra variance explained by the market | coefficient |
|---|---:|---:|
| **opportunity plays per game** | **4.60%** | +0.674 |
| pass rate | 0.89% | +0.0042 |
| target rate | 0.26% | +0.0007 |

So the win total carries real incremental information about **team play volume**
— 4.6% of the variance the prior season leaves unexplained, which is exactly
where a market opinion on team quality should show up. It is also partly
redundant with what the history already says: the feature correlates +0.38 with
prior plays per game and −0.42 with prior sack rate.

That 4.6% then has to survive the trip to a player's season total, through role
allocation and per-opportunity efficiency, both of which carry far more variance
than team play volume does. By the time it arrives it is under a quarter of a
percent.

## What this rules out, and what it does not

It rules out the version of the hypothesis worth ruling out: that the team layer
was missing a cheap, powerful market signal. It was not. The layer is close to
saturated by its own lagged rates, and the market's incremental contribution is
real, small, and diluted further downstream.

It does not rule out win totals helping somewhere else. Two places remain
untried and are better motivated *because* of this result:

- **Game script**, which needs the game-level spread rather than the season
  total. A team projected to trail passes more, and that is a within-season,
  per-game effect this feature cannot express.
- **Player props**, which price receptions, yards and touchdowns directly. Those
  are opinions about the efficiency layer — where every feature tried so far has
  come back null, and where a market input has never been tested.

The flag stays off, kept so the negative result stays reproducible.
