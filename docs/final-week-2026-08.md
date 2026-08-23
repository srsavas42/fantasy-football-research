# Should the season end a week early? (2026-08-19)

Opened from a question about resting: teams with seeding settled sit their
starters in the last week, so the season totals this package predicts carry
production no preseason input could forecast. The premise is true. It is also
much smaller than the folk version, and it turns out to be the weaker of the two
arguments for making the change.

Measured by `scripts/measure_final_week.py` over 2015–2024.

## Resting is real, and narrower than advertised

Of 1188 established starters — 55%+ snap share over the season's other weeks,
10+ games — **7.1%** played under half their usual snaps in the finale. Split by
the team's record entering that week:

| team entering the final week | n | rested |
|---|---:|---:|
| 11+ wins (seed likely settled) | 240 | **11.2%** |
| 6 to 10 wins (still alive) | 654 | 6.0% |
| 5 or fewer wins (eliminated) | 261 | 6.1% |

Clinched teams rest at nearly double the rate of teams still playing for
something. Eliminated teams do not: at 6.1% they are indistinguishable from
contenders. The common framing — "teams with nothing to play for" — is wrong in
its second half. A bad team has every reason to keep playing its starters.

Rested starters scored 4.37 against 12.29 for a normal load, so the expected
cost to a season total is about **0.6 points**, against a 144-point mean on the
drafted pool.

## The noise argument does not carry the decision

The final week is **+4.4%** harder to predict from a player's own average over
the other weeks than a typical week is, and carries **5.8%** of all season
points.

Scoring the same projection — an oracle handed each player's realized per-game
rate and told nothing about who misses time — against both targets:

| target | n | observed mean | MAE | MAE % |
|---|---:|---:|---:|---:|
| full season (as shipped) | 2942 | 131.75 | 15.29 | 11.6% |
| without the final week | 2942 | 124.76 | 14.15 | **11.3%** |

**0.26 percentage points.** And this oracle is the most favourable setting the
change will ever see: it has perfect rate knowledge, so the final week's noise
is the largest proportion of its error it can be. The shipped model sits at 38%
on the drafted pool, where the same absolute noise is a far smaller share.

### The number this nearly reported

Letting each target qualify its own population — players with 10+ games in
*that* target — let the shorter season keep a different set of players, and the
gap read **−1.43** percentage points. On a matched population it is −0.26. Five
and a half times smaller, and the difference was entirely which players were in
the pool.

## The argument that does carry it

Almost no fantasy league plays the NFL's final week. Championship week is week
17 of 18 since 2021, and week 16 of 17 before that. The season total this
package publishes **includes a week nobody's team played.**

That is not an accuracy claim and does not depend on any measurement above.
Even if the final week were perfectly predictable it would not belong in the
target, because it is not part of the season being asked about. The model is
answering a slightly different question than a drafter is asking.

## What the change costs

Bigger than it looks, and not a one-line filter.

- **The label and the feature history have to move together.** Prior-season
  rate features computed over the full season feeding a short-season target is
  a train/serve mismatch of exactly the kind this branch has already fixed
  twice.
- **`team_games` and the exposure layer move.** Availability is modelled as
  games over team games; both change.
- **Every promoted decision was gated against the full-season target.** The
  postseason role features, `cold_role_innovation`, the innovation cap and the
  snap prior were all accepted on margins measured against an 18-week season.
  Their margins are small enough that none of them can be assumed to survive
  a change in what is being predicted.

## Recommendation

Make the change, for the relevance argument alone, and do not sell it as an
accuracy improvement — the honest estimate is 0.26 percentage points on the
friendliest possible measurement.

Do it as its own re-baselining pass, after the ADP ablation and task 33, where
the only thing that moves is the definition of the season and every gate is
re-run against it. Folding it into other work would make both uninterpretable.
