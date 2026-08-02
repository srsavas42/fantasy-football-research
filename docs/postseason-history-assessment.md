# Should postseason stats feed the late-season role signal?

Assessed 2026-08-02, nflverse 2014-2024. Short answer: **yes, as a separate
flagged feature — not by relaxing the regular-season filter.** The signal is
real and about the size of a promoted volume-v3 pathway, but the obvious
implementation would quietly break the pipeline's exposure accounting.

## Where postseason is dropped today

One choke point: `ingest.load_weekly` filters `season_type == "REG"`
(`src/ffmodel/data/ingest.py:152`). Nothing downstream ever sees a playoff row,
so `LATE_SEASON_START_WEEK = 10` currently means "weeks 10-17/18 of the regular
season" and stops at the regular-season finale.

## 1. The signal is real

The late-season role enters the model through `prior_target_role` /
`prior_carry_role`, both `0.65 * full_season + 0.35 * late_season`. Replacing
the late component with one computed over late regular season **plus**
postseason, scored against the following season's realized share on the same
team, over 2014-2024 returning players:

| response | current `0.65·full + 0.35·late` | with postseason in the late term | change |
|---|---:|---:|---:|
| target share — MAE | 0.03555 | 0.03488 | **-1.9%** |
| target share — Spearman | 0.7866 | 0.7930 | +0.8% |
| carry share — MAE | 0.11269 | 0.11090 | **-1.6%** |
| carry share — Spearman | 0.6953 | 0.7049 | +1.4% |

Scored on the 38% of returning player-seasons where the signal actually changes
(1,035 target rows, 309 carry rows). Pooled over every returning player the
effect is roughly 0.7% and 0.6%. For comparison, the pathways promoted in
volume-v3 moved pooled target MAE 0.27% and carry MAE 0.98%, so this is a
competitive candidate rather than a rounding error.

The late-breakout case that motivates the idea does benefit, but the evidence is
thin on its own: restricting to players whose late share exceeded their full
share by more than 5 points, target MAE improves 1.7% (n=39) and carry MAE 2.0%
(n=74). Those samples are too small to carry a promotion decision by themselves.

## 2. Why not simply relax the `REG` filter

Three things break, in increasing order of severity.

**Exposure accounting.** `games`, `team_games`, and
`observed_availability = games / team_games` all count regular-season games, and
the availability model's Beta-Binomial takes `team_games` as its trial count.
Letting playoff weeks through inflates `games` for playoff teams only, pushes
`observed_availability` above 1, and silently changes what "a season" means for
40% of teams.

**Team-total coherence.** `team_season_volume` builds the denominators every
share divides by, and the pipeline asserts
`pass attempts = player targets + no-target attempts` per draw. Postseason rows
would enter player numerators and team denominators at different rates depending
on which aggregation saw them, and the draw-level conservation checks would
start failing for reasons unrelated to the models.

**Scale.** Playoff rotations are shorter. Mean top-1 within-team target share
rises from 0.242 in late regular season to 0.279 in the postseason — about 15%
more concentrated. Pooling the two into one "late" number averages two different
distributions, which is the same class of mistake as the cold-start prior fixed
in S1. It happens to help here because the correlation is high (0.66-0.80
depending on position), but it is help despite the construction, not because of
it.

## 3. Recommended shape

Add postseason as its own lagged, explicitly-flagged feature set, leaving every
regular-season aggregate and all exposure accounting untouched:

- `prior_post_target_share`, `prior_post_carry_share`, `prior_post_pass_share` —
  shares within the team's postseason totals, so the scale question is handled
  by keeping the two populations separate rather than by pooling them.
- `prior_post_games` — 1 to 4, and the model's own reliability weight. The
  median qualifying team plays 2; 62 of 142 qualifying team-seasons play only 1.
- `prior_post_available` — 0/1. Only 142 of 352 team-seasons (40%) have any
  postseason at all, and the missingness is emphatically **not at random**:
  a missing value means the team was not good enough to qualify. This is exactly
  the pattern `COMBINE_FEATURES` already uses ("absence carried as its own
  signal rather than imputed") and that `pass_sacks_available` /
  `snap_counts_observed` use for measurement coverage. Follow it rather than
  imputing a zero share, which would read as "played and earned nothing".

Leakage is not a concern: season Y's postseason finishes in February of Y+1,
before Y+1's week 1, so it is legitimately available to a Y+1 preseason
projection under the existing lag contract.

## 4. Cost and sequencing

The ingest change is small — thread a `season_type` parameter through
`load_weekly` and add a postseason-only aggregation alongside
`player_team_season_usage` — but it touches the feature contract that every
model reads, so it needs the full volume-v3 acceptance gate (three holdouts, all
required metrics, no protected pass-stream regression).

I would sequence it **after** the QB workload availability coupling. That
candidate is measured at 20x this one's effect size on the layer it touches, and
both change the same allocation, so running them together would make neither
attributable.
