# Total-season scoring v1 validation

Validation date: 2026-07-18. Scoring v1 combines the promoted volume-v3
posterior with efficiency-v2 marginals and converts each coherent simulated
stat line into standard, half-PPR, and PPR fantasy points.

## Simulation contract

For each player and posterior draw, the simulator samples completions,
interceptions, passing touchdowns, receptions, receiving touchdowns, rushing
touchdowns, passing/receiving/rushing yards, and fumbles lost conditional on
the corresponding volume exposure. It enforces:

```text
completions + interceptions <= pass attempts
passing touchdowns          <= completions
receptions                   <= targets
receiving touchdowns         <= receptions
rushing touchdowns           <= carries
fumbles lost                 <= pass attempts + targets + carries
```

Passing and receiving yards are zero when their realized completion or
reception count is zero. The same stat draws are scored under all three scoring
systems, preserving draw-level comparisons. The production pipeline can be
fit, saved, loaded, and used end to end; validation assembles the frozen
volume-v2 posterior plus the two promoted volume-v3 component replacements so
the candidate is compared against the exact accepted volume architecture.

## Main walk-forward result

The benchmark uses the same volume-v3 draws and the accepted efficiency-v1
point means with event noise. The challenger changes only the efficiency layer
to the full efficiency-v2 posterior. Results pool the 2022-2024 holdouts.

| Scoring | Model | MAE | CRPS | 80% coverage | 95% coverage | MAE wins | CRPS wins |
|---|---|---:|---:|---:|---:|---:|---:|
| Standard | accepted point | 33.7053 | 24.5766 | 0.744 | 0.882 | - | - |
| Standard | posterior efficiency | 33.6736 | 24.5915 | 0.754 | 0.892 | 2/3 | 1/3 |
| Half-PPR | accepted point | 38.8811 | 28.5412 | 0.745 | 0.879 | - | - |
| Half-PPR | posterior efficiency | 38.8392 | 28.5459 | 0.750 | 0.886 | 3/3 | 1/3 |
| PPR | accepted point | 44.2479 | 32.6130 | 0.747 | 0.879 | - | - |
| PPR | posterior efficiency | 44.2055 | 32.6115 | 0.752 | 0.888 | 2/3 | 1/3 |

The posterior-efficiency model improves MAE by roughly 0.09%-0.11%, but CRPS
is effectively flat and wins only one holdout for each scoring system. Its 95%
coverage also remains below the 90% promotion floor. Total-season scoring v1
therefore **does not pass the promotion gate**.

## Calibration and dependence ablations

- Scaling efficiency dispersion from 0.75x through 1.50x did not produce a
  candidate that passed accuracy, CRPS stability, and coverage together.
- Shrinking total fantasy-point draws to 0.90x around their mean improved PPR
  CRPS by 0.78%, but collapsed coverage to 0.662/0.800 at the 80%/95% levels.
- Expanding total dispersion to 1.10x or more raised 95% coverage above 0.90
  but worsened CRPS by roughly 1.5% or more.
- An empirical Gaussian copula captured plausible positive within-pathway
  residual relationships, including passing YPA with passing-TD rate and catch
  rate with receiving YPT. With 15% correlation shrinkage it still worsened PPR
  MAE and CRPS and won zero CRPS folds.

The failure is therefore not a one-parameter spread problem. Simple marginal
rescaling cannot jointly repair sharpness and coverage, and a static residual
copula adds dependence without identifying the latent player-season state
that drives both opportunity and efficiency.

## Decision and next challenger

The coherent simulator and end-to-end pipeline remain in the codebase as a
tested research implementation. The independent 1.0x result is the reference
candidate, not a published production projection.

### Draw-conditioned volume-to-efficiency handoff (rejected)

The first follow-up architecture evaluated every fitted efficiency mean at the
matching simulated per-team-game volume draw. This was a directed dependency:
volume was still generated first, and no realized efficiency or scoring outcome
was fed back into it. The intent was to represent a continuous shared role
state more faithfully than the static copula.

It did not clear the scoring gate. The 2022-2024 pooled comparison against the
accepted point baseline was:

| Scoring | Model | MAE | CRPS | 80% coverage | 95% coverage |
|---|---|---:|---:|---:|---:|
| Standard | accepted point | 33.7053 | 24.5766 | 0.7440 | 0.8824 |
| Standard | draw-conditioned | 33.7072 | 24.6127 | 0.7557 | 0.8903 |
| Half-PPR | accepted point | 38.8811 | 28.5412 | 0.7446 | 0.8792 |
| Half-PPR | draw-conditioned | 38.8809 | 28.5720 | 0.7551 | 0.8883 |
| PPR | accepted point | 44.2479 | 32.6130 | 0.7466 | 0.8792 |
| PPR | draw-conditioned | 44.2430 | 32.6387 | 0.7531 | 0.8883 |

CRPS worsened in the pooled result for every format and in two of the three
holdouts. The extra dependence broadened distributions slightly, but it did
not recover the required 95% coverage and harmed sharpness. The implementation
therefore remains an opt-in research flag (`draw_conditioned_efficiency=False`
by default), not a production change.

The next total-scoring challenger should model latent role/regime states
jointly with volume and efficiency: for example, starter/committee/replacement
states that affect playing time, opportunity share, and per-opportunity
efficiency in the same draw. It should be accepted only if it improves pooled
CRPS, wins at least two of three holdouts, preserves point accuracy, and clears
both total-point coverage floors without post-hoc widening.

Machine-readable reports are stored beneath:

- `.cache/season-average-validation/season-scoring-v1/`
- `.cache/season-average-validation/season-scoring-v1-scale/`
- `.cache/season-average-validation/season-scoring-v1-point-scale/`
- `.cache/season-average-validation/season-scoring-v1-copula/`

```powershell
python scripts/validate_season_scoring_posteriors.py `
  --holdouts 2022 2023 2024 `
  --dispersion-scales 1.0 --point-dispersion-scales 1.0 `
  --dependence independent --draw-conditioned-efficiency `
  --output-dir .cache/season-average-validation/season-scoring-v1
```
