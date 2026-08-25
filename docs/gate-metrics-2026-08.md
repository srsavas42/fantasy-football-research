# Two new gate metrics, and what they say about the ADP promotion (2026-08-23)

## Why the old gate was one metric wearing three hats

MAE, RMSE and CRPS all measure the same thing — distance from truth in points —
at the centre, in the tails, and over the whole distribution. Coverage at 80%
and 95% samples the calibration curve twice. Nothing knew that the product is a
**ranked list**, and nothing separated *being calibrated* from *being
informative*.

## Resolution and reliability (gate)

The CRPS decomposes (Hersbach) as `CRPS = reliability + potential`, with
`potential = uncertainty − resolution`:

- **reliability** — are the stated probabilities honest? Moved by making the
  posterior wider or narrower.
- **resolution** — does the forecast tell outcomes apart? A model issuing the
  climatological distribution to everyone scores zero however well calibrated.
- **uncertainty** — the CRPS of climatology. A property of the population, so
  it is the yardstick, not a score.

This is the metric the session's central finding was about. The availability
layer was unbiased pooled while missing in opposite directions on drafted and
undrafted players — a resolution failure — and it was diagnosed indirectly, via
bias splits and an in-sample refit. This measures it directly.

The implementation is checked by the identity: `reliability + potential` must
equal `empirical_crps` exactly. It did not at first — the upper outlier bin
divided by `1 − rate` instead of `rate`, which is the natural-looking mistake
and breaks the identity by exactly the amount the ensemble is over-confident.

## Ordering (gate)

Scored **within position**, because that is how a projection is consumed. A
pooled rank correlation is inflated by the gap between positions —
quarterbacks outscore tight ends, and getting that right is not skill. A test
pins this: a projection that knows only a player's position scores 0.85 pooled
and 0.0 within position.

- **spearman** — rank agreement
- **concordance** — share of player pairs ordered correctly (0.5 is chance)
- **top-12** — share of the projected top twelve at a position that finished
  top twelve. One starter per team, so it is the set being chosen.

Top-k must be within group too. Pooled it is near-constant between arms — the
top twelve overall is mostly quarterbacks either way — which made it read 0.333
in all four folds under both models before it was fixed.

## Diagnostics

**PIT shape** — the full calibration curve. Flat is calibrated, U-shaped is
over-confident, humped is over-dispersed. A *shifted* forecast also piles mass
into one tail, so the shift is tested first: reporting bias as over-confidence
would send a reader to widen a posterior when the centre is what is wrong.

**CRPS skill score against the draft board** — turns "1.1% better" into a share
of the gap that was available.

## What they say about the ADP promotion

Drafted pool, 2022–2025, raw (no ADP in availability) → joint (promoted):

| metric | raw | joint | |
|---|---:|---:|---|
| **resolution** | 8.159 | **8.748** | +7.2%, better |
| **reliability** | 1.067 | **1.000** | −6.3%, better |
| CRPS | 44.377 | 43.721 | better |
| pit deviation | 0.267 | 0.239 | better |
| spearman | 0.5580 | 0.5572 | flat |
| concordance | 0.6988 | 0.6985 | flat |
| top-12 | 0.4896 | 0.4805 | slightly worse |
| skill vs board | −0.155 | −0.139 | still negative |

**The promotion improved the distribution and did nothing for the ordering.**
Resolution up 7.2%, reliability down 6.3%, calibration shape closer to flat —
and rank correlation, pairwise concordance and top-12 hit rate all flat to
marginally worse. Every one of those ordering moves is well inside noise, so
the honest reading is *no ordering effect*, not a small loss.

That is a real qualification on what was shipped, and it is not visible in MAE
or CRPS. The feature makes the intervals better and does not help anyone rank
players. It is consistent with the mechanism: availability governs exposure and
therefore the *width* of a season-total posterior far more than it reorders
players within a position.

Two other things fall out.

**The model still loses to the board.** Skill score −0.139 means the joint
model's CRPS is 13.9% *higher* than a rank curve's on the players people draft.
It improved from −0.155, closing about a tenth of the deficit. This is the
shipped blend's whole justification, now stated as a share rather than a
percentage difference.

**The exposure under-projection shows up as a calibration shape.** Three of four
folds report "shifted (projections run low)" for both arms — an independent
confirmation of the −4% drafted-pool exposure bias, arrived at from the PIT
rather than from projected-versus-observed games.

## Note on multiple comparisons

Adding metrics adds chances to be lucky. The gate is **resolution** and
**within-position ordering**, alongside the existing MAE/CRPS and the 0.25%
materiality floor. PIT shape and the skill score are diagnostics: they explain a
result, they do not authorise a promotion. "Improved on two of seven metrics" is
not a promotion argument.
