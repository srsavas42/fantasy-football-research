# The availability layer had a resolution failure, not a level bias (2026-08-22)

## The symptom, and the fix it seemed to call for

Projected games ran 3.7% under observed pooled and 6.0% under on the drafted
pool. That reads as a calibration problem with a one-parameter fix: shift the
intercept up.

It is not, and the intercept fix would have made things worse.

## What settles it

Fit the availability layer alone and score it on **the rows it was trained on**.
Holdout 2024, so training is 2015–2023:

| population | in sample | held out |
|---|---:|---:|
| all | **−0.2%** | −3.5% |
| drafted | **−6.7%** | −8.5% |
| undrafted | **+5.3%** | +1.2% |

Pooled in-sample bias is essentially zero — there is no level to correct. But
the model misses in *opposite directions* on two halves of its own training
data. A model that is unbiased overall while getting both halves wrong is not
mis-levelled; it cannot tell the halves apart. Shifting the intercept would fix
one half by breaking the other.

Era drift was checked first and ruled out. Mean availability by season runs
0.799, 0.581, 0.785, 0.774, 0.672, 0.732, 0.670 across 2015–2021 against
0.669–0.700 across the holdouts, so the training seasons are if anything *more*
available and drift would bias projections high, not low.

## The information that was missing, and why

The draft board. The reason it was missing is documented in
`_enable_market_adp_features`:

> Availability, the team layer and the efficiency layer are left alone in this
> arm — ADP plausibly informs all three, but adding it everywhere at once would
> make a null result uninterpretable and a gain unattributable.

That was a defensible choice. Its effect was that the ADP ablation's null result
**never tested the layer carrying the defect**, so "ADP doesn't help" was never
established for availability.

## Layer-level effect

Holdout 2024:

| population | base | +ADP |
|---|---:|---:|
| pooled | −3.5% | −2.9% |
| **drafted** | **−8.5%** | **−4.1%** |
| RB drafted | −10.8% | −5.6% |
| WR drafted | −9.0% | −4.2% |
| TE drafted | −6.0% | −1.7% |

In sample the flip largely collapses: running backs from −5.6%/+6.3% to
−1.4%/+0.5%, tight ends from −5.8%/+1.7% to −1.4%/−0.4%. Receivers keep about
half of theirs (−5.4%/+4.8%), so this narrows the resolution failure rather than
closing it.

## Scoring gate

Paired 2022–2024, zero divergences in both arms, ADP columns confirmed present
in the candidate's fitted design and absent from the baseline's:

| population | MAE | CRPS | folds improved |
|---|---:|---:|---|
| all rostered | −0.93% | −1.11% | 3/3, 3/3 |
| drafted | −0.44% | −1.44% | 2/3, 3/3 |
| undrafted | −1.83% | −0.48% | 3/3, 3/3 |

Everything clears the 0.25% materiality floor. Drafted-pool MAE is the one
non-unanimous metric (+0.36%, −0.72%, −0.94%) and meets the two-of-three
stability rule rather than needing a waiver. Coverage unmoved. **Promoted.**

## The shape is the evidence

The largest MAE gain is on undrafted players, the largest CRPS gain on drafted
ones, and the pooled figure is smaller than either. An intercept shift could not
produce that — it moves both groups the same direction. The diagnosis said the
layer could not separate the halves, and the fix improves each half in the way
that half was wrong.

## What is left, and what it is not

Receivers keep about half their shrinkage. The obvious reading — weak signal at
that position — is wrong:

| position | YoY availability r | drafted/undrafted gap | undrafted share of rows |
|---|---:|---:|---:|
| QB | 0.116 | 0.145 | 61.3% |
| RB | 0.149 | 0.211 | 50.2% |
| **WR** | **0.194** | **0.262** | 60.7% |
| TE | 0.149 | 0.193 | 73.6% |

Receiver availability is the **most** predictable of the four year over year,
and receivers have the **largest** gap to resolve. The residual is not missing
signal; it is that receivers have the most separation to capture, so a given
degree of shrinkage shows up largest there.

One mechanism is concrete and testable: the model carries position-specific
*intercepts* but a single shared *slope vector*. Receivers are 37.9% of all
training rows and 60.7% undrafted, so the shared ADP slope is fitted largely on
fringe receivers. Position-specific ADP slopes are the natural candidate — with
the prior lowered by the fact that ADP interactions were measured and rejected
in the role layers (see [adp-ablation-2026-08.md](adp-ablation-2026-08.md)),
which is a different layer and so not decisive.

That arm is deliberately not run yet. The [exposure
target](exposure-target-2026-08.md) moves exactly this quantity — it widens the
receiver gap from 0.262 to 0.297 — so testing interactions now would be testing
against a target that is about to change.

## Note on what a win here means

ADP is a forecast. A model that reads it is partly following the market, and a
gain does not mean the model found something the market had not. It means the
two together beat the history alone.

## The quarterback room: null (2026-08-23)

The same board given to the passing-share softmax, its hurdle, and pass
attempts per snap, scored against the promoted configuration on 2022–2024 with
zero divergences:

| population | MAE | CRPS | folds improved |
|---|---:|---:|---|
| all rostered | −0.08% | −0.09% | 2/3, 2/3 |
| drafted | −0.16% | −0.14% | 2/3, 2/3 |
| undrafted | +0.07% | −0.00% | 1/3, 2/3 |

Nothing clears the 0.25% materiality floor. This was the highest prior of the
layers the original ADP arm excluded — the room's evidence about who starts is
`qb_depth_rank` and `qb_listed_starter`, both read off preseason depth charts —
and the prior was wrong.

The likely reason is that the information now arrives upstream. Availability is
ADP-informed as of the promotion above, and the workload softmax takes the
availability draws as its exposure offset while its gate is coupled to the same
draws. By the time the room is allocated, the board has already spoken through
exposure. Adding it a second time restates what is there.

That is consistent rather than contradictory: ADP helped where the information
was missing and does nothing where it had already entered. The flag stays off.
