"""Should the blend weight vary with draft rank, and should a third arm join it?

Two proposals, both sensible on the face of the tier table, and both measured
here rather than assumed. Every model's advantage over the board grows steadily
deeper into the draft:

    MAE relative to ADP     pipeline   flat_ridge     blend
    top50                      +8.6%        +2.2%     -1.4%
    51_150                     +0.6%        -1.3%     -3.9%
    151_300                    +0.3%        -2.6%     -4.9%

which looks like an argument for trusting the model more as rank rises, and for
finding the rank where one forecast overtakes another.

**The pattern is real and the inference from it is wrong.** The optimal weight at
a rank is the slope of (actual - ADP) on (model - ADP) among players near it.
Fitting that slope with a log-rank interaction, the interaction is not
distinguishable from zero -- pipeline t = -0.47 (p = 0.64), ridge t = -0.65
(p = 0.52) -- and what trend there is runs the *other* way, toward slightly less
weight deeper:

    rank            10      50     100     300
    pipeline     0.424   0.396   0.384   0.365
    flat_ridge   0.594   0.497   0.455   0.388

So there is no crossover to locate. The tier pattern has a different cause:
relative benefit is the weight times the *size* of the disagreement, and
disagreements grow deeper into the draft because the board is less precise
there. The same constant weight buys more improvement at rank 250 than at rank
5 without ever needing to change.

The three-way stack is also empty. Fitted leave-one-season-out, adding the flat
ridge to a pipeline-plus-ADP stack moves MAE by hundredths and its coefficient
will not hold still -- -0.043, +0.171, -0.030 across the three folds, straddling
zero. That is what a redundant arm looks like: the ridge is fitted on the
features the pipeline already consumes and carries ADP among them, so it has
nothing of its own to contribute.

    two-way MAE   three-way MAE   ridge coefficient
    2023  57.03           57.13              -0.043
    2024  54.57           54.54              +0.171
    2025  53.23           53.30              -0.030

    python scripts/screen_dynamic_blend.py
"""
import warnings; warnings.filterwarnings("ignore")
import json
from pathlib import Path

import numpy as np
from scipy import stats

SEASONS = (2023, 2024, 2025)
RANKS = (10, 25, 50, 100, 150, 200, 300)


def load():
    out = {}
    for season in SEASONS:
        path = Path(f"reports/flat_{season}.json")
        if not path.exists():
            raise SystemExit(f"{path} missing; run validate_flat_baseline.py first")
        fold = json.loads(path.read_text("utf-8"))["folds"][str(season)]
        out[season] = {k: np.array(v, dtype=float) for k, v in fold["rows"].items()}
    return out


def rank_varying_weight(observed, adp, rank, prediction, label):
    """Optimal weight as a function of log rank, with the interaction tested."""
    gap = prediction - adp
    error = observed - adp
    centred = np.log(rank) - np.log(rank).mean()
    design = np.column_stack([np.ones(len(gap)), gap, gap * centred])
    beta, *_ = np.linalg.lstsq(design, error, rcond=None)
    residual = error - design @ beta
    dof = len(error) - design.shape[1]
    standard_error = np.sqrt(
        np.diag(np.linalg.pinv(design.T @ design)) * (residual @ residual) / dof
    )
    t = beta[2] / standard_error[2]
    p = 2 * (1 - stats.t.cdf(abs(t), dof))
    print(f"\n{label}: weight = {beta[1]:.3f} {beta[2]:+.3f} * (log rank - "
          f"{np.log(rank).mean():.2f})")
    print(f"  interaction t = {t:+.2f}, p = {p:.3g} -> "
          f"{'varies with rank' if p < 0.05 else 'no rank dependence'}")
    cells = "  ".join(
        f"{r}:{beta[1] + beta[2] * (np.log(r) - np.log(rank).mean()):+.3f}" for r in RANKS
    )
    print(f"  implied weight by rank -- {cells}")


def three_way(seasons):
    print(f"\n{'=' * 66}\ndoes the flat ridge add anything to pipeline + ADP?\n{'=' * 66}")
    print(f"  {'fold':6} {'two-way':>9} {'three-way':>11} {'ridge coef':>12}")
    for test in SEASONS:
        train = [s for s in SEASONS if s != test]
        target = np.concatenate([seasons[s]["observed"] for s in train])

        def fit(columns):
            design = np.column_stack(
                [np.ones(len(target))]
                + [np.concatenate([seasons[s][c] for s in train]) for c in columns]
            )
            beta, *_ = np.linalg.lstsq(design, target, rcond=None)
            held = np.column_stack(
                [np.ones(len(seasons[test]["observed"]))]
                + [seasons[test][c] for c in columns]
            )
            return beta, held @ beta

        _, two = fit(["pipeline", "adp"])
        beta3, three = fit(["pipeline", "adp", "flat_ridge"])
        observed = seasons[test]["observed"]
        print(f"  {test:6} {np.abs(observed - two).mean():9.2f} "
              f"{np.abs(observed - three).mean():11.2f} {beta3[3]:+12.3f}")
    print("\n  a coefficient that changes sign across folds is a redundant arm,")
    print("  not a weak one.")


def main() -> int:
    seasons = load()
    pull = lambda key: np.concatenate([seasons[s][key] for s in SEASONS])
    observed, adp, rank = pull("observed"), pull("adp"), pull("rank")
    print(f"{len(observed)} drafted player-seasons")
    print("\nthe optimal weight at a rank is the slope of (actual - ADP) on")
    print("(model - ADP) among players near it")
    for label in ("pipeline", "flat_ridge"):
        rank_varying_weight(observed, adp, rank, pull(label), label)
    three_way(seasons)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
