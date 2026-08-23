"""Does predicting points as volume times efficiency cost accuracy by itself?

The pipeline never regresses fantasy points on anything. It projects exposure,
allocates it into role shares, draws a volume, draws an efficiency from a
separately fitted posterior, and multiplies. A rank curve that regresses points
directly beats it. The natural conclusion is that the composition is lossy —
but that has not been tested, and "the architecture is wrong" is too expensive a
conclusion to reach by elimination.

This tests it with the architecture held as the only variable. One estimator,
one information set (draft rank and position), one machinery (per-position log
fit with residuals resampled from nearby ranks). Three ways of getting to
points:

* **direct** — regress season points on rank.
* **composed, independent** — regress *opportunity* on rank, regress *points per
  opportunity* on rank, draw each independently, multiply.
* **composed, dependence preserved** — the same two regressions, but each draw
  takes its opportunity residual and its rate residual from the *same* training
  player, so whatever correlation exists between busting on volume and busting
  on efficiency survives.

The third arm matters because the second is not merely noisier. For positively
correlated factors ``E[XY] = E[X]E[Y] + Cov(X, Y)``, so drawing them
independently loses the covariance term outright — a bias, not just a spread.
If the independent arm is bad and the dependence-preserving arm is fine, the
lesson is about how the pipeline couples its layers, not about whether it should
have layers at all.

Opportunity is targets plus carries plus pass attempts: every chance the player
had to score, which is the quantity the allocation layers exist to divide up.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

_spec = importlib.util.spec_from_file_location(
    "_adp_only", Path(__file__).with_name("benchmark_adp_only.py")
)
_adp_only = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_adp_only)

MODEL_POSITIONS = _adp_only.MODEL_POSITIONS
RANK_WINDOW = _adp_only.RANK_WINDOW
MIN_RESIDUALS = _adp_only.MIN_RESIDUALS
score = _adp_only.score


def opportunity(frame: pd.DataFrame) -> pd.Series:
    total = sum(
        pd.to_numeric(frame.get(name), errors="coerce").fillna(0.0)
        for name in ("targets", "rush_att", "pass_att")
    )
    return total


def curve(train: pd.DataFrame, test: pd.DataFrame, y: str):
    """Per-position log fit, returning test centres and per-row residual pools.

    Residual pools are returned as index arrays into ``train`` rather than as
    values, so two responses fitted on the same rows can draw the same training
    player for both and keep their dependence.
    """
    centres = np.zeros(len(test))
    pools: list[np.ndarray] = [np.array([], dtype=int)] * len(test)
    residuals = np.zeros(len(train))
    test_positions = test.position.to_numpy()
    train_rank = train.adp_rank.to_numpy(float)
    for position in MODEL_POSITIONS:
        fit_mask = train.position.eq(position).to_numpy()
        want = test_positions == position
        if not want.any():
            continue
        if fit_mask.sum() < MIN_RESIDUALS:
            fit_mask = np.ones(len(train), dtype=bool)
        coefficients = np.polyfit(
            np.log(train_rank[fit_mask]), train[y].to_numpy(float)[fit_mask], 1
        )
        residuals[fit_mask] = train[y].to_numpy(float)[fit_mask] - np.polyval(
            coefficients, np.log(train_rank[fit_mask])
        )
        centres[want] = np.polyval(
            coefficients, np.log(test.adp_rank.to_numpy(float)[want])
        )
        fit_index = np.flatnonzero(fit_mask)
        for i in np.flatnonzero(want):
            near = np.abs(train_rank[fit_index] - test.adp_rank.iloc[i]) <= RANK_WINDOW
            chosen = fit_index[near] if near.sum() >= MIN_RESIDUALS else fit_index
            pools[i] = chosen
    return centres, pools, residuals


def draw(centres, pools, residuals, draws, rng, shared=None):
    """Predictive samples. ``shared`` reuses another response's chosen rows."""
    picks = shared
    if picks is None:
        picks = np.stack(
            [rng.choice(pool, size=draws, replace=True) for pool in pools]
        )
    return centres[:, None] + residuals[picks], picks


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--holdouts", type=int, nargs="+", default=[2022, 2023, 2024])
    parser.add_argument("--scoring", default="ppr")
    parser.add_argument(
        "--cache-dir", type=Path, default=Path(".cache/ffmodel-wf-2025-adp2")
    )
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    output = args.output or Path("scripts/validation_runs/composition_cost.json")

    pool = _adp_only.prepare(args.cache_dir, args.scoring)
    pool["opportunity"] = opportunity(pool)
    pool = pool[pool.opportunity.gt(0)].copy()
    pool["rate"] = pool.points / pool.opportunity

    totals: dict[str, list[tuple[float, float, int]]] = {}
    correlations = []
    for holdout in args.holdouts:
        train, test = pool[pool.season.lt(holdout)], pool[pool.season.eq(holdout)]
        if test.empty:
            continue
        observed = test.points.to_numpy(float)

        pts_c, pts_p, pts_r = curve(train, test, "points")
        opp_c, opp_p, opp_r = curve(train, test, "opportunity")
        rate_c, rate_p, rate_r = curve(train, test, "rate")
        correlations.append(float(np.corrcoef(opp_r, rate_r)[0, 1]))

        rng = np.random.default_rng(50000 + holdout)
        direct, _ = draw(pts_c, pts_p, pts_r, args.draws, rng)

        rng = np.random.default_rng(50000 + holdout)
        opp_a, _ = draw(opp_c, opp_p, opp_r, args.draws, rng)
        rate_a, _ = draw(rate_c, rate_p, rate_r, args.draws, rng)

        rng = np.random.default_rng(50000 + holdout)
        opp_b, picks = draw(opp_c, opp_p, opp_r, args.draws, rng)
        # Same training player for both factors, so their joint behaviour is
        # whatever the data shows rather than an independence assumption.
        rate_b, _ = draw(rate_c, rate_p, rate_r, args.draws, rng, shared=picks)

        for label, samples in (
            ("direct", direct),
            ("composed, independent", np.maximum(opp_a, 0.0) * np.maximum(rate_a, 0.0)),
            ("composed, dependence kept", np.maximum(opp_b, 0.0) * np.maximum(rate_b, 0.0)),
        ):
            r = score(observed, np.maximum(samples, 0.0))
            totals.setdefault(label, []).append((r["mae"], r["crps"], r["n"]))
            totals.setdefault(f"_bias_{label}", []).append(
                (r["bias"], r["coverage_95"], r["n"])
            )

    print(
        f"\nONE ESTIMATOR, ONE INFORMATION SET, THREE ARCHITECTURES "
        f"({args.scoring.upper()}, drafted pool)\n"
    )
    print(f"  {'route to points':28s} {'MAE':>8s} {'CRPS':>8s} {'bias':>8s} {'cov95':>7s}")
    pooled: dict[str, dict[str, float]] = {}
    for label in ("direct", "composed, independent", "composed, dependence kept"):
        vals, extra = totals[label], totals[f"_bias_{label}"]
        weight = sum(v[2] for v in vals)
        pooled[label] = {
            "mae": sum(v[0] * v[2] for v in vals) / weight,
            "crps": sum(v[1] * v[2] for v in vals) / weight,
            "bias": sum(v[0] * v[2] for v in extra) / weight,
            "coverage_95": sum(v[1] * v[2] for v in extra) / weight,
        }
        p = pooled[label]
        print(
            f"  {label:28s} {p['mae']:>8.2f} {p['crps']:>8.2f} {p['bias']:>+8.2f} "
            f"{p['coverage_95']:>7.3f}"
        )

    d, i, k = (
        pooled["direct"],
        pooled["composed, independent"],
        pooled["composed, dependence kept"],
    )
    print(
        f"\n  residual correlation between opportunity and rate: "
        f"{np.mean(correlations):+.3f}"
    )
    print(
        f"  composing costs {(i['mae'] - d['mae']) / d['mae']:+.1%} MAE independently, "
        f"{(k['mae'] - d['mae']) / d['mae']:+.1%} with dependence kept"
    )
    print(f"  the model's own drafted-pool MAE is 58.90 against this direct arm's "
          f"{d['mae']:.2f}")

    pooled["_residual_correlation"] = float(np.mean(correlations))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(pooled, indent=2, sort_keys=True), encoding="utf-8")
    print(f"\nwrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
