"""What does the model add over reading the draft board?

The ablation showed ADP is worth something as a covariate. That raises the
sharper question: how much of the model's accuracy on drafted players was ever
more than ADP in the first place? A projection that cannot beat the draft board
it was built alongside is an expensive way to restate consensus.

The baseline here is deliberately the strongest honest version, because beating
a weak one proves nothing:

* **It gets a distribution, not a point.** A point forecast cannot be scored on
  CRPS or coverage, and comparing the model's MAE against a point baseline's MAE
  while quietly dropping the other two metrics would flatter whichever side is
  better at the mean. Predictive draws come from resampling the curve's own
  residuals, so the spread is whatever the data says it is rather than a normal
  assumption bolted on.
* **It knows position.** "RB7" is part of what a drafter reads off the board, and
  a quarterback taken 40th overall is a different proposition from a running back
  taken 40th. Curves are fitted per position.
* **It is fitted out of sample.** The rank-to-points relationship comes from
  seasons strictly before the holdout, and the residual pool with it. Fitting on
  the holdout would let the baseline see the answer, which would make it lose
  for the wrong reason.

Residuals are drawn from players at nearby ranks rather than pooled across the
whole board, because spread scales with level: the gap between a hit and a bust
at pick 5 is not the gap at pick 250.

Scored on exactly the rows the walk-forward scored, using the frames' own
``adp_drafted`` column, so the populations are identical by construction rather
than by a second join that might disagree.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from ffmodel.evaluation.efficiency_posterior import observed_scoring_rows
from ffmodel.evaluation.metrics import empirical_crps, interval_coverage
from ffmodel.simulation.scoring import fantasy_points

MODEL_POSITIONS = ("QB", "RB", "WR", "TE")
# Residuals are drawn from players within this many ranks. Wide enough to have a
# pool at every rank, narrow enough that a bust at pick 5 is not modelled with
# the spread of pick 250.
RANK_WINDOW = 60
MIN_RESIDUALS = 40


def prepare(cache_dir: Path, scoring: str) -> pd.DataFrame:
    rows = pd.read_pickle(cache_dir / "player_rows.pkl")
    missing = {"adp_drafted", "adp_rank"} - set(rows.columns)
    if missing:
        raise SystemExit(
            f"{cache_dir} has no {sorted(missing)}. Build it with "
            "scripts/augment_cache_features.py --feature market-adp"
        )
    rows = rows[rows.position.isin(MODEL_POSITIONS)].copy().reset_index(drop=True)
    rows["points"] = np.nan
    for _, block in rows.groupby("season"):
        rows.loc[block.index, "points"] = fantasy_points(
            observed_scoring_rows(block.reset_index(drop=True)), scoring
        ).to_numpy()
    named = (
        pd.to_numeric(rows.get("is_replacement_player"), errors="coerce").fillna(0).ne(1)
    )
    drafted = pd.to_numeric(rows["adp_drafted"], errors="coerce").eq(1)
    return rows[named & drafted & rows.points.notna()].copy()


def project_pooled(
    train: pd.DataFrame, test: pd.DataFrame, draws: int, seed: int
) -> np.ndarray:
    """One curve for the whole board, ignoring position.

    Kept because ``benchmark_adp_baselines.py`` used this shape and it is the
    reason that script reported the model beating the market. A single curve
    cannot know that a quarterback taken 40th is the fourth at his position
    while a running back taken 40th is the fifteenth, so it misses by position
    in a way no drafter would. Reporting both makes the earlier number
    reproducible instead of leaving two of our own figures in disagreement.
    """
    rng = np.random.default_rng(seed)
    coefficients = np.polyfit(np.log(train.adp_rank), train.points, 1)
    residuals = train.points.to_numpy(float) - np.polyval(
        coefficients, np.log(train.adp_rank.to_numpy(float))
    )
    centre = np.polyval(coefficients, np.log(test.adp_rank.to_numpy(float)))
    drawn = centre[:, None] + rng.choice(residuals, size=(len(test), draws))
    return np.maximum(drawn, 0.0)


def project(train: pd.DataFrame, test: pd.DataFrame, draws: int, seed: int) -> np.ndarray:
    """Predictive samples for each test row from rank alone.

    Per-position log fit for the centre, empirical residuals from nearby ranks
    for the spread. Both come only from ``train``.
    """
    rng = np.random.default_rng(seed)
    samples = np.zeros((len(test), draws), dtype=float)
    for position in MODEL_POSITIONS:
        fit = train[train.position.eq(position)]
        want = test.position.eq(position).to_numpy()
        if not want.any():
            continue
        if len(fit) < MIN_RESIDUALS:
            # Not enough history to fit this position; fall back to the whole
            # board rather than inventing a curve from a handful of rows.
            fit = train
        coefficients = np.polyfit(np.log(fit.adp_rank), fit.points, 1)
        residuals = fit.points.to_numpy() - np.polyval(
            coefficients, np.log(fit.adp_rank.to_numpy())
        )
        fit_ranks = fit.adp_rank.to_numpy()
        block = test[want]
        centre = np.polyval(coefficients, np.log(block.adp_rank.to_numpy()))
        drawn = np.zeros((len(block), draws), dtype=float)
        for i, rank in enumerate(block.adp_rank.to_numpy()):
            near = np.abs(fit_ranks - rank) <= RANK_WINDOW
            pool = residuals[near] if near.sum() >= MIN_RESIDUALS else residuals
            drawn[i] = centre[i] + rng.choice(pool, size=draws, replace=True)
        # A season total cannot be negative, and the curve is linear in log rank
        # so it will go under zero deep on the board.
        samples[want] = np.maximum(drawn, 0.0)
    return samples


def score(observed: np.ndarray, samples: np.ndarray) -> dict[str, float]:
    mean = samples.mean(axis=1)
    return {
        "n": int(len(observed)),
        "observed_mean": float(observed.mean()),
        "mae": float(np.abs(mean - observed).mean()),
        "rmse": float(np.sqrt(np.mean((mean - observed) ** 2))),
        "bias": float((mean - observed).mean()),
        "crps": float(empirical_crps(observed, samples).mean()),
        "coverage_80": float(interval_coverage(observed, samples, 0.80)["coverage"]),
        "coverage_95": float(interval_coverage(observed, samples, 0.95)["coverage"]),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--holdouts", type=int, nargs="+", default=[2022, 2023, 2024])
    parser.add_argument("--scoring", default="ppr")
    parser.add_argument(
        "--cache-dir", type=Path, default=Path(".cache/ffmodel-wf-2025-adp")
    )
    parser.add_argument("--runs", type=Path, default=Path("scripts/validation_runs"))
    parser.add_argument("--baseline", default="adpoff")
    parser.add_argument("--candidate", default="adpon")
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    output = args.output or args.runs / "adp_only_benchmark.json"

    pool = prepare(args.cache_dir, args.scoring)
    key = f"{args.scoring}_drafted"
    arms = {
        name: json.loads((args.runs / f"scoring_{name}.json").read_text())
        for name in (args.baseline, args.candidate)
    }

    folds: list[dict[str, object]] = []
    for holdout in args.holdouts:
        train = pool[pool.season.lt(holdout)]
        test = pool[pool.season.eq(holdout)]
        if test.empty or len(train) < MIN_RESIDUALS:
            continue
        observed = test.points.to_numpy(dtype=float)
        samples = project(train, test, args.draws, seed=1000 + holdout)
        entry: dict[str, object] = {
            "holdout": holdout,
            "ADP only": score(observed, samples),
            "ADP, one curve": score(
                observed, project_pooled(train, test, args.draws, seed=2000 + holdout)
            ),
        }
        for name, report in arms.items():
            if str(holdout) in report and key in report[str(holdout)]:
                entry[name] = {
                    k: v for k, v in report[str(holdout)][key].items() if k != "scoring"
                }
        folds.append(entry)

    if not folds:
        raise SystemExit("no holdout had both a fitted curve and a scored model")

    labels = ["ADP only", "ADP, one curve", args.baseline, args.candidate]
    pretty = {
        "ADP only": "ADP, per position",
        "ADP, one curve": "ADP, one curve",
        args.baseline: "model, no ADP",
        args.candidate: "model, with ADP",
    }
    print(f"\n{args.scoring.upper()} SEASON TOTALS ON THE DRAFTED POOL\n")
    print(
        f"  {'holdout':>8s} {'projection':16s} {'n':>5s} {'MAE':>8s} {'RMSE':>8s} "
        f"{'bias':>8s} {'CRPS':>8s} {'cov80':>7s} {'cov95':>7s}"
    )
    for entry in folds:
        for label in labels:
            row = entry.get(label)
            if row is None:
                continue
            print(
                f"  {entry['holdout']:>8d} {pretty[label]:16s} {row['n']:>5d} "
                f"{row['mae']:>8.2f} {row['rmse']:>8.2f} "
                f"{row.get('bias', float('nan')):>8.2f} {row['crps']:>8.2f} "
                f"{row['coverage_80']:>7.3f} {row['coverage_95']:>7.3f}"
            )
        print()

    pooled: dict[str, dict[str, float]] = {}
    for label in labels:
        present = [e for e in folds if label in e]
        if len(present) != len(folds):
            continue
        weight = sum(e[label]["n"] for e in present)
        pooled[label] = {
            metric: sum(e[label][metric] * e[label]["n"] for e in present) / weight
            for metric in ("mae", "rmse", "crps")
        }
    print(f"  {'projection':16s} {'MAE':>8s} {'RMSE':>8s} {'CRPS':>8s}   vs ADP only")
    reference = pooled.get("ADP only")
    for label in labels:
        row = pooled.get(label)
        if row is None:
            continue
        against = (
            ""
            if label == "ADP only"
            else f"   MAE {(row['mae'] - reference['mae']) / reference['mae']:+.1%}, "
            f"CRPS {(row['crps'] - reference['crps']) / reference['crps']:+.1%}"
        )
        print(
            f"  {pretty[label]:16s} {row['mae']:>8.2f} {row['rmse']:>8.2f} "
            f"{row['crps']:>8.2f}{against}"
        )

    payload = {
        "scoring": args.scoring,
        "holdouts": [e["holdout"] for e in folds],
        "rank_window": RANK_WINDOW,
        "draws": args.draws,
        "folds": folds,
        "pooled": pooled,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"\nwrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
