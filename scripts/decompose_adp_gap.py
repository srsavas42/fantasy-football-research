"""Why does a rank curve beat a model that contains the rank?

The model loses to an ADP-only estimator by 8.1% MAE and 14.4% CRPS on the
drafted pool, and it loses while carrying ``adp_log_rank`` in four of its
submodels. Something between the column and the prediction is destroying the
signal. This narrows down what.

The shipped pipeline is a generalised linear model in its features: every
submodel forms a linear predictor from standardised, SVD-rotated columns and
passes it through a link. Position enters as an additive effect. So the model
can represent "quarterbacks score more than running backs" and "points fall
with draft rank", but it cannot represent "points fall with draft rank *at a
different rate* for quarterbacks than for running backs" — that is an
interaction, and there is no term for it.

The ADP-only baseline fits a separate curve per position, so it has exactly
that interaction for free. If the interaction is the gap, then rebuilding the
baseline the way the model sees the world should collapse its advantage.

The ladder, each rung adding one thing, all fitted on seasons before the
holdout and scored on the rows the walk-forward scored:

1. **intercept only** — no rank at all. The floor.
2. **one curve** — a single log-rank slope for the whole board, position
   ignored. What the earlier benchmark used.
3. **shared slope, position intercepts** — the model's functional form: one
   rank effect, position shifts the level.
4. **per-position slopes** — the interaction. The baseline that beat the model.
5. **per-position, no functional form** — a local mean over nearby ranks within
   position. Tests whether anything is left after the interaction that a log
   curve still misses.

Rungs 3 and 4 are the ones that matter. If most of the gap opens between them,
the fix is an interaction term rather than a different model family.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from ffmodel.evaluation.metrics import empirical_crps, interval_coverage

import importlib.util

_spec = importlib.util.spec_from_file_location(
    "_adp_only", Path(__file__).with_name("benchmark_adp_only.py")
)
_adp_only = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_adp_only)

MODEL_POSITIONS = _adp_only.MODEL_POSITIONS
RANK_WINDOW = _adp_only.RANK_WINDOW
MIN_RESIDUALS = _adp_only.MIN_RESIDUALS
score = _adp_only.score


def _draws(centre: np.ndarray, residuals: np.ndarray, draws: int, rng) -> np.ndarray:
    """Predictive samples: a centre plus resampled residuals, floored at zero."""
    drawn = centre[:, None] + rng.choice(residuals, size=(len(centre), draws))
    return np.maximum(drawn, 0.0)


def intercept_only(train, test, draws, rng):
    centre = np.full(len(test), train.points.mean())
    return _draws(centre, train.points.to_numpy(float) - train.points.mean(), draws, rng)


def one_curve(train, test, draws, rng):
    c = np.polyfit(np.log(train.adp_rank), train.points, 1)
    fitted = np.polyval(c, np.log(train.adp_rank.to_numpy(float)))
    centre = np.polyval(c, np.log(test.adp_rank.to_numpy(float)))
    return _draws(centre, train.points.to_numpy(float) - fitted, draws, rng)


def shared_slope(train, test, draws, rng):
    """One rank slope, position shifts the intercept. The model's form.

    Fitted as least squares on [log rank, position dummies] so the slope is
    common and only the level moves, which is exactly what an additive position
    effect in a linear predictor can express.
    """
    def design(frame):
        columns = [np.log(frame.adp_rank.to_numpy(float)), np.ones(len(frame))]
        for position in MODEL_POSITIONS[:-1]:
            columns.append(frame.position.eq(position).to_numpy(float))
        return np.column_stack(columns)

    beta, *_ = np.linalg.lstsq(design(train), train.points.to_numpy(float), rcond=None)
    fitted = design(train) @ beta
    return _draws(
        design(test) @ beta, train.points.to_numpy(float) - fitted, draws, rng
    )


def per_position_slope(train, test, draws, rng, seed):
    return _adp_only.project(train, test, draws, seed)


def shared_slope_all_rostered(train_all, test, draws, rng):
    """Rung 3's form, but fitted on every rostered player, as the model is.

    The obvious objection to the whole comparison: the rank curves are fitted on
    drafted players and scored on drafted players, while the model is fitted on
    the full roster, most of which is fringe. A baseline specialised to the
    population it is tested on has an unearned edge.

    This removes that edge. The training set is every rostered row, undrafted
    players included at their sentinel rank, which is exactly the mixture the
    model sees. If it still beats the model, specialisation was not the
    explanation.
    """
    return shared_slope(train_all, test, draws, rng)


def per_position_local(train, test, draws, rng):
    """No functional form: the mean of nearby same-position players.

    A k-nearest-rank regressor. Whatever a log curve leaves on the table within
    a position, this picks up; if it scores no better, the log shape is the
    right one and the remaining error is not about the shape of the rank curve.
    """
    samples = np.zeros((len(test), draws), dtype=float)
    for position in MODEL_POSITIONS:
        fit = train[train.position.eq(position)]
        want = test.position.eq(position).to_numpy()
        if not want.any():
            continue
        if len(fit) < MIN_RESIDUALS:
            fit = train
        fit_ranks = fit.adp_rank.to_numpy(float)
        fit_points = fit.points.to_numpy(float)

        def local(ranks):
            out = np.empty(len(ranks))
            for i, rank in enumerate(ranks):
                near = np.abs(fit_ranks - rank) <= RANK_WINDOW
                out[i] = fit_points[near].mean() if near.sum() >= 10 else fit_points.mean()
            return out

        residuals = fit_points - local(fit_ranks)
        block = test[want]
        centre = local(block.adp_rank.to_numpy(float))
        drawn = np.zeros((len(block), draws))
        for i, rank in enumerate(block.adp_rank.to_numpy(float)):
            near = np.abs(fit_ranks - rank) <= RANK_WINDOW
            pool = residuals[near] if near.sum() >= MIN_RESIDUALS else residuals
            drawn[i] = centre[i] + rng.choice(pool, size=draws, replace=True)
        samples[want] = np.maximum(drawn, 0.0)
    return samples


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--holdouts", type=int, nargs="+", default=[2022, 2023, 2024])
    parser.add_argument("--scoring", default="ppr")
    parser.add_argument(
        "--cache-dir", type=Path, default=Path(".cache/ffmodel-wf-2025-adp")
    )
    parser.add_argument("--runs", type=Path, default=Path("scripts/validation_runs"))
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    output = args.output or args.runs / "adp_gap_decomposition.json"

    pool = _adp_only.prepare(args.cache_dir, args.scoring)
    everyone = _adp_only.prepare(args.cache_dir, args.scoring, drafted_only=False)
    key = f"{args.scoring}_drafted"
    models = {
        label: json.loads((args.runs / f"scoring_{name}.json").read_text())
        for label, name in (("model, no ADP", "adpoff"), ("model, with ADP", "adpon"))
    }

    rungs = [
        ("1. intercept only", lambda tr, te, d, r, s: intercept_only(tr, te, d, r)),
        ("2. one curve", lambda tr, te, d, r, s: one_curve(tr, te, d, r)),
        ("3. shared slope, pos intercepts", lambda tr, te, d, r, s: shared_slope(tr, te, d, r)),
        ("4. per-position slopes", per_position_slope),
        ("5. per-position, no form", lambda tr, te, d, r, s: per_position_local(tr, te, d, r)),
    ]
    control = "3b. rung 3, fitted on all rostered"

    totals: dict[str, list[tuple[float, float, int]]] = {}
    for holdout in args.holdouts:
        train, test = pool[pool.season.lt(holdout)], pool[pool.season.eq(holdout)]
        if test.empty:
            continue
        observed = test.points.to_numpy(dtype=float)
        for label, build in rungs:
            rng = np.random.default_rng(90000 + holdout)
            result = score(observed, build(train, test, args.draws, rng, 1000 + holdout))
            totals.setdefault(label, []).append(
                (result["mae"], result["crps"], result["n"])
            )
        rng = np.random.default_rng(90000 + holdout)
        totals.setdefault(control, []).append(
            (lambda r: (r["mae"], r["crps"], r["n"]))(
                score(
                    observed,
                    shared_slope_all_rostered(
                        everyone[everyone.season.lt(holdout)], test, args.draws, rng
                    ),
                )
            )
        )
        for label, report in models.items():
            row = report.get(str(holdout), {}).get(key)
            if row:
                totals.setdefault(label, []).append((row["mae"], row["crps"], row["n"]))

    print(f"\n{args.scoring.upper()} SEASON TOTALS, DRAFTED POOL, POOLED OVER "
          f"{args.holdouts}\n")
    print(f"  {'projection':34s} {'MAE':>8s} {'CRPS':>8s}")
    pooled: dict[str, dict[str, float]] = {}
    order = [label for label, _ in rungs] + [control] + list(models)
    for label in order:
        vals = totals.get(label)
        if not vals or len(vals) != len(args.holdouts):
            continue
        weight = sum(v[2] for v in vals)
        pooled[label] = {
            "mae": sum(v[0] * v[2] for v in vals) / weight,
            "crps": sum(v[1] * v[2] for v in vals) / weight,
        }
        print(f"  {label:34s} {pooled[label]['mae']:>8.2f} {pooled[label]['crps']:>8.2f}")

    a = pooled.get("3. shared slope, pos intercepts")
    b = pooled.get("4. per-position slopes")
    m = pooled.get("model, with ADP")
    if a and b and m:
        interaction = a["mae"] - b["mae"]
        gap = m["mae"] - b["mae"]
        print(
            f"\n  the position-by-rank interaction alone is worth "
            f"{interaction:.2f} MAE ({interaction / a['mae']:.1%})"
        )
        print(f"  the model's whole deficit against rung 4 is {gap:.2f} MAE")
        print(
            f"  so the interaction accounts for {min(interaction / gap, 1.0):.0%} "
            "of what the model is missing"
            if gap > 0
            else "  the model is not behind rung 4"
        )
        pooled["_interaction_mae"] = float(interaction)
        pooled["_model_deficit_mae"] = float(gap)

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(pooled, indent=2, sort_keys=True), encoding="utf-8")
    print(f"\nwrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
