"""Which terms recover drafted-pool accuracy while training on everybody?

Rung 3b of the gap decomposition fitted one log-rank slope over every rostered
player and lost 3.6 CRPS points against the same form fitted on drafted players
only. That was read as a population problem — the model is trained on a
fringe-heavy roster and judged on the drafted half — and it suggested
reweighting the likelihood.

It may be simpler than that. Rung 3b had a single slope for the whole roster,
so it could not say that the rank-to-points relationship *differs* for a player
the market declined to rank. The pipeline has the same limitation in a milder
form: it carries ``adp_drafted`` as a main effect, which shifts the level, but
nothing that lets the slope change with it.

If adding that interaction to an all-rostered fit recovers what the
drafted-only fit achieved, then there is no population problem — there is a
missing term, and the fix is three columns rather than a reweighted likelihood
across six submodels.

Every rung trains on **all rostered players** and scores on the drafted pool, so
the training distribution is held at the model's throughout and only the terms
change. The drafted-only fit is carried along as the target to reach.
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
score = _adp_only.score


def design(frame: pd.DataFrame, terms: set[str]) -> np.ndarray:
    """Least-squares design for a named set of terms.

    ``drafted`` is the market's own flag, not a rank threshold, so an unranked
    player is identified the same way the pipeline identifies him.
    """
    rank = np.log(pd.to_numeric(frame.adp_rank, errors="coerce").to_numpy(float))
    drafted = pd.to_numeric(frame.adp_drafted, errors="coerce").to_numpy(float)
    columns = [np.ones(len(frame))]
    if "rank" in terms:
        columns.append(rank)
    if "position" in terms:
        for position in MODEL_POSITIONS[:-1]:
            columns.append(frame.position.eq(position).to_numpy(float))
    if "drafted" in terms:
        columns.append(drafted)
    if "drafted_x_rank" in terms:
        columns.append(drafted * rank)
    if "position_x_rank" in terms:
        for position in MODEL_POSITIONS[:-1]:
            columns.append(frame.position.eq(position).to_numpy(float) * rank)
    if "drafted_x_position" in terms:
        for position in MODEL_POSITIONS[:-1]:
            columns.append(frame.position.eq(position).to_numpy(float) * drafted)
    return np.column_stack(columns)


def fit_and_draw(train, test, terms, draws, rng):
    X, y = design(train, terms), train.points.to_numpy(float)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    residuals = y - X @ beta
    centre = design(test, terms) @ beta
    return np.maximum(centre[:, None] + rng.choice(residuals, size=(len(test), draws)), 0.0)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--holdouts", type=int, nargs="+", default=[2022, 2023, 2024])
    parser.add_argument("--scoring", default="ppr")
    parser.add_argument(
        "--cache-dir", type=Path, default=Path(".cache/ffmodel-wf-2025-adp")
    )
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    output = args.output or Path("scripts/validation_runs/adp_term_probe.json")

    drafted_pool = _adp_only.prepare(args.cache_dir, args.scoring)
    everyone = _adp_only.prepare(args.cache_dir, args.scoring, drafted_only=False)

    ladder = [
        ("rank + position (rung 3b)", {"rank", "position"}),
        ("  + drafted", {"rank", "position", "drafted"}),
        ("  + drafted x rank", {"rank", "position", "drafted", "drafted_x_rank"}),
        (
            "  + position x rank",
            {"rank", "position", "drafted", "drafted_x_rank", "position_x_rank"},
        ),
        (
            "  + drafted x position",
            {
                "rank",
                "position",
                "drafted",
                "drafted_x_rank",
                "position_x_rank",
                "drafted_x_position",
            },
        ),
    ]

    totals: dict[str, list[tuple[float, float, int]]] = {}
    for holdout in args.holdouts:
        test = drafted_pool[drafted_pool.season.eq(holdout)]
        if test.empty:
            continue
        observed = test.points.to_numpy(float)
        train_all = everyone[everyone.season.lt(holdout)]
        for label, terms in ladder:
            rng = np.random.default_rng(70000 + holdout)
            r = score(observed, fit_and_draw(train_all, test, terms, args.draws, rng))
            totals.setdefault(label, []).append((r["mae"], r["crps"], r["n"]))
        # The target: same form, trained only on drafted players.
        rng = np.random.default_rng(70000 + holdout)
        train_drafted = drafted_pool[drafted_pool.season.lt(holdout)]
        r = score(
            observed,
            fit_and_draw(train_drafted, test, {"rank", "position"}, args.draws, rng),
        )
        totals.setdefault("target: rung 3, drafted-only fit", []).append(
            (r["mae"], r["crps"], r["n"])
        )

    print(
        f"\nALL TRAINED ON EVERY ROSTERED PLAYER, SCORED ON THE DRAFTED POOL "
        f"({args.scoring.upper()}, {args.holdouts})\n"
    )
    print(f"  {'terms':34s} {'MAE':>8s} {'CRPS':>8s}")
    pooled: dict[str, dict[str, float]] = {}
    for label, _ in ladder:
        vals = totals[label]
        weight = sum(v[2] for v in vals)
        pooled[label] = {
            "mae": sum(v[0] * v[2] for v in vals) / weight,
            "crps": sum(v[1] * v[2] for v in vals) / weight,
        }
        print(f"  {label:34s} {pooled[label]['mae']:>8.2f} {pooled[label]['crps']:>8.2f}")
    label = "target: rung 3, drafted-only fit"
    vals = totals[label]
    weight = sum(v[2] for v in vals)
    pooled[label] = {
        "mae": sum(v[0] * v[2] for v in vals) / weight,
        "crps": sum(v[1] * v[2] for v in vals) / weight,
    }
    print(f"\n  {label:34s} {pooled[label]['mae']:>8.2f} {pooled[label]['crps']:>8.2f}")
    print(f"  {'model, with ADP':34s} {58.90:>8.2f} {43.72:>8.2f}")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(pooled, indent=2, sort_keys=True), encoding="utf-8")
    print(f"\nwrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
