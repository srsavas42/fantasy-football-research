"""Walk-forward screen for splitting the reserve flag and flooring PUP.

Two changes are on trial and they are separable, so they are scored separately
as well as together. A run that only reported the combination could not say
which half earned the result, and one of them could be paying for the other.

``truncation`` enforces the mandatory minimum a week-1 PUP or non-football-injury
placement carries -- four games from 2022, six before. It is a cap on the
outcome, never a subtraction: the reserve coefficient is already fitted on
players in exactly this position, so removing games as well would charge one
injury twice. It can only remove predictive mass the rules forbid, so it should
not hurt; if it does, that is evidence the fitted mean was leaning on the
forbidden region.

``reserve-split`` gives injured reserve, PUP and non-football-injury their own
deviations from the pooled flag. On 2021-2025 week-1 placements those
populations miss 16.2, 13.5 and 15.8 games, and injured reserve supplies 266 of
the 683 pooled rows, so one shared coefficient is fitted mostly on injured
reserve and applied to everyone.

Scored on the whole holdout and on the reserve population the change is about,
because a change touching 4% of rows can be swamped in a pooled average and
still be the right change.

    python scripts/validate_pup_availability.py --holdouts 2023 2024 2025 \
        --cache-dir .cache/ffmodel-2026 --report-json reports/pup.json
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from ffmodel.evaluation.metrics import point_and_distribution
from ffmodel.models.season_availability import (
    RESERVE_KIND_FEATURES,
    SeasonAvailabilityModel,
)

ARMS = {
    "baseline": (False, False),
    "truncation": (True, False),
    "reserve-split": (False, True),
    "both": (True, True),
}

# The package's materiality floor; a smaller move is not claimed either way.
MATERIAL = 0.0025


def _evaluate(train, test, *, truncate, split, fit_kwargs, seed):
    frame = test.copy()
    import gc
    if not truncate:
        # The baseline must not see the floor at all, and zeroing the column is
        # how the model is told there is none.
        frame["mandatory_missed_games"] = 0.0
    model = SeasonAvailabilityModel(
        extra_features=RESERVE_KIND_FEATURES if split else ()
    ).fit(train, **fit_kwargs)
    prediction = model.predict_samples(frame, seed=seed)
    rows = prediction.rows
    observed = pd.to_numeric(
        rows["observed_availability"], errors="coerce"
    ).fillna(0.0).to_numpy(dtype=float)
    samples = prediction.availability

    out = {
        "overall": point_and_distribution(observed, samples),
        "features": len(model.feature_names),
    }
    groups = {
        "reserve": pd.to_numeric(rows.get("roster_reserve"), errors="coerce").fillna(0).gt(0),
        "pup_or_nfi": (
            pd.to_numeric(rows.get("roster_pup"), errors="coerce").fillna(0).gt(0)
            | pd.to_numeric(rows.get("roster_nfi"), errors="coerce").fillna(0).gt(0)
        ),
        "injured_reserve": pd.to_numeric(
            rows.get("roster_injured_reserve"), errors="coerce"
        ).fillna(0).gt(0),
    }
    for name, mask in groups.items():
        mask = mask.to_numpy()
        if mask.sum() >= 5:
            out[name] = point_and_distribution(observed[mask], samples[mask])
    # Twelve fits in one process, each holding a posterior and a draw array.
    # Without this the run dies partway through with no traceback, which reads
    # as a crash rather than as running out of room.
    del model, prediction, samples, frame
    gc.collect()
    return out


def _report(report: dict, args) -> int:
    """Pool the folds that are present, print the arms table, and write it.

    Tolerant of a missing fold on purpose: the folds are run as separate
    processes, so a partial set is a normal intermediate state rather than an
    error, and the fold count printed beside each average says how many
    actually contributed.
    """
    folds = report["folds"]
    holdouts = [h for h in args.holdouts if str(h) in folds]
    print(f"\n{'=' * 78}\npooled vs baseline, by population\n{'=' * 78}")
    pooled: dict[str, object] = {}
    for population in ("overall", "reserve", "pup_or_nfi", "injured_reserve"):
        rows = []
        for name in ARMS:
            values = [
                folds[str(h)][name][population]
                for h in holdouts
                if population in folds[str(h)][name]
            ]
            if not values:
                continue
            rows.append(
                {
                    "arm": name,
                    "n": int(np.mean([v["n"] for v in values])),
                    **{
                        metric: float(np.mean([v[metric] for v in values]))
                        for metric in ("mae", "rmse", "crps", "coverage_80")
                    },
                }
            )
        if not rows:
            continue
        table = pd.DataFrame(rows).set_index("arm")
        base = table.loc["baseline"]
        for metric in ("mae", "crps"):
            table[f"{metric}_delta"] = (table[metric] - base[metric]) / base[metric]
        # Won every fold, not just on average: a pooled win carried by one
        # holdout is what this repo's gate exists to reject.
        for name in ARMS:
            scored = [h for h in holdouts if population in folds[str(h)][name]]
            wins = sum(
                folds[str(h)][name][population]["crps"]
                < folds[str(h)]["baseline"][population]["crps"]
                for h in scored
            )
            table.loc[name, "crps_folds_won"] = f"{wins}/{len(scored)}"
        print(f"\n-- {population} (n~{int(table['n'].iloc[0])}) --")
        print(
            table[
                ["mae", "crps", "coverage_80", "mae_delta", "crps_delta",
                 "crps_folds_won"]
            ].to_string(
                float_format=lambda v: f"{v:.5f}" if abs(v) > 1e-3 else f"{v:+.2%}"
            )
        )
        pooled[population] = table.reset_index().to_dict("records")
    report["pooled"] = pooled
    print(f"\nmateriality floor {MATERIAL:.2%}; a smaller move is not a result")
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(json.dumps(report, indent=2, default=str), "utf-8")
    print(f"wrote {args.report_json}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--holdouts", type=int, nargs="+", default=[2023, 2024, 2025])
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/ffmodel-2026"))
    parser.add_argument("--draws", type=int, default=400)
    parser.add_argument("--tune", type=int, default=400)
    parser.add_argument("--chains", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report-json", type=Path, default=Path("reports/pup.json"))
    parser.add_argument(
        "--merge",
        type=Path,
        nargs="+",
        default=None,
        help="pool per-fold reports from earlier single-holdout runs instead of "
        "fitting. Twelve fits in one process exhausts this container, so each "
        "fold is run separately and combined here.",
    )
    args = parser.parse_args(argv)

    if args.merge:
        folds: dict = {}
        for path in args.merge:
            folds.update(json.loads(path.read_text("utf-8"))["folds"])
        args.holdouts = sorted(int(key) for key in folds)
        report: dict[str, object] = {"holdouts": args.holdouts, "folds": folds}
        return _report(report, args)

    player_rows = pd.read_pickle(args.cache_dir / "player_rows.pkl")
    required = [*RESERVE_KIND_FEATURES, "mandatory_missed_games"]
    missing = [c for c in required if c not in player_rows]
    if missing:
        raise SystemExit(
            f"{args.cache_dir} is missing {missing}. These frames predate the "
            "reserve split, and every arm would silently score as the baseline. "
            "Rebuild with scripts/build_projection_cache.py"
        )

    # The cache carries 300+ columns and the availability layer reads about
    # twenty. Dragging the rest through twelve fits is most of this script's
    # memory and none of its result.
    from ffmodel.models.season_availability import AVAILABILITY_FEATURES

    keep = dict.fromkeys(
        [
            "season", "team", "player_key", "player_name", "position",
            "games", "team_games", "snap_games", "observed_availability",
            "snap_availability", "suspended_games", "mandatory_missed_games",
            "roster_suspended", "is_replacement_player",
            *AVAILABILITY_FEATURES, *RESERVE_KIND_FEATURES,
        ]
    )
    player_rows = player_rows[[c for c in keep if c in player_rows.columns]].copy()

    fit_kwargs = {"draws": args.draws, "tune": args.tune, "chains": args.chains}
    report: dict[str, object] = {"holdouts": args.holdouts, "folds": {}}

    for holdout in args.holdouts:
        train = player_rows[player_rows.season.lt(holdout)].copy()
        test = player_rows[player_rows.season.eq(holdout)].copy()
        if train.empty or test.empty:
            raise SystemExit(f"holdout {holdout} has no train or test rows")
        fold = {}
        for name, (truncate, split) in ARMS.items():
            fold[name] = _evaluate(
                train, test, truncate=truncate, split=split,
                fit_kwargs=fit_kwargs, seed=args.seed,
            )
            block = fold[name]["overall"]
            print(
                f"{holdout} {name:14s} CRPS {block['crps']:.5f}  "
                f"MAE {block['mae']:.5f}  cov80 {block['coverage_80']:.3f}",
                flush=True,
            )
        report["folds"][str(holdout)] = fold

    return _report(report, args)


if __name__ == "__main__":
    raise SystemExit(main())
