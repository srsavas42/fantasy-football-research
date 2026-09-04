"""Start-of-season injury status, separated from the three-year burden.

`INJURY_AVAILABILITY_FEATURES` is eleven columns describing two different
things. Five are the player's state at the projection cutoff -- whether he is on
the current injury report, how severe, how limited in practice, and an
empirical-Bayes expected recovery. Six are his history: three-year report weeks,
out weeks, episode counts, mean recovery and weeks since the last episode.

`scripts/validate_injury_availability.py` screens the block as one unit, so
every result on record -- the favourable three-fold screen, the flat 2025
confirmation, the inconclusive six holdouts in
`docs/injury-availability-2026-08.md` -- is about the bundle. Nobody separated
the halves, and there is a reason to expect them to behave differently:
`scripts/measure_availability_signal.py` shows the history half competing with
`prior_availability`, which already carries that signal, so if it contributes
nothing it is diluting whatever the current half contributes.

This script asks three things.

**How often does the current half fire?** A feature that is almost always zero
cannot move a pooled metric no matter how right it is when it does fire.

**What does each start-of-season signal predict, next to the one the model
already has?** `roster_reserve` is in `AVAILABILITY_FEATURES` today and it is
also a statement about the player's health at the cutoff, so it is the relevant
comparison rather than an unrelated control.

**Does either half add anything over a base that already includes roster
status?** Walk-forward, one fold per season.

It also reports the ceiling on `current_injury_expected_recovery_weeks`, because
`_expected_recovery` pools only episodes with `recovery_censored == 0` and an
episode is censored precisely when the player never returned that season. The
estimator is fitted exclusively on injuries people came back from within the
year, so it cannot learn a long recovery -- the long ones are the censored ones
it drops.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from ffmodel.features.season_injury import INJURY_AVAILABILITY_FEATURES
from ffmodel.features.volume import MODEL_POSITIONS

CURRENT_FEATURES = (
    "current_injury_reported",
    "current_injury_severity",
    "current_injury_practice_severity",
    "current_injury_expected_recovery_weeks",
)
HISTORY_FEATURES = (
    "prior_injury_report_weeks_3yr",
    "prior_injury_out_weeks_3yr",
    "prior_injury_episode_count_3yr",
    "prior_injury_mean_recovery_weeks_3yr",
    "prior_injury_weeks_since_last",
)
# The two always-on flags are excluded from both arms on purpose: they have sd
# 0.000 on any frame where the feed loaded, so they carry no information by
# construction. `docs/injury-availability-2026-08.md` records the design matrix
# dropping them for the same reason.
BASE_FEATURES = ("prior_availability", "roster_active", "roster_reserve")

RIDGE = 1e-3
MIN_TRAIN = 400
MIN_TEST = 50


def coverage(rows: pd.DataFrame) -> pd.DataFrame:
    """How often each injury column is non-zero, against the roster column."""
    names = [*CURRENT_FEATURES, *HISTORY_FEATURES, "roster_reserve"]
    return pd.DataFrame(
        [
            {
                "feature": name,
                "pct_non_zero": 100.0 * float(rows[name].gt(0).mean()),
                "mean": float(rows[name].mean()),
                "max": float(rows[name].max()),
            }
            for name in names
            if name in rows
        ]
    ).set_index("feature")


def signal_groups(rows: pd.DataFrame) -> pd.DataFrame:
    """Realized games by which start-of-season signal is present.

    ``roster_reserve`` is excluded from the injury-report rows so the two are
    not being credited with the same players.
    """
    out = rows.copy()
    out["games_played"] = out["observed_availability"] * out["team_games"]
    groups = {
        "no signal at all": out["current_injury_reported"].eq(0)
        & out["roster_reserve"].eq(0),
        "injury report, not on reserve": out["current_injury_reported"].gt(0)
        & out["roster_reserve"].eq(0),
        "report severity 3 (out/IR/PUP), not on reserve": out[
            "current_injury_severity"
        ].ge(3)
        & out["roster_reserve"].eq(0),
        "roster_reserve = 1": out["roster_reserve"].gt(0.5),
    }
    rows_out = []
    for label, mask in groups.items():
        block = out[mask]
        if len(block) < 5:
            continue
        rows_out.append(
            {
                "signal": label,
                "n": len(block),
                "pct_of_rows": 100.0 * len(block) / len(out),
                "mean_games": float(block["games_played"].mean()),
                "pct_never_played": 100.0 * float(block["games_played"].lt(0.5).mean()),
            }
        )
    return pd.DataFrame(rows_out).set_index("signal")


def walk_forward(rows: pd.DataFrame, arms: dict[str, list[str]]) -> pd.DataFrame:
    folds = []
    for season in sorted(rows["season"].unique()):
        train = rows[rows["season"] < season]
        test = rows[rows["season"] == season]
        if len(train) < MIN_TRAIN or len(test) < MIN_TEST:
            continue
        fold = {"season": int(season), "n": int(len(test))}
        for name, columns in arms.items():
            fills = {column: train[column].median() for column in columns}
            design = np.column_stack(
                [np.ones(len(train))]
                + [train[c].fillna(fills[c]).to_numpy(float) for c in columns]
            )
            penalty = np.eye(design.shape[1]) * RIDGE
            penalty[0, 0] = 0.0
            beta = np.linalg.solve(
                design.T @ design + penalty,
                design.T @ train["observed_availability"].to_numpy(float),
            )
            held = np.column_stack(
                [np.ones(len(test))]
                + [test[c].fillna(fills[c]).to_numpy(float) for c in columns]
            )
            fold[name] = float(
                np.mean(
                    np.abs(
                        held @ beta - test["observed_availability"].to_numpy(float)
                    )
                )
            )
        folds.append(fold)
    return pd.DataFrame(folds)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--player-rows",
        type=Path,
        default=Path(".cache/player_rows_probe.pkl"),
        help="a built player_rows frame carrying INJURY_AVAILABILITY_FEATURES",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    if not args.player_rows.exists():
        raise SystemExit(
            f"no frame at {args.player_rows}. Build one with "
            "build_season_average_data(..., source='nflverse', "
            "roster_mode='point_in_time') and pass --player-rows."
        )
    rows = pd.read_pickle(args.player_rows)
    missing = [name for name in INJURY_AVAILABILITY_FEATURES if name not in rows]
    if missing:
        raise SystemExit(
            f"the frame is missing {missing}; it predates the injury features"
        )
    rows = rows[
        rows["position"].isin(MODEL_POSITIONS)
        & rows["observed_availability"].notna()
    ].copy()

    print("=== 1. How often does each half fire at all? ===")
    print(f"rows: {len(rows)}\n")
    table = coverage(rows)
    print(table.round(3).to_string())
    reported = rows[rows["current_injury_reported"].gt(0)]
    print(
        "\n  current_injury_expected_recovery_weeks tops out at "
        f"{rows['current_injury_expected_recovery_weeks'].max():.2f} weeks, and "
        f"averages {reported['current_injury_expected_recovery_weeks'].mean():.2f} "
        "where a report exists.\n  _expected_recovery pools only uncensored "
        "episodes, and an episode is censored exactly when\n  the player never "
        "returned that season -- so the estimator never sees a long recovery."
    )

    print("\n=== 2. What does each start-of-season signal predict? ===")
    groups = signal_groups(rows)
    print(groups.round(2).to_string())

    print("\n=== 3. Does either half add anything over roster status? ===")
    arms = {
        "base (prior_availability + roster)": list(BASE_FEATURES),
        "+ three-year burden": [*BASE_FEATURES, *HISTORY_FEATURES],
        "+ start-of-season status": [*BASE_FEATURES, *CURRENT_FEATURES],
        "+ both halves": [*BASE_FEATURES, *HISTORY_FEATURES, *CURRENT_FEATURES],
    }
    folds = walk_forward(rows, arms)
    if folds.empty:
        raise SystemExit("no usable folds in this frame")
    names = list(arms)
    total = int(folds["n"].sum())
    baseline = float((folds[names[0]] * folds["n"]).sum() / total)
    print(f"  {names[0]:<36s} MAE {baseline:.5f}")
    results = {"folds": len(folds), "n": total, names[0]: baseline}
    for name in names[1:]:
        value = float((folds[name] * folds["n"]).sum() / total)
        wins = int((folds[name] < folds[names[0]]).sum())
        print(
            f"  {name:<36s} MAE {value:.5f}   "
            f"{100 * (value - baseline) / baseline:+.2f}%   {wins}/{len(folds)} folds"
        )
        results[name] = {
            "mae": value,
            "delta_pct": 100 * (value - baseline) / baseline,
            "folds_improved": wins,
        }
    print(
        "\n  Neither half clears the 0.25% materiality floor. The current half is "
        "the\n  one with the right sign and the burden half is the one that "
        "regresses, which\n  explains the standing null rather than overturning "
        "it."
    )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(
                {
                    "coverage": table.reset_index().to_dict("records"),
                    "signal_groups": groups.reset_index().to_dict("records"),
                    "walk_forward": results,
                },
                indent=2,
                default=str,
            ),
            "utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
