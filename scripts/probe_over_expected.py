"""Triage an "over expected" column before spending a ladder run on it.

Every metric in this family -- expected fantasy points, rush yards over
expected, completion percentage over expected, expected YAC -- is built the same
way: take what a player did, price the situations he did it in, and report the
difference. That construction makes them strong single predictors and it is also
exactly why they can be worthless here. The pricing model's inputs are
opportunities, and this model already reads the opportunities directly.

So the question is never "does this column correlate with next week" -- it
always does -- but "does it correlate with next week once the usage features
already in the design are removed". This script answers that for any candidate,
and reports the same number for raw actual points as the yardstick: a candidate
whose residual correlation is below that of the column it would replace has
nothing the model cannot already see.

A caveat this script cannot escape, and the reason its verdict is triage rather
than proof: a probe run against a subset of the model's inputs measures the
subset, not the model. A candidate that clears this bar still has to survive the
ladder.

    python scripts/probe_over_expected.py
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from ffmodel.weekly import FEATURES_CACHE
from ffmodel.weekly.features import relevant_population

# The opportunity columns the model already carries. An "over expected" metric
# is a function of these plus a pricing model; if it is nothing more than that,
# the residual correlation below goes to zero.
USAGE_FEATURES = (
    "prior_targets_recent",
    "prior_rush_att_recent",
    "prior_pass_att_recent",
    "prior_target_share_recent",
    "prior_rush_share_recent",
    "prior_snap_share_recent",
    "prior_target_share_last",
    "prior_rush_share_last",
    "prior_snap_share_last",
)

CANDIDATES = {
    "expected points (last week)": "prior_points_exp_last",
    "expected points (recency)": "prior_points_exp_recent",
    "actual points (last week)": "prior_points_last",
    "actual points (recency)": "prior_points_recent",
}


def _residual(frame: pd.DataFrame, column: str, basis: list[str]) -> tuple[np.ndarray, float]:
    """What is left of ``column`` after the basis explains it, and how much that was."""
    y = frame[column].to_numpy(float)
    X = np.column_stack([np.ones(len(frame))] + [frame[c].to_numpy(float) for c in basis])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    fitted = X @ beta
    explained = 1.0 - ((y - fitted) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    return y - fitted, float(explained)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=FEATURES_CACHE)
    parser.add_argument("--target", default="points")
    args = parser.parse_args(argv)

    frame = pd.read_pickle(args.features)
    frame = frame[relevant_population(frame)]

    basis = [c for c in USAGE_FEATURES if c in frame.columns]
    candidates = {k: v for k, v in CANDIDATES.items() if v in frame.columns}
    needed = [args.target, *basis, *candidates.values()]
    frame = frame.dropna(subset=needed).reset_index(drop=True)
    print(f"n = {len(frame)}, usage basis = {len(basis)} columns\n")

    points = frame[args.target].to_numpy(float)
    points_residual, _ = _residual(frame, args.target, basis)

    rows = []
    for label, column in candidates.items():
        residual, explained = _residual(frame, column, basis)
        rows.append(
            {
                "column": label,
                "explained by usage (R2)": round(explained, 4),
                "corr with next week": round(float(np.corrcoef(frame[column], points)[0, 1]), 4),
                "corr, usage removed": round(float(np.corrcoef(residual, points_residual)[0, 1]), 4),
            }
        )
    print(pd.DataFrame(rows).to_string(index=False))
    print(
        "\nRead the last column, not the second-to-last: a candidate below the"
        "\nrow for actual points is telling the model something it already knows."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
