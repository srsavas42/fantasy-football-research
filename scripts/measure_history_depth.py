"""How many prior seasons does each layer read, and how many should it?

The package is not uniform about this and the inconsistency is not documented
anywhere, so it is worth measuring rather than asserting.

`features/season_pathways.py` builds a genuine multi-season state for every
player: a career exponentially-weighted mean (`alpha = 0.50`, so the last
season carries half the weight, the one before a quarter) and a one-season
`_trend` difference, over eight production inputs and five efficiency ones.
`build_season_average_data` attaches them to every frame. But only one model
reads any of them -- `SeasonSnapShareModel.extra_features` defaults to
`SNAP_HISTORY_FEATURES`. Everything else in the shipping configuration sees a
single lagged season:

- the availability regression's `AVAILABILITY_FEATURES` carries
  `prior_availability` and no history term, though `prior_availability_3yr` is
  sitting in the same frame. Both baselines are reported for that layer: against
  `prior_availability` alone the career mean is worth -1.51%, against the full
  ten-feature design it is worth -0.10%, and only the second is the question the
  layer actually poses;
- the role allocators put `log(role_prior)` in as an offset built from the
  previous season alone;
- `lagged_efficiency_rows` shifts `Y` onto `Y+1` and there is no `prior2_*`
  column anywhere in the package.

This script asks, per layer, what a second and third season are worth. The
challengers are the cheapest possible: the same lag-1 feature plus a career
EWMA, plus a trend, or plus an explicit second lag. Walk-forward, one fold per
season, each fold fitted only on earlier ones.

A fold whose training rows cannot support an arm -- an all-missing trend column
in the earliest seasons, for instance -- is dropped from every arm rather than
scored, because a silently NaN arm reads as a large improvement once the
weighted mean skips it.

Two things make the efficiency answer smaller than it looks at first. The lag-1
efficiency feature is already `shrunk_*`, partially pooled toward the position
mean, so it has absorbed some of what an EWMA would do. And the response is a
single noisy season, which caps how much any predictor can win.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from ffmodel.models.season_availability import (
    AVAILABILITY_FEATURES,
    AVAILABILITY_HISTORY_FEATURES,
)

# Matches features/season_pathways.HISTORY_ALPHA, so the challenger here is the
# same quantity the package already builds rather than a new invention.
HISTORY_ALPHA = 0.50
RIDGE = 1e-3
MIN_TEST = 30

EFFICIENCY_ARMS = (
    ("rec_td_rate", "rec_td", "targets", ("WR", "TE", "RB"), 40),
    ("rush_td_rate", "rush_td", "rush_att", ("RB",), 60),
    ("rec_catch_rate", "receptions", "targets", ("WR", "TE", "RB"), 40),
    ("rec_yards_per_target", "rec_yds", "targets", ("WR", "TE", "RB"), 40),
    ("rush_yards_per_carry", "rush_yds", "rush_att", ("RB",), 60),
)


def walk_forward(
    frame: pd.DataFrame,
    response: str,
    arms: dict[str, list[str]],
    *,
    weight: str | None,
    min_train: int,
) -> pd.DataFrame:
    """Held-out MAE per arm per season, skipping folds no arm can support."""
    folds = []
    for season in sorted(frame["season"].unique()):
        train = frame[frame["season"] < season]
        test = frame[frame["season"] == season]
        if len(train) < min_train or len(test) < MIN_TEST:
            continue
        fold = {"season": int(season), "n": int(len(test))}
        usable = True
        for name, columns in arms.items():
            fills = {column: train[column].median() for column in columns}
            if any(not np.isfinite(value) for value in fills.values()):
                usable = False
                break
            design = np.column_stack(
                [np.ones(len(train))]
                + [train[c].fillna(fills[c]).to_numpy(float) for c in columns]
            )
            observed = train[response].to_numpy(float)
            weights = (
                np.ones(len(train)) if weight is None else train[weight].to_numpy(float)
            )
            weights = weights / weights.mean()
            penalty = np.eye(design.shape[1]) * RIDGE
            penalty[0, 0] = 0.0
            beta = np.linalg.solve(
                design.T @ (design * weights[:, None]) + penalty,
                design.T @ (observed * weights),
            )
            held = np.column_stack(
                [np.ones(len(test))]
                + [test[c].fillna(fills[c]).to_numpy(float) for c in columns]
            )
            error = float(np.mean(np.abs(held @ beta - test[response].to_numpy(float))))
            if not np.isfinite(error):
                usable = False
                break
            fold[name] = error
        if usable:
            folds.append(fold)
    return pd.DataFrame(folds)


def report(folds: pd.DataFrame, arms: dict[str, list[str]], label: str) -> dict:
    if folds.empty:
        print(f"--- {label}: no usable folds ---\n")
        return {}
    names = list(arms)
    total = int(folds["n"].sum())
    baseline = float((folds[names[0]] * folds["n"]).sum() / total)
    print(f"--- {label} | {len(folds)} folds, n = {total} ---")
    print(f"  {names[0]:<22s} MAE {baseline:.5f}")
    out = {"folds": len(folds), "n": total, names[0]: baseline}
    for name in names[1:]:
        value = float((folds[name] * folds["n"]).sum() / total)
        wins = int((folds[name] < folds[names[0]]).sum())
        print(
            f"  {name:<22s} MAE {value:.5f}   "
            f"{100 * (value - baseline) / baseline:+.2f}%   {wins}/{len(folds)} folds"
        )
        out[name] = {
            "mae": value,
            "delta_pct": 100 * (value - baseline) / baseline,
            "folds_improved": wins,
        }
    print()
    return out


def efficiency_transitions(
    efficiency: pd.DataFrame, target: str, numerator: str, denominator: str,
    positions: tuple[str, ...], floor: int,
) -> pd.DataFrame:
    """Lagged shrunk feature plus its own career history, against next season."""
    rows = efficiency[efficiency["position"].isin(positions)].copy()
    column = f"shrunk_{target}"
    grouped = rows.groupby("player_key")[column]
    rows["ewma"] = grouped.transform(
        lambda series: series.ewm(
            alpha=HISTORY_ALPHA, ignore_na=True, min_periods=1
        ).mean()
    )
    rows["lag2"] = grouped.shift(1)
    rows["trend"] = rows[column] - rows["lag2"]

    later = rows[["season", "player_key", numerator, denominator]].copy()
    later["season"] -= 1
    later = later.rename(
        columns={numerator: "numerator_next", denominator: "exposure_next"}
    )
    paired = rows.merge(later, on=["season", "player_key"], how="inner")
    paired = paired[
        paired[denominator].ge(floor) & paired["exposure_next"].ge(floor)
    ].copy()
    paired["rate_next"] = paired["numerator_next"] / paired["exposure_next"]
    paired = paired.dropna(subset=[column, "rate_next"])
    # A player with no second season falls back to his first: the arm is then
    # identical to lag-1 for that row rather than dropped, which is what a model
    # serving these features would actually do.
    paired["lag2"] = paired["lag2"].fillna(paired[column])
    paired["trend"] = paired["trend"].fillna(0.0)
    return paired


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache"))
    parser.add_argument(
        "--player-rows",
        type=Path,
        default=Path(".cache/player_rows_probe.pkl"),
        help="a built player_rows frame, for the snap and availability layers",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    results = {}

    rows_path = args.player_rows
    if rows_path.exists():
        player_rows = pd.read_pickle(rows_path)

        print("=== AVAILABILITY: the column is in the frame and unread ===")
        print("prior_availability_3yr is built for every row. AVAILABILITY_FEATURES")
        print("does not list it, so the shipping availability model never sees it.")
        print("Both baselines are reported on purpose. The thin one answers 'does a")
        print("career mean beat last season alone', which is not the question the")
        print("layer poses; the full one answers 'does it add to what the layer")
        print("already has', which is, and gives a much smaller answer.\n")
        available = player_rows[
            player_rows["position"].isin(("QB", "RB", "WR", "TE"))
            & player_rows["prior_availability"].notna()
            & player_rows["observed_availability"].notna()
        ].copy()
        for name in AVAILABILITY_FEATURES:
            if name not in available:
                available[name] = np.nan
        thin = {
            "lag-1 only": ["prior_availability"],
            "+ career ewma": ["prior_availability", *AVAILABILITY_HISTORY_FEATURES],
        }
        folds = walk_forward(
            available, "observed_availability", thin, weight=None, min_train=400
        )
        results["availability_thin_baseline"] = report(
            folds, thin, "AVAILABILITY vs prior_availability alone (the wrong baseline)"
        )
        full = {
            "full design": list(AVAILABILITY_FEATURES),
            "+ career ewma": [*AVAILABILITY_FEATURES, *AVAILABILITY_HISTORY_FEATURES],
        }
        folds = walk_forward(
            available, "observed_availability", full, weight=None, min_train=400
        )
        results["availability"] = report(
            folds, full, "AVAILABILITY vs the full shipping design (the right one)"
        )

        print("=== SNAP SHARE: the one layer that already reads history ===")
        skill = player_rows[
            player_rows["position"].isin(("RB", "WR", "TE"))
            & player_rows["prior_snap_share"].notna()
            & player_rows["snap_share"].notna()
        ]
        arms = {
            "lag-1 only": ["prior_snap_share", "prior_availability"],
            "+ ewma + trend": [
                "prior_snap_share",
                "prior_availability",
                "prior_snap_share_3yr",
                "prior_snap_share_trend",
                "prior_availability_3yr",
            ],
        }
        folds = walk_forward(skill, "snap_share", arms, weight=None, min_train=400)
        results["snap_share"] = report(folds, arms, "SNAP SHARE")
    else:
        print(
            f"no player_rows frame at {rows_path}; skipping the availability and "
            "snap layers.\nBuild one with build_season_average_data and pass "
            "--player-rows.\n"
        )

    efficiency_path = args.cache_dir / "season_efficiency.pkl"
    if not efficiency_path.exists():
        print(f"no efficiency frame at {efficiency_path}; run "
              "measure_efficiency_reversion.py first.")
        return 0
    efficiency = pd.read_pickle(efficiency_path).sort_values(["player_key", "season"])

    print("=== EFFICIENCY: the layer with no multi-season features at all ===")
    for target, numerator, denominator, positions, floor in EFFICIENCY_ARMS:
        paired = efficiency_transitions(
            efficiency, target, numerator, denominator, positions, floor
        )
        column = f"shrunk_{target}"
        arms = {
            "lag-1 only": [column],
            "+ career ewma": [column, "ewma"],
            "+ ewma + trend": [column, "ewma", "trend"],
            "+ explicit lag-2": [column, "lag2"],
        }
        folds = walk_forward(
            paired, "rate_next", arms, weight="exposure_next", min_train=400
        )
        results[target] = report(folds, arms, f"EFFICIENCY {target}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2, default=str), "utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
