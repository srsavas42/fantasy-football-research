"""Does red-zone role add anything the frame does not already carry?

Three nulls this session came from features that recombined information already
present. The discriminating question is not "is this signal real" — red-zone
role obviously exists — but "is it already in the frame under another name".

The frame has `shrunk_rush_td_rate`, `shrunk_rec_td_rate` and their priors, so
it already knows who scored touchdowns per opportunity and already regresses
that toward the mean. What it does not have is *where the opportunities were*.
The claim under test is that field position is a persistent property of a
player's role which touchdown rate only measures with a season of noise.

## Ordered in advance

Three targets are available and testing all three without saying which matters
is how a 1-in-3 result gets reported as a finding. The primary hypothesis is
stated first and the others are exploratory:

1. **Primary — touchdown rate.** If red-zone role is a real trait, its clearest
   effect is on touchdowns per opportunity next season, over and above the
   shrunk touchdown rate the frame already carries. This is the test that
   decides whether the feature is worth wiring in.
2. Secondary — role share. Whether the allocators' own targets (carry share,
   target share) are better predicted with zone differentials present.
3. Exploratory — total fantasy points per game.

Every probe runs **with the frame's existing predictors present**. The
interaction probe earlier today reported a 4.11% gain in a room containing three
features and +0.04% in the room the model actually lives in; a probe against a
subset of the model's inputs measures the subset.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from ffmodel.features.redzone import REDZONE_FEATURES, add_redzone_features

MIN_GAMES = 8
MATERIAL = 0.0025

# What the frame already knows about each target, so the probe asks for
# increments rather than for the signal from scratch.
BASE = {
    "rush_td_rate": [
        "prior_rush_epa_per_carry",
        "prior_carry_role",
        "prior_snap_share",
        "age",
        "experience",
    ],
    "rec_td_rate": [
        "prior_rec_epa_per_target",
        "prior_target_role",
        "prior_snap_share",
        "age",
        "experience",
    ],
    "carry_share": ["prior_carry_role", "prior_snap_share", "age", "experience"],
    "target_share": ["prior_target_role", "prior_snap_share", "age", "experience"],
}
PRIOR_TD = {"rush_td_rate": "shrunk_rush_td_rate", "rec_td_rate": "shrunk_rec_td_rate"}
POSITIONS = ("QB", "RB", "WR", "TE")


def design(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    parts = [np.ones(len(frame))]
    for position in POSITIONS[:-1]:
        parts.append(frame.position.eq(position).to_numpy(float))
    for name in columns:
        values = pd.to_numeric(frame[name], errors="coerce")
        filled = values.fillna(values.median() if values.notna().any() else 0.0)
        parts.append(filled.to_numpy(float))
    return np.column_stack(parts)


def walk_forward(frame: pd.DataFrame, target: str, columns: list[str], holdouts) -> float:
    errors, n = 0.0, 0
    for holdout in holdouts:
        train = frame[frame.season < holdout]
        test = frame[frame.season == holdout]
        if len(test) == 0 or len(train) < 100:
            continue
        y = train[target].to_numpy(float)
        beta, *_ = np.linalg.lstsq(design(train, columns), y, rcond=None)
        predicted = design(test, columns) @ beta
        errors += np.abs(predicted - test[target].to_numpy(float)).sum()
        n += len(test)
    return errors / max(n, 1)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-dir", type=Path, default=Path(".cache/ffmodel-wf-2025-adp2")
    )
    parser.add_argument("--usage", type=Path, default=Path(".cache/zone_usage.pkl"))
    parser.add_argument("--holdouts", type=int, nargs="+", default=[2020, 2021, 2022, 2023, 2024])
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    output = args.output or Path("scripts/validation_runs/redzone_probe.json")

    rows = pd.read_pickle(args.cache_dir / "player_rows.pkl")
    usage = pd.read_pickle(args.usage)
    rows = add_redzone_features(rows, usage=usage)
    rows = rows[rows.position.isin(POSITIONS)].copy()
    named = (
        pd.to_numeric(rows.get("is_replacement_player"), errors="coerce").fillna(0).ne(1)
    )
    games = pd.to_numeric(rows.get("games"), errors="coerce")
    rows = rows[named & games.ge(MIN_GAMES)].reset_index(drop=True)

    present = [c for c in REDZONE_FEATURES if c in rows]
    coverage = {c: float(pd.to_numeric(rows[c], errors="coerce").notna().mean()) for c in present}
    print(f"\nrows: {len(rows)}   zone-feature coverage:")
    for name, value in coverage.items():
        print(f"  {name:38s} {value:>6.1%}")

    order = [
        ("PRIMARY", "rush_td_rate"),
        ("PRIMARY", "rec_td_rate"),
        ("secondary", "carry_share"),
        ("secondary", "target_share"),
    ]
    results: dict[str, dict[str, float]] = {}
    print(f"\n{'target':>14s}  {'rank':9s} {'base MAE':>10s} {'+ zone':>10s} {'change':>9s}")
    for rank, target in order:
        if target not in rows:
            print(f"{target:>14s}  {rank:9s}  column absent")
            continue
        block = rows[pd.to_numeric(rows[target], errors="coerce").notna()].copy()
        columns = [c for c in BASE[target] if c in block]
        if target in PRIOR_TD and PRIOR_TD[target] in block:
            columns.append(PRIOR_TD[target])
        base = walk_forward(block, target, columns, args.holdouts)
        with_zone = walk_forward(block, target, columns + present, args.holdouts)
        change = (with_zone - base) / base
        results[target] = {
            "rank": rank,
            "n": int(len(block)),
            "base": base,
            "with_zone": with_zone,
            "change": change,
        }
        print(
            f"{target:>14s}  {rank:9s} {base:>10.5f} {with_zone:>10.5f} {change:>+8.2%}"
        )

    print()
    primary = [v for v in results.values() if v["rank"] == "PRIMARY"]
    won = [v for v in primary if v["change"] < -MATERIAL]
    if won:
        print(f"  PRIMARY: {len(won)} of {len(primary)} touchdown-rate targets improve "
              f"beyond {MATERIAL:.2%}")
    else:
        print(f"  PRIMARY: no touchdown-rate target improves beyond {MATERIAL:.2%}. "
              "The frame\n  already carries what red-zone role would have told it.")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps({"coverage": coverage, "results": results}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"\nwrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
