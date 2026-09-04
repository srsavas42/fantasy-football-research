"""Does knowing the roof, the temperature and the wind improve a weekly projection?

Three rungs, identical except for the columns under test, walked forward over
2023/2024/2025:

``hurdle+everything+ngs/position``
    The shipped next-week model. The control.

``+roof``
    One column: is this game indoors. Known when the schedule is published, so a
    win here is directly shippable.

``+weather``
    Roof plus temperature, wind, a freezing indicator, a high-wind indicator and
    a missing-reading flag. The two readings are recorded *at* the game, so this
    rung measures the **ceiling** a perfect forecast would reach -- not something
    a Sunday-morning projection could have had. If the ceiling is null the
    forecast version is null a fortiori and the question is settled cheaply; if
    it is not, the shippable version reads Open-Meteo's previous-run archive via
    :mod:`ffmodel.data.weather`.

    python scripts/validate_weekly_weather.py --holdouts 2023 2024 2025
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import pandas as pd

from ffmodel.weekly import FEATURES_CACHE
from ffmodel.weekly.evaluate import report, walk_forward
from ffmodel.weekly.nextweek import Hurdle
from ffmodel.weekly.weather import WEATHER_COLUMNS, attach_weather

CONTROL = "hurdle+everything+ngs/position"
COLUMNS = [
    "mae",
    "rmse",
    "crps",
    "bias",
    "coverage_80",
    "coverage_95",
    "within_group_spearman",
    "within_group_top_k",
    "pit_deviation",
]

# The package's promotion floor: a change smaller than this on CRPS is reported
# as a tie regardless of its sign, because three folds cannot resolve it.
MATERIALITY = 0.0025

BASE = {
    "use_team": True,
    "use_matchup": True,
    "use_phase": True,
    "use_script": True,
    "use_adp": True,
    "use_news": True,
    "use_snaps": True,
    "use_recent": True,
    "use_pedigree": True,
    "use_charting": True,
    "by_position": True,
}


def ladder() -> list:
    return [
        Hurdle(name=CONTROL, **BASE),
        Hurdle(name="hurdle+everything+ngs+roof/position", use_roof=True, **BASE),
        Hurdle(name="hurdle+everything+ngs+weather/position", use_weather=True, **BASE),
    ]


def _deltas(results: dict, population: str) -> pd.DataFrame:
    """Each rung against the control, as a signed percentage change."""
    table = report(results, population).set_index("estimator")
    if CONTROL not in table.index:
        return pd.DataFrame()
    rows = []
    for name in table.index:
        if name == CONTROL:
            continue
        row = {"estimator": name, "n": int(table.loc[name, "n"])}
        for metric in ("mae", "crps"):
            if metric not in table.columns:
                continue
            base = float(table.loc[CONTROL, metric])
            got = float(table.loc[name, metric])
            # Lower is better for both, so a negative change is an improvement.
            row[f"{metric}_delta_pct"] = 100.0 * (got - base) / base if base else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def _per_fold(results: dict, population: str) -> pd.DataFrame:
    """CRPS by fold, because a pooled win built on one season is not a win."""
    rows = []
    for fold in results.get("folds", []):
        entry = {"holdout": fold["holdout"]}
        for name, block in fold.get("estimators", {}).items():
            scored = block.get(population)
            if scored is not None and "crps" in scored:
                entry[name] = round(float(scored["crps"]), 4)
        rows.append(entry)
    return pd.DataFrame(rows)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--holdouts", type=int, nargs="+", default=[2023, 2024, 2025])
    parser.add_argument("--draws", type=int, default=800)
    parser.add_argument("--features", type=Path, default=FEATURES_CACHE)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    if not args.features.exists():
        raise SystemExit(
            f"no feature cache at {args.features}; run scripts/validate_weekly.py first"
        )
    frame = pd.read_pickle(args.features)
    frame = attach_weather(frame)
    joined = frame["roof_indoor"].notna().mean()
    print(f"panel {frame.shape[0]} rows, conditions joined on {joined:.1%} of them")
    print(
        "indoor "
        f"{frame['roof_indoor'].mean():.1%}, reading missing "
        f"{frame['wx_missing'].mean():.1%}"
    )
    missing = [c for c in WEATHER_COLUMNS if c not in frame.columns]
    if missing:
        raise SystemExit(f"weather columns absent after attach: {missing}")

    results = walk_forward(
        frame,
        ladder(),
        target="points",
        holdouts=args.holdouts,
        draws=args.draws,
    )

    payload: dict[str, object] = {"next_week_weather": results}
    for population in ("relevant", "panel", "relevant_early", "relevant_mid", "relevant_late"):
        table = report(results, population)
        if table.empty:
            continue
        keep = ["estimator", "n"] + [c for c in COLUMNS if c in table.columns]
        print(f"\n-- {population} --")
        print(table[keep].round(4).to_string(index=False))
        delta = _deltas(results, population)
        if not delta.empty:
            print("\n   against the control (negative = better):")
            print("   " + delta.round(3).to_string(index=False).replace("\n", "\n   "))
            for _, row in delta.iterrows():
                change = row.get("crps_delta_pct", 0.0) / 100.0
                verdict = (
                    "tie (below the 0.25% floor)"
                    if abs(change) < MATERIALITY
                    else ("improves" if change < 0 else "degrades")
                )
                print(f"   {row['estimator']}: {verdict}")

    fold_table = _per_fold(results, "relevant")
    if not fold_table.empty:
        print("\n-- CRPS by fold, relevant population --")
        print(fold_table.to_string(index=False))

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, default=str), "utf-8")
        print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
