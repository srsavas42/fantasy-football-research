"""Does the snap model reproduce the draft-slot spread the flat rate prior assumes?

The rookie-prior refit (docs/target-competition-2026-09.md) rests on an
arithmetic identity: a rookie's volume share is his exposure times his per-snap
rate, the softmax already carries ``log(exposure)`` as an offset, so the prior
belongs on the rate. Measured over every rookie in the cache, round 1 against
undrafted:

    observed snap share      8.66x
    observed target share   13.59x
    implied per-snap rate    1.57x

The refit's curve spans 1.67x, which is the right answer *for the rate*. But
the identity only closes if the model's **projected** exposure carries the 8.66x
itself. It cannot be assumed: the snap model is projecting a player with no NFL
history, and shrinkage toward the position mean is exactly what a hierarchical
model does with a row it knows nothing about. Whatever spread it fails to carry
has to live somewhere, and under the legacy curve it lived in the steepness the
diagnosis called a double count.

So: fit once, predict the holdout, and compare projected against observed snap
share by draft bucket. Snap predictions are identical under both curves -- the
paired walk-forward measured +0.000% on every fold -- so one run answers it for
both arms.

    python scripts/diagnose_rookie_exposure_spread.py --holdout 2024
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _walkforward_data import add_common_arguments, load_frames

from ffmodel.features.season_average import SeasonAverageData
from ffmodel.models.volume_season_average import SeasonAverageVolumePipeline

BUCKETS = ("rd1", "rd2-3", "rd4-7", "undrafted")


def bucket_of(pick) -> str:
    if pick is None or pd.isna(pick):
        return "undrafted"
    pick = float(pick)
    if pick <= 32:
        return "rd1"
    return "rd2-3" if pick <= 100 else "rd4-7"


def main(argv=None) -> None:
    parser = add_common_arguments(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--holdout", type=int, default=2024)
    parser.add_argument(
        "--output", type=Path, default=Path("scripts/validation_runs/rookie_exposure.json")
    )
    args = parser.parse_args(argv)

    player_rows, team_rows = load_frames(args.cache_dir)
    train = SeasonAverageData(
        team_rows[team_rows.season < args.holdout].copy(),
        player_rows[player_rows.season < args.holdout].copy(),
    )
    test = SeasonAverageData(
        team_rows[team_rows.season == args.holdout].copy(),
        player_rows[player_rows.season == args.holdout].copy(),
    )
    pipeline = SeasonAverageVolumePipeline()
    pipeline.fit(train, draws=args.draws, tune=args.tune, chains=args.chains)
    prediction = pipeline.predict_samples(test, seed=42)

    rows = prediction.player_rows
    projected = prediction.snap_share.mean(axis=1)
    observed = pd.to_numeric(rows["snap_share"], errors="coerce").fillna(0.0).to_numpy(float)
    rookie = (
        pd.to_numeric(rows.get("cold_start", pd.Series(0, index=rows.index)), errors="coerce")
        .fillna(0).eq(1)
        & pd.to_numeric(
            rows.get("experience", pd.Series(np.nan, index=rows.index)), errors="coerce"
        ).fillna(99).le(0)
        & rows["position"].isin(["WR", "RB", "TE"])
    ).to_numpy()

    buckets = np.array([bucket_of(p) for p in rows["overall_pick"]])
    report: dict[str, object] = {"holdout": args.holdout, "buckets": {}}
    for name in BUCKETS:
        mask = rookie & (buckets == name)
        if not mask.any():
            continue
        report["buckets"][name] = {
            "n": int(mask.sum()),
            "projected_snap": float(projected[mask].mean()),
            "observed_snap": float(observed[mask].mean()),
        }

    top, bottom = report["buckets"].get("rd1"), report["buckets"].get("undrafted")
    if top and bottom:
        report["spread"] = {
            "projected": top["projected_snap"] / bottom["projected_snap"],
            "observed": top["observed_snap"] / bottom["observed_snap"],
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    print(f"{'bucket':>10s} {'n':>4s} {'projected':>10s} {'observed':>10s} {'proj/obs':>9s}")
    for name in BUCKETS:
        cell = report["buckets"].get(name)
        if cell is None:
            continue
        print(f"{name:>10s} {cell['n']:>4d} {cell['projected_snap']:>10.4f} "
              f"{cell['observed_snap']:>10.4f} "
              f"{cell['projected_snap'] / cell['observed_snap']:>9.2f}")
    if "spread" in report:
        print(f"\nrd1/undrafted spread: projected {report['spread']['projected']:.2f}x "
              f"against observed {report['spread']['observed']:.2f}x")
    print(f"-> {args.output}")


if __name__ == "__main__":
    main()
