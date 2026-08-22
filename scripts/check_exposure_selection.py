"""Is the availability under-projection a bias, or a resolution failure?

The volume decomposition says every component is projected light on the drafted
pool, at every position and on both folds -- games by 5-13%, snap share by
12-26%, carries and targets by up to 23%. A single mechanism producing an
identical sign across four positions and four unrelated components is unlikely.
A selection effect producing it is not.

The drafted pool is chosen, before the season, as the players expected to play.
If the model's availability projection is calibrated over the whole rostered
population but does not separate the drafted from the undrafted as sharply as
the market does, then it is necessarily low on the drafted pool and high on the
rest -- with no bias anywhere in the model. That is shrinkage, and it is a
resolution failure rather than a level one. The fix for the two is different:
a level bias wants a corrected intercept, a resolution failure wants a feature
that separates the groups, which is exactly what preseason ADP is.

The test distinguishes them in one line. Score the same projection on the
drafted pool and on everyone. Bias on both, same sign and size, is a level
problem. Bias that flips sign between them is selection.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

POSITIONS = ("QB", "RB", "WR", "TE")


def _summary(name: str, frame: pd.DataFrame) -> dict:
    projected = frame["projected_games"].to_numpy(float)
    observed = pd.to_numeric(frame["games"], errors="coerce").to_numpy(float)
    keep = np.isfinite(projected) & np.isfinite(observed)
    p, o = projected[keep], observed[keep]
    if len(p) < 10:
        return {"population": name, "n": int(len(p))}
    return {
        "population": name,
        "n": int(len(p)),
        "projected": float(p.mean()),
        "observed": float(o.mean()),
        "bias": float((p - o).mean()),
        "bias_pct": float((p - o).mean() / o.mean()) if o.mean() else float("nan"),
        "mae": float(np.abs(p - o).mean()),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--holdouts", type=int, nargs="+", default=[2022, 2023, 2024, 2025])
    parser.add_argument(
        "--out-dir", type=Path, default=Path(".cache/holdout-predictions")
    )
    parser.add_argument("--label", default="shipping")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("scripts/validation_runs/exposure_selection.json"),
    )
    args = parser.parse_args(argv)

    blocks = []
    for holdout in args.holdouts:
        base = args.out_dir / f"{args.label}_{holdout}"
        rows = pd.read_parquet(base.with_suffix(".rows.parquet"))
        payload = np.load(base.with_suffix(".samples.npz"))
        if "games_active" not in payload:
            raise SystemExit(
                f"{base}.samples.npz predates the exposure draws; re-run "
                "scripts/export_holdout_predictions.py"
            )
        rows = rows.copy()
        rows["projected_games"] = np.asarray(payload["games_active"], float).mean(axis=1)
        blocks.append(rows)

    frame = pd.concat(blocks, ignore_index=True)
    # Replacement rows are synthetic position buckets, not players. Including
    # them would put a third population in a comparison about two.
    frame = frame[
        pd.to_numeric(frame.get("is_replacement_player"), errors="coerce")
        .fillna(0)
        .ne(1)
    ].reset_index(drop=True)
    drafted = pd.to_numeric(frame["adp_drafted"], errors="coerce").eq(1)

    report: dict[str, object] = {
        "holdouts": args.holdouts,
        "pooled": [
            _summary("all rostered", frame),
            _summary("drafted", frame[drafted]),
            _summary("undrafted", frame[~drafted]),
        ],
        "positions": {},
    }
    for position in POSITIONS:
        at = frame["position"].eq(position)
        if at.sum() < 30:
            continue
        report["positions"][position] = [
            _summary("all rostered", frame[at]),
            _summary("drafted", frame[at & drafted]),
            _summary("undrafted", frame[at & ~drafted]),
        ]

    def table(title: str, entries: list[dict]) -> None:
        print(f"  {title}")
        print(
            f"    {'population':14s} {'n':>5s} {'projected':>10s} {'observed':>9s} "
            f"{'bias':>8s} {'bias %':>8s} {'MAE':>7s}"
        )
        for e in entries:
            if "bias" not in e:
                continue
            print(
                f"    {e['population']:14s} {e['n']:>5d} {e['projected']:>10.2f} "
                f"{e['observed']:>9.2f} {e['bias']:>+8.2f} {e['bias_pct']:>+7.1%} "
                f"{e['mae']:>7.2f}"
            )
        print()

    print(f"\nPROJECTED GAMES vs OBSERVED, holdouts {args.holdouts}\n")
    table("POOLED", report["pooled"])
    for position, entries in report["positions"].items():
        table(position, entries)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
