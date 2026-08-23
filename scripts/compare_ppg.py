"""Projected points per game against what players actually averaged.

The package projects and is graded on season totals, which is what a drafter
buys. But a season total is a rate times an exposure, and the two can be wrong
in opposite directions without the total saying so. This scores the rate on its
own.

The comparison is against three observed denominators, because "per game" means
three different things and the difference between them is the whole question.

``games``
    Weeks the player was on the active roster. This is what the model's
    availability layer targets and therefore what its own ``games_active``
    draws count, so it is the only denominator under which the projected and
    observed rates are the same quantity. It counts a week the player dressed
    and never took a snap.
``played``
    Weeks he recorded a stat line. Drops the dressed-but-unused weeks.
``full``
    Weeks he recorded a stat line and was not visibly pulled -- snap share at
    least half his own season median. Drops the games he left early.

Each step raises the observed average by removing near-zero games from the
numerator and a game from the denominator. A projection that is well calibrated
against ``games`` will look low against ``full`` by construction, and the size
of that gap is the interesting number: it says how much of the model's per-game
projection is really an unstated availability discount.

Both halves are paired per draw. ``points_draw / games_draw`` keeps whatever
correlation the simulation put between a player's exposure and his production;
taking the ratio of the two posterior means would quietly assume there is none.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from ffmodel.evaluation.partial_games import per_game_rates

POSITIONS = ("QB", "RB", "WR", "TE")
# Below this the rate is not estimated, it is one or two games of noise, and a
# mean over such rows is dominated by players nobody drafted to start.
MIN_GAMES = 4


def _load(out_dir: Path, label: str, holdout: int):
    base = out_dir / f"{label}_{holdout}"
    rows = pd.read_parquet(base.with_suffix(".rows.parquet"))
    payload = np.load(base.with_suffix(".samples.npz"))
    if "games_active" not in payload:
        raise SystemExit(
            f"{base}.samples.npz has no games_active. Re-run "
            "scripts/export_holdout_predictions.py -- the exposure draws were "
            "added after this file was written."
        )
    return rows, np.asarray(payload["samples"], float), np.asarray(
        payload["games_active"], float
    )


def _summary(name: str, projected: np.ndarray, observed: np.ndarray) -> dict:
    keep = np.isfinite(projected) & np.isfinite(observed)
    p, o = projected[keep], observed[keep]
    if len(p) < 10:
        return {"denominator": name, "n": int(len(p))}
    return {
        "denominator": name,
        "n": int(len(p)),
        "projected": float(p.mean()),
        "observed": float(o.mean()),
        "bias": float((p - o).mean()),
        "bias_pct": float((p - o).mean() / o.mean()) if o.mean() else float("nan"),
        "mae": float(np.abs(p - o).mean()),
        "r": float(np.corrcoef(p, o)[0, 1]) if len(p) > 2 else float("nan"),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--holdouts", type=int, nargs="+", default=[2022, 2023, 2024, 2025])
    parser.add_argument(
        "--out-dir", type=Path, default=Path(".cache/holdout-predictions")
    )
    parser.add_argument("--label", default="shipping")
    parser.add_argument("--scoring", default="ppr")
    parser.add_argument("--drafted-only", action="store_true", default=True)
    parser.add_argument(
        "--output", type=Path, default=Path("scripts/validation_runs/ppg_comparison.json")
    )
    args = parser.parse_args(argv)

    rates = per_game_rates(args.holdouts, scoring=args.scoring)
    blocks = []
    for holdout in args.holdouts:
        rows, points, games = _load(args.out_dir, args.label, holdout)
        # Per draw, then average. A player with a real chance of missing the
        # season has draws at zero games; those carry no rate and are dropped
        # from his posterior mean rather than being given one.
        with np.errstate(divide="ignore", invalid="ignore"):
            ppg_draws = np.where(games > 0, points / games, np.nan)
        frame = rows.copy()
        frame["projected_ppg"] = np.nanmean(ppg_draws, axis=1)
        frame["projected_games"] = games.mean(axis=1)
        blocks.append(frame)

    frame = pd.concat(blocks, ignore_index=True)
    frame = frame[
        pd.to_numeric(frame.get("is_replacement_player"), errors="coerce")
        .fillna(0)
        .ne(1)
    ]
    if args.drafted_only:
        frame = frame[pd.to_numeric(frame["adp_drafted"], errors="coerce").eq(1)]
    frame = frame.merge(rates, on=["player_id", "season"], how="left")

    # Three observed rates over one population. Restricting each to its own
    # usable rows would score three different sets of players and make the
    # columns incomparable, which is the one thing this table exists to avoid.
    played = pd.to_numeric(frame["weeks"], errors="coerce")
    roster = pd.to_numeric(frame["games"], errors="coerce")
    usable = (
        roster.ge(MIN_GAMES)
        & played.ge(MIN_GAMES)
        & pd.to_numeric(frame["full_weeks"], errors="coerce").ge(MIN_GAMES)
        & frame["clean_ppg"].notna()
        # One team only. A season-average row stops at a mid-season trade and
        # the weekly totals do not, so for a traded player the roster rate and
        # the played rate have different numerators and the ladder stops being
        # a ladder -- on 2022-2025 it made the played rate come out *below* the
        # roster rate for backs and receivers, which dropping a scoreless week
        # cannot do.
        & pd.to_numeric(frame["teams"], errors="coerce").eq(1)
    )
    frame = frame[usable.fillna(False)].reset_index(drop=True)

    frame["roster_ppg"] = frame["observed"] / pd.to_numeric(frame["games"], errors="coerce")
    observed_rates = {
        "games (roster-active)": frame["roster_ppg"].to_numpy(float),
        "played (stat line)": frame["raw_ppg"].to_numpy(float),
        "full (not pulled)": frame["clean_ppg"].to_numpy(float),
    }
    projected = frame["projected_ppg"].to_numpy(float)

    report: dict[str, object] = {
        "holdouts": args.holdouts,
        "scoring": args.scoring,
        "n": int(len(frame)),
        "partial_game_rate": float(
            frame["partial_weeks"].sum() / frame["weeks"].sum()
        ),
        "seasons_with_a_partial": float(frame["partial_weeks"].gt(0).mean()),
        "pooled": [_summary(k, projected, v) for k, v in observed_rates.items()],
        "positions": {},
        "exposure": {
            "projected_games": float(frame["projected_games"].mean()),
            "roster_games": float(frame["games"].mean()),
            "played_weeks": float(frame["weeks"].mean()),
            "full_weeks": float(frame["full_weeks"].mean()),
        },
    }
    for position in POSITIONS:
        mask = frame["position"].eq(position).to_numpy()
        if mask.sum() < 10:
            continue
        report["positions"][position] = [
            _summary(k, projected[mask], v[mask]) for k, v in observed_rates.items()
        ]

    print(
        f"\nPPG, drafted pool, holdouts {args.holdouts}, n={report['n']}\n"
        f"  {report['partial_game_rate']:.1%} of played games are partial; "
        f"{report['seasons_with_a_partial']:.1%} of seasons contain one\n"
        f"  exposure: projected {report['exposure']['projected_games']:.2f}, "
        f"roster {report['exposure']['roster_games']:.2f}, "
        f"played {report['exposure']['played_weeks']:.2f}, "
        f"full {report['exposure']['full_weeks']:.2f}\n"
    )

    def table(title: str, entries: list[dict]) -> None:
        print(f"  {title}")
        print(
            f"    {'denominator':22s} {'n':>4s} {'proj':>7s} {'obs':>7s} "
            f"{'bias':>8s} {'bias %':>8s} {'MAE':>7s} {'r':>6s}"
        )
        for e in entries:
            if "bias" not in e:
                continue
            print(
                f"    {e['denominator']:22s} {e['n']:>4d} {e['projected']:>7.2f} "
                f"{e['observed']:>7.2f} {e['bias']:>+8.2f} {e['bias_pct']:>+7.1%} "
                f"{e['mae']:>7.2f} {e['r']:>6.3f}"
            )
        print()

    table("POOLED", report["pooled"])
    for position, entries in report["positions"].items():
        table(position, entries)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
