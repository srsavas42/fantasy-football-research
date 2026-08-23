"""Score two saved holdout exports on the metrics the point gate cannot see.

MAE, RMSE and CRPS are one quantity read three ways: distance from truth in
points, at the centre, in the tails, and over the whole distribution. Two things
they miss are scored here, from posterior samples already on disk -- no fitting.

**Resolution.** The CRPS decomposition splits reliability (are the stated
probabilities honest) from resolution (does the forecast tell outcomes apart).
The defect this package spent a session on was a resolution failure diagnosed
indirectly, from bias splits and an in-sample refit. This measures it.

**Ordering.** The product is a ranked list within a position, and nothing in the
gate knew that. A projection can improve its MAE while ordering players worse.

Two diagnostics come along: the PIT shape, which is the whole calibration curve
rather than the two points interval coverage samples, and a CRPS skill score
against the draft board, which turns "1.1% better" into a share of the gap that
was actually available.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from ffmodel.evaluation.metrics import (
    crps_decomposition,
    crps_skill_score,
    empirical_crps,
    ordering_metrics,
    pit_calibration,
)
from ffmodel.models.market_blend import RankCurve

TOP_K = 12


def load(directory: Path, label: str, holdout: int):
    base = directory / f"{label}_{holdout}"
    rows = pd.read_parquet(base.with_suffix(".rows.parquet"))
    samples = np.load(base.with_suffix(".samples.npz"))["samples"].astype(float)
    named = (
        pd.to_numeric(rows.get("is_replacement_player"), errors="coerce")
        .fillna(0)
        .ne(1)
        .to_numpy()
    )
    keep = named & rows["observed"].notna().to_numpy()
    return rows[keep].reset_index(drop=True), samples[keep]


def score(rows: pd.DataFrame, samples: np.ndarray, reference: np.ndarray | None) -> dict:
    observed = rows["observed"].to_numpy(float)
    mean = samples.mean(axis=1)
    parts = crps_decomposition(observed, samples)
    ordering = ordering_metrics(mean, observed, rows["position"].to_numpy(), k=TOP_K)
    calibration = pit_calibration(observed, samples)
    out = {
        "n": int(len(rows)),
        "crps": parts["crps"],
        "reliability": parts["reliability"],
        "resolution": parts["resolution"],
        "uncertainty": parts["uncertainty"],
        "spearman": ordering.get("within_group_spearman", float("nan")),
        "concordance": ordering.get("within_group_concordance", float("nan")),
        "top_k": ordering.get("within_group_top_k", ordering["top_k"]),
        "pit_deviation": calibration["deviation"],
        "pit_shape": calibration["shape"],
        "by_position": ordering.get("by_group", {}),
    }
    if reference is not None:
        out["skill_vs_board"] = crps_skill_score(
            empirical_crps(observed, samples), empirical_crps(observed, reference)
        )
    return out


def board_samples(cache: Path, rows: pd.DataFrame, holdout: int, draws: int) -> np.ndarray | None:
    """A rank-curve forecast for the same rows, as the skill-score reference.

    Fitted on every prior season in the cache, drafted rows only -- the same
    construction the shipped blend uses, so the skill score is against the
    baseline this package is actually judged against.
    """
    from ffmodel.evaluation.efficiency_posterior import observed_scoring_rows
    from ffmodel.simulation.scoring import fantasy_points

    frame = pd.read_pickle(cache / "player_rows.pkl")
    frame = frame[frame.season.lt(holdout)].reset_index(drop=True)
    frame = frame[
        pd.to_numeric(frame.get("is_replacement_player"), errors="coerce").fillna(0).ne(1)
    ].reset_index(drop=True)
    points = np.full(len(frame), np.nan)
    for _, block in frame.groupby("season"):
        points[block.index] = fantasy_points(
            observed_scoring_rows(block.reset_index(drop=True)), "ppr"
        ).to_numpy()
    usable = np.isfinite(points)
    try:
        curve = RankCurve().fit(frame[usable].reset_index(drop=True), points[usable])
    except ValueError:
        return None
    return curve.predict_samples(rows, draws=draws, seed=holdout)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--holdouts", type=int, nargs="+", default=[2022, 2023, 2024, 2025])
    parser.add_argument(
        "--baseline", type=Path, default=Path(".cache/holdout-predictions-preavail")
    )
    parser.add_argument(
        "--candidate", type=Path, default=Path(".cache/holdout-predictions")
    )
    parser.add_argument("--label", default="shipping")
    parser.add_argument(
        "--cache-dir", type=Path, default=Path(".cache/ffmodel-wf-2025-adp2")
    )
    parser.add_argument("--drafted-only", action="store_true")
    parser.add_argument(
        "--output", type=Path, default=Path("scripts/validation_runs/gate_metrics.json")
    )
    args = parser.parse_args(argv)

    report: dict[str, object] = {"holdouts": args.holdouts, "folds": {}}
    for holdout in args.holdouts:
        base_rows, base_samples = load(args.baseline, args.label, holdout)
        cand_rows, cand_samples = load(args.candidate, args.label, holdout)
        if len(base_rows) != len(cand_rows):
            # The two exports must cover identical rows or every difference
            # below is partly a population difference.
            key = ["player_id", "season"]
            shared = base_rows[key].merge(cand_rows[key], on=key)
            base_mask = base_rows[key].apply(tuple, axis=1).isin(
                shared.apply(tuple, axis=1)
            )
            cand_mask = cand_rows[key].apply(tuple, axis=1).isin(
                shared.apply(tuple, axis=1)
            )
            base_rows, base_samples = base_rows[base_mask].reset_index(drop=True), base_samples[base_mask.to_numpy()]
            cand_rows, cand_samples = cand_rows[cand_mask].reset_index(drop=True), cand_samples[cand_mask.to_numpy()]
        if args.drafted_only:
            base_keep = pd.to_numeric(base_rows["adp_drafted"], errors="coerce").eq(1).to_numpy()
            cand_keep = pd.to_numeric(cand_rows["adp_drafted"], errors="coerce").eq(1).to_numpy()
            base_rows, base_samples = base_rows[base_keep].reset_index(drop=True), base_samples[base_keep]
            cand_rows, cand_samples = cand_rows[cand_keep].reset_index(drop=True), cand_samples[cand_keep]

        reference = board_samples(
            args.cache_dir, cand_rows, holdout, cand_samples.shape[1]
        )
        # The curve has nothing to say about undrafted players; score the skill
        # comparison only where it has an opinion, on identical rows.
        if reference is not None:
            ranked = np.isfinite(reference).all(axis=1)
            skill_reference = reference[ranked]
        else:
            ranked, skill_reference = None, None

        entry = {
            "raw": score(base_rows, base_samples, None),
            "joint": score(cand_rows, cand_samples, None),
        }
        if skill_reference is not None and ranked.any():
            entry["raw"]["skill_vs_board"] = crps_skill_score(
                empirical_crps(base_rows["observed"].to_numpy(float)[ranked], base_samples[ranked]),
                empirical_crps(cand_rows["observed"].to_numpy(float)[ranked], skill_reference),
            )
            entry["joint"]["skill_vs_board"] = crps_skill_score(
                empirical_crps(cand_rows["observed"].to_numpy(float)[ranked], cand_samples[ranked]),
                empirical_crps(cand_rows["observed"].to_numpy(float)[ranked], skill_reference),
            )
        report["folds"][str(holdout)] = entry

    population = "drafted pool" if args.drafted_only else "all rostered"
    print(f"\nGATE METRICS, {population}\n")
    header = (
        f"  {'holdout':>7s} {'resolution':>21s} {'reliability':>21s} "
        f"{'spearman':>19s} {'top12':>17s}"
    )
    print(header)
    print(f"  {'':>7s} {'raw -> joint':>21s} {'raw -> joint':>21s} "
          f"{'raw -> joint':>19s} {'raw -> joint':>17s}")
    for holdout, entry in report["folds"].items():
        r, j = entry["raw"], entry["joint"]
        print(
            f"  {holdout:>7s} {r['resolution']:>9.3f} -> {j['resolution']:<9.3f} "
            f"{r['reliability']:>9.3f} -> {j['reliability']:<9.3f} "
            f"{r['spearman']:>8.3f} -> {j['spearman']:<8.3f} "
            f"{r['top_k']:>7.3f} -> {j['top_k']:<7.3f}"
        )

    def pooled(arm: str, key: str) -> float:
        values = [report["folds"][str(h)][arm][key] for h in args.holdouts]
        return float(np.mean([v for v in values if np.isfinite(v)]))

    print("\n  pooled")
    for key in ("resolution", "reliability", "crps", "spearman", "concordance", "top_k", "pit_deviation"):
        raw, joint = pooled("raw", key), pooled("joint", key)
        better = "higher" if key in {"resolution", "spearman", "concordance", "top_k"} else "lower"
        arrow = "better" if ((joint > raw) == (better == "higher")) else "worse"
        print(f"    {key:14s} {raw:>9.4f} -> {joint:>9.4f}   ({arrow}, {better} is better)")
    if "skill_vs_board" in report["folds"][str(args.holdouts[0])]["joint"]:
        raw, joint = pooled("raw", "skill_vs_board"), pooled("joint", "skill_vs_board")
        print(f"    {'skill vs board':14s} {raw:>9.4f} -> {joint:>9.4f}   (share of the board's CRPS removed)")
    print("\n  PIT shape")
    for holdout, entry in report["folds"].items():
        print(f"    {holdout}: raw {entry['raw']['pit_shape']:<40s} joint {entry['joint']['pit_shape']}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
