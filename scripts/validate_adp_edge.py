"""Is the probability of beating ADP calibrated, and does it rank?

Fitted leave-one-season-out across the three scored holdouts: the curve for 2025
is fitted on 2023 and 2024 only, and so on. Each holdout's gaps already come
from a pipeline that never saw that season, so the projection side is clean; this
adds the same discipline to the logistic on top of it.

Two things to check and they are different. Calibration asks whether a stated
60% happens 60% of the time -- if it does not, the number is not usable as a
probability however well it sorts. Discrimination asks whether the ordering is
right at all, which is what a drafter uses when choosing between two players.
A model can be perfectly calibrated and useless (predict the base rate every
time, AUC 0.5) or sharp and misleading (right ordering, wrong levels).

    python scripts/validate_adp_edge.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ffmodel.models.adp_edge import (
    AdpEdgeModel,
    auc,
    brier_score,
    calibration_table,
)


def load(paths) -> dict[int, dict[str, np.ndarray]]:
    seasons: dict[int, dict[str, np.ndarray]] = {}
    for path in paths:
        blob = json.loads(Path(path).read_text("utf-8"))
        for season, fold in blob["folds"].items():
            rows = fold["rows"]
            model = np.array(rows["model_mean"], dtype=float)
            adp = np.array(rows["adp_mean"], dtype=float)
            observed = np.array(rows["observed"], dtype=float)
            seasons[int(season)] = {
                "gap": model - adp,
                "beat": (observed > adp).astype(float),
                "rank": np.array(rows["rank"], dtype=float),
            }
    return seasons


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reports", nargs="+",
        default=["reports/mva_2023.json", "reports/mva_2024.json", "reports/mva_2025.json"],
    )
    parser.add_argument(
        "--report-json", type=Path, default=Path("reports/adp_edge.json")
    )
    args = parser.parse_args(argv)

    seasons = load(args.reports)
    years = sorted(seasons)
    print(f"seasons {years}; "
          f"{sum(len(seasons[y]['gap']) for y in years)} drafted player-seasons")

    held_probability, held_beat, held_rank, held_gap = [], [], [], []
    print(f"\n{'holdout':>8} {'train n':>8} {'slope':>8} {'base rate':>10} {'AUC':>7} {'Brier':>8}")
    print("-" * 56)
    summary = {}
    for year in years:
        train_gap = np.concatenate([seasons[y]["gap"] for y in years if y != year])
        train_beat = np.concatenate([seasons[y]["beat"] for y in years if y != year])
        model = AdpEdgeModel().fit(train_gap, train_beat)
        probability = model.predict(seasons[year]["gap"])
        beat = seasons[year]["beat"]
        summary[str(year)] = {
            "slope": model.slope, "intercept": model.intercept,
            "base_rate": model.base_rate, "fitted": model.fitted,
            "auc": auc(probability, beat), "brier": brier_score(probability, beat),
            "n": int(len(beat)),
        }
        print(f"{year:>8} {len(train_gap):>8} {model.slope:>+8.3f} "
              f"{model.base_rate:>10.1%} {summary[str(year)]['auc']:>7.3f} "
              f"{summary[str(year)]['brier']:>8.4f}")
        held_probability.append(probability)
        held_beat.append(beat)
        held_rank.append(seasons[year]["rank"])
        held_gap.append(seasons[year]["gap"])

    probability = np.concatenate(held_probability)
    beat = np.concatenate(held_beat)
    rank = np.concatenate(held_rank)
    gap = np.concatenate(held_gap)

    print(f"\npooled out-of-sample: AUC {auc(probability, beat):.3f}  "
          f"Brier {brier_score(probability, beat):.4f}  "
          f"(always predicting the base rate scores "
          f"{brier_score(np.full(len(beat), beat.mean()), beat):.4f})")

    print("\ncalibration -- a stated rate should happen at that rate")
    table = calibration_table(probability, beat)
    if not table.empty:
        table["error"] = table.actual - table.predicted
        print(table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    print("\nby draft tier")
    print(f"  {'tier':12} {'n':>5} {'AUC':>7} {'mean p':>8} {'actual':>8}")
    for name, low, high in (("top50", 1, 50), ("51_150", 51, 150), ("151_300", 151, 300)):
        mask = np.isfinite(rank) & (rank >= low) & (rank <= high)
        if mask.sum() < 30:
            continue
        print(f"  {name:12} {int(mask.sum()):5d} {auc(probability[mask], beat[mask]):7.3f} "
              f"{probability[mask].mean():8.1%} {beat[mask].mean():8.1%}")

    print("\nwhat the model would tell a drafter, by how hard it disagrees")
    edges = np.quantile(gap, [0, 0.2, 0.4, 0.6, 0.8, 1.0])
    labels = ["model much lower", "model lower", "agree", "model higher", "model much higher"]
    print(f"  {'bucket':20} {'n':>5} {'stated p':>9} {'actual':>8}")
    for i, label in enumerate(labels):
        low, high = edges[i], edges[i + 1]
        mask = (gap >= low) & (gap <= high) if i == 4 else (gap >= low) & (gap < high)
        if mask.sum() < 10:
            continue
        print(f"  {label:20} {int(mask.sum()):5d} {probability[mask].mean():9.1%} "
              f"{beat[mask].mean():8.1%}")

    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(json.dumps(summary, indent=2, default=str), "utf-8")
    print(f"\nwrote {args.report_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
