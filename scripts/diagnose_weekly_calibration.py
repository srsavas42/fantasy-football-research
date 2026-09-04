"""Is the hurdle's wide interval a miscalibration, or the atom it was built to have?

The hurdle improves CRPS over the recency mean and simultaneously reports 80%
coverage well above 0.80. Read naively that is a model that got better and worse
at once, and the temptation is to widen or narrow something until the coverage
number looks right.

Central-interval coverage is not a proper scoring rule, and it behaves badly on a
distribution with a point mass. If a player has a 12% chance of not playing, 12%
of the predictive sits at exactly zero, so the 10th percentile *is* zero and any
positive outcome falls inside the interval from below. The interval is not too
wide; the summary is wrong for the shape.

This separates the two explanations rather than asserting one:

**The availability half** gets a reliability table. Group rows by predicted play
probability and compare against the realised rate. If those columns agree, the
zero mass is the right size and nothing about it needs adjusting.

**The magnitude half** gets its own coverage and PIT, computed only on rows where
the player actually played. That distribution has no atom, so coverage means
what it usually means there. If it lands near nominal, the pooled number is the
atom and not a defect.

If instead the availability model is over-confident, or the played-only coverage
is off, then the pooled figure is a real calibration failure and the model needs
work -- which is the outcome this script exists to be able to report.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from ffmodel.evaluation.metrics import interval_coverage, pit_calibration
from ffmodel.weekly import FEATURES_CACHE
from ffmodel.weekly.features import add_features, relevant_population
from ffmodel.weekly.frame import load_panel
from ffmodel.weekly.nextweek import Hurdle, HistoryMean

BINS = (0.0, 0.3, 0.5, 0.7, 0.85, 0.95, 1.0001)


def reliability(probability: np.ndarray, played: np.ndarray) -> pd.DataFrame:
    """Predicted play probability against the rate actually observed."""
    index = np.clip(np.searchsorted(BINS, probability, side="right") - 1, 0, len(BINS) - 2)
    rows = []
    for bucket in range(len(BINS) - 1):
        want = index == bucket
        if want.sum() < 50:
            continue
        rows.append(
            {
                "bucket": f"{BINS[bucket]:.2f}-{BINS[bucket + 1]:.2f}",
                "n": int(want.sum()),
                "predicted": float(probability[want].mean()),
                "observed": float(played[want].mean()),
                "gap": float(probability[want].mean() - played[want].mean()),
            }
        )
    return pd.DataFrame(rows)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--holdouts", type=int, nargs="+", default=[2023, 2024, 2025])
    parser.add_argument("--draws", type=int, default=800)
    parser.add_argument(
        "--features", type=Path, default=FEATURES_CACHE
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    frame = (
        pd.read_pickle(args.features)
        if args.features.exists()
        else add_features(load_panel(range(2016, 2026)))
    )

    payload: dict[str, object] = {"folds": []}
    for holdout in args.holdouts:
        train = frame[frame["season"] < holdout]
        test = frame[frame["season"] == holdout]
        if train.empty or test.empty:
            continue
        keep = relevant_population(test).to_numpy(bool)
        test = test[keep]
        observed = test["points"].to_numpy(float)
        played = test["played"].to_numpy(int)

        model = Hurdle(use_team=False).fit(train, train["points"].to_numpy(float))
        probability = model.play_probability(test)
        samples = model.predict_samples(test, draws=args.draws, seed=holdout)

        reference = HistoryMean(
            column="prior_points_recent", name="recency"
        ).fit(train, train["points"].to_numpy(float))
        reference_samples = reference.predict_samples(test, draws=args.draws, seed=holdout)

        on_field = played == 1
        conditional = model.magnitude_samples(test, draws=args.draws, seed=holdout)
        entry = {
            "holdout": int(holdout),
            "n": int(len(test)),
            "play_rate": float(played.mean()),
            "predicted_play_rate": float(probability.mean()),
            "reliability": reliability(probability, played).to_dict("records"),
            "pooled": {
                "hurdle_cov80": float(
                    interval_coverage(observed, samples, 0.8)["coverage"]
                ),
                "recency_cov80": float(
                    interval_coverage(observed, reference_samples, 0.8)["coverage"]
                ),
                "hurdle_pit": pit_calibration(observed, samples)["deviation"],
                "recency_pit": pit_calibration(observed, reference_samples)["deviation"],
            },
            "magnitude_on_played": {
                "n": int(on_field.sum()),
                "cov80": float(
                    interval_coverage(observed[on_field], conditional[on_field], 0.8)[
                        "coverage"
                    ]
                ),
                "cov95": float(
                    interval_coverage(observed[on_field], conditional[on_field], 0.95)[
                        "coverage"
                    ]
                ),
                "pit": pit_calibration(observed[on_field], conditional[on_field])[
                    "deviation"
                ],
                "pit_shape": pit_calibration(
                    observed[on_field], conditional[on_field]
                )["shape"],
                "bias": float(
                    conditional[on_field].mean(axis=1).mean() - observed[on_field].mean()
                ),
            },
        }
        payload["folds"].append(entry)

        print(f"\n=== {holdout} (relevant rows, n={len(test)}) ===")
        print(
            f"  play rate observed {played.mean():.3f}, predicted "
            f"{probability.mean():.3f}"
        )
        print("\n  availability reliability:")
        print("   " + reliability(probability, played).round(4).to_string(index=False))
        print(
            f"\n  pooled cov80   hurdle {entry['pooled']['hurdle_cov80']:.3f} "
            f"vs recency {entry['pooled']['recency_cov80']:.3f}"
        )
        block = entry["magnitude_on_played"]
        print(
            f"  magnitude|played  cov80 {block['cov80']:.3f}  cov95 "
            f"{block['cov95']:.3f}  bias {block['bias']:+.2f}  "
            f"PIT {block['pit']:.3f} ({block['pit_shape']})"
        )

    print(
        "\nReading it: the two halves are diagnosed separately because only the "
        "second\ncan be read off the pooled number. If the reliability columns "
        "agree, the zero\nmass is the right size. If the magnitude interval sits "
        "near 0.80 on the weeks\nhe played, the pooled 0.89 is the atom widening a "
        "central interval and not a\nmiscalibration -- a point mass at zero puts "
        "the 10th percentile at zero for\nanyone with a meaningful chance of "
        "sitting out, so every positive outcome\nclears it from below."
    )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, default=str), "utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
