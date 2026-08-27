"""Blend the weekly rest-of-season model with the draft board, by horizon.

The weekly model beats a naive ADP curve everywhere except the draft, where it
loses on every metric. That is not a surprise and not a bug: in week 1 the model
has no in-season information, and the board carries an offseason the box scores
have not recorded yet -- a trade, a rookie, a vacated backfield. By week 5 the
model has usage the board will never see and wins by 10%; by week 11, by 17%.

So the weight on the model should not be one number. It should start low and
rise as the season gives the model something the board does not have, which is
exactly what the variance-optimal weight does on its own if it is estimated
separately per horizon:

    observed - curve = a + b * (model - curve)

``b`` is the share of the model's disagreement with the board that turns out to
be right. Estimated within each horizon bucket, it *is* the schedule this
document is arguing for, read off the data rather than imposed.

Two choices carried over from the season layer's blend, both load bearing:

**Mixture, not averaging.** Each draw comes from one forecast or the other.
Averaging paired draws gives the same mean and a distribution narrower than
either input, which wrecks calibration -- 0.689 against a nominal 0.80 at season
level. This package's output is a distribution, so the narrow one is not a
candidate whatever its MAE.

**The weight comes from earlier holdouts only.** Fitting it on the fold being
scored would be reporting an in-sample number. The first holdout therefore has
no weight to use and goes unscored, exactly as 2022 did at season level.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from ffmodel.models.market_blend import blend_samples, slope_weight
from ffmodel.weekly.evaluate import score
from ffmodel.weekly.features import add_features, relevant_population
from ffmodel.weekly.frame import load_panel
from ffmodel.weekly.market import WeeklyRankCurve, attach_adp
from ffmodel.weekly.restofseason import OFFSET, TARGET, DirectTotal, add_rest_of_season_target

BUCKETS = (("early", 1, 4), ("mid", 5, 10), ("late", 11, 18))


def _bucket_labels(week: np.ndarray) -> np.ndarray:
    labels = np.empty(len(week), dtype=object)
    for name, low, high in BUCKETS:
        labels[(week >= low) & (week <= high)] = name
    return labels


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--holdouts", type=int, nargs="+", default=[2023, 2024, 2025])
    parser.add_argument("--draws", type=int, default=800)
    parser.add_argument(
        "--features", type=Path, default=Path(".cache/weekly_features_2016_2025.pkl")
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    frame = (
        pd.read_pickle(args.features)
        if args.features.exists()
        else add_features(attach_adp(load_panel(range(2016, 2026))))
    )
    frame = add_rest_of_season_target(frame)

    holdouts = sorted(args.holdouts)
    history: list[dict] = []
    payload: dict[str, object] = {"folds": [], "buckets": [b[0] for b in BUCKETS]}

    for holdout in holdouts:
        train = frame[frame["season"] < holdout]
        test = frame[frame["season"] == holdout]
        if train.empty or test.empty:
            continue
        keep = relevant_population(test).to_numpy(bool)
        drafted = pd.to_numeric(test["adp_drafted"], errors="coerce").eq(1).to_numpy()
        test = test[keep & drafted]
        if test.empty:
            continue

        observed = test[TARGET].to_numpy(float)
        week = pd.to_numeric(test["week"], errors="coerce").to_numpy(float)
        labels = _bucket_labels(week)
        position = test["position"].astype(str).to_numpy()

        model = DirectTotal(use_team=True, use_phase=True, use_adp=True).fit(
            train, train[TARGET].to_numpy(float)
        )
        curve = WeeklyRankCurve(per_game=False, offset=OFFSET).fit(
            train, train["points"].to_numpy(float)
        )
        model_samples = model.predict_samples(test, draws=args.draws, seed=holdout)
        curve_samples = curve.predict_samples(test, draws=args.draws, seed=holdout)

        # Weights from earlier holdouts only. The first fold has none and is
        # recorded but not scored.
        weights: dict[str, float] = {}
        for name, _, _ in BUCKETS:
            rows = [h for h in history if h["bucket"] == name]
            if not rows:
                continue
            weights[name] = slope_weight(
                np.concatenate([r["observed"] for r in rows]),
                np.concatenate([r["model"] for r in rows]),
                np.concatenate([r["curve"] for r in rows]),
            )

        entry: dict[str, object] = {
            "holdout": int(holdout),
            "n": int(len(test)),
            "weights": {k: round(v, 4) for k, v in weights.items()},
            "scored": bool(weights),
        }

        if weights:
            blended = model_samples.copy()
            for name, _, _ in BUCKETS:
                want = labels == name
                if not want.any() or name not in weights:
                    continue
                blended[want] = blend_samples(
                    model_samples[want],
                    curve_samples[want],
                    weights[name],
                    seed=holdout + 1,
                )
            entry["scores"] = {
                "model": score(observed, model_samples, groups=position),
                "curve": score(observed, curve_samples, groups=position),
                "blend": score(observed, blended, groups=position),
            }
            for name, _, _ in BUCKETS:
                want = labels == name
                if want.sum() < 50:
                    continue
                entry[f"scores_{name}"] = {
                    "model": score(observed[want], model_samples[want], groups=position[want]),
                    "curve": score(observed[want], curve_samples[want], groups=position[want]),
                    "blend": score(observed[want], blended[want], groups=position[want]),
                }
        payload["folds"].append(entry)

        # Feed this fold forward so the next one has a weight to use.
        for name, _, _ in BUCKETS:
            want = labels == name
            if not want.any():
                continue
            history.append(
                {
                    "bucket": name,
                    "observed": observed[want],
                    "model": model_samples[want].mean(axis=1),
                    "curve": curve_samples[want].mean(axis=1),
                }
            )

    for fold in payload["folds"]:
        print(f"\n=== {fold['holdout']} (drafted, n={fold['n']}) ===")
        if not fold["scored"]:
            print("  no earlier holdout to take a weight from; not scored")
            continue
        print(f"  weights: {fold['weights']}")
        for label in ("", "early", "mid", "late"):
            key = "scores" if not label else f"scores_{label}"
            if key not in fold:
                continue
            block = fold[key]
            print(f"  -- {label or 'all'} --")
            for arm in ("curve", "model", "blend"):
                row = block[arm]
                print(
                    f"     {arm:6s} MAE {row['mae']:7.2f}  CRPS {row['crps']:7.2f}  "
                    f"cov80 {row['coverage_80']:.3f}  rho {row.get('within_group_spearman', float('nan')):.3f}"
                )

    print(
        "\nReading it: the blend is worth shipping only if it beats *both* the "
        "curve and\nthe model where each of them is strong -- the curve early, "
        "the model late. A\nblend that merely lands between them has bought "
        "nothing."
    )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, default=str), "utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
