"""Choose the efficiency exposure floor on an inner fold, never on the holdout.

Each efficiency response is fitted only on rows clearing its ``min_exposure``
and then scored on every row, so the fitted mean describes high-usage players
and is extrapolated onto a majority that never entered the fit — 57% of
quarterback rows, 58% of receiving rows and 82% of rushing rows sit below their
own floor. Lowering the floor admits that population. Both likelihoods already
downweight a thin row correctly, so the hard floor is doing by exclusion what
the likelihood does by weighting, and pays for it by selecting on usage.

The open question was which value to lower it to. Sweeping candidates on the
2022/2023/2024 holdouts and keeping the winner would be selecting on the test
set, which is the process risk the review recorded as S9. So this does the
selection one level in:

    for each outer holdout H:
        inner train = seasons < H-1
        inner test  = season H-1          <- the floor is chosen here
        pick the floor minimising inner error
        refit on seasons < H with that floor
        score on H                        <- the holdout is only ever scored

The floor may come out different per outer fold. That is a real result about
how stable the choice is, not a defect to average away, and it is reported
per fold.

    python scripts/select_exposure_floor.py --floors 1 3 5 10 25
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from _walkforward_data import DEFAULT_CACHE, HOLDOUTS, load_frames

from ffmodel.evaluation.metrics import empirical_crps
from ffmodel.models.efficiency_season_average import (
    SeasonAveragePosteriorEfficiencyPipeline,
)


def _score(pipeline, rows: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Per-response error over every scoreable row, floor or no floor.

    Scoring deliberately ignores ``min_exposure``: the whole question is how
    well the fit describes the rows it excluded, so restricting the score to
    the rows it kept would hide the effect being measured.
    """
    out: dict[str, dict[str, float]] = {}
    for target, model in pipeline.models.items():
        spec = model.spec
        observed = pd.to_numeric(rows.get(spec.target), errors="coerce")
        exposure = pd.to_numeric(rows.get(spec.exposure), errors="coerce")
        eligible = (
            rows["position"].astype(str).str.upper().isin(spec.positions)
            & observed.notna()
            & np.isfinite(observed)
            & exposure.fillna(0).gt(0)
        )
        subset = rows[eligible]
        if subset.empty:
            continue
        try:
            prediction = model.predict_samples(subset)
        except Exception:
            continue
        # ``predict_samples`` returns its own row frame; score against that
        # rather than ``subset``, which it may have reindexed.
        truth = pd.to_numeric(
            prediction.rows[spec.target], errors="coerce"
        ).to_numpy(float)
        rate = np.asarray(prediction.rate, dtype=float)
        keep = np.isfinite(truth)
        if not keep.any():
            continue
        truth, rate = truth[keep], rate[keep]
        out[target] = {
            "n": int(len(truth)),
            "mae": float(np.abs(truth - rate.mean(axis=1)).mean()),
            "crps": float(empirical_crps(truth, rate).mean()),
        }
    return out


def _fit(rows: pd.DataFrame, floor: int | None, sample_kwargs) -> object:
    pipeline = SeasonAveragePosteriorEfficiencyPipeline(exposure_floor=floor)
    pipeline.fit(rows, **sample_kwargs)
    return pipeline


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("label", nargs="?", default="exposure_floor")
    parser.add_argument("--floors", nargs="+", type=int, default=[1, 3, 5, 10])
    parser.add_argument("--draws", type=int, default=500)
    parser.add_argument("--tune", type=int, default=500)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--holdouts", nargs="+", type=int, default=list(HOLDOUTS))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("scripts/validation_runs")
    )
    args = parser.parse_args(argv)

    player_rows, _ = load_frames(args.cache_dir)
    sample_kwargs = {"draws": args.draws, "tune": args.tune, "chains": args.chains}
    # ``None`` means "each response keeps its own specified floor" and is the
    # incumbent. It is always a candidate, so the sweep can decline to change
    # anything.
    candidates: list[int | None] = [None, *sorted(set(args.floors))]

    report: dict[str, object] = {"candidates": [str(c) for c in candidates]}
    for holdout in args.holdouts:
        started = time.perf_counter()
        inner_test_season = holdout - 1
        inner_train = player_rows[player_rows.season < inner_test_season]
        inner_test = player_rows[player_rows.season == inner_test_season]

        inner: dict[str, dict[str, dict[str, float]]] = {}
        for floor in candidates:
            scores = _score(_fit(inner_train, floor, sample_kwargs), inner_test)
            inner[str(floor)] = scores
            print(
                f"[{holdout}] inner floor={floor}: "
                + ", ".join(f"{t} mae={s['mae']:.4f}" for t, s in sorted(scores.items())),
                flush=True,
            )

        # Compare candidates response by response against the incumbent, then
        # average the relative changes. This avoids adding a completion rate to
        # yards per attempt, which pooling raw MAE would do.
        incumbent = inner[str(None)]
        relative: dict[str, float] = {}
        for name, scores in inner.items():
            shared = set(scores) & set(incumbent)
            if not shared:
                relative[name] = float("inf")
                continue
            relative[name] = float(
                np.mean(
                    [
                        (scores[t]["mae"] - incumbent[t]["mae"]) / abs(incumbent[t]["mae"])
                        for t in sorted(shared)
                        if incumbent[t]["mae"]
                    ]
                )
            )
        chosen_name = min(relative, key=relative.get)
        chosen = None if chosen_name == "None" else int(chosen_name)
        print(
            f"[{holdout}] inner fold picks floor={chosen} "
            f"({relative[chosen_name]:+.2%} vs incumbent)",
            flush=True,
        )

        outer_train = player_rows[player_rows.season < holdout]
        outer_test = player_rows[player_rows.season == holdout]
        report[str(holdout)] = {
            "inner_test_season": inner_test_season,
            "inner_relative_mae": relative,
            "chosen_floor": chosen_name,
            "outer_incumbent": _score(_fit(outer_train, None, sample_kwargs), outer_test),
            "outer_selected": _score(_fit(outer_train, chosen, sample_kwargs), outer_test),
            "seconds": time.perf_counter() - started,
        }
        print(f"[{holdout}] done in {report[str(holdout)]['seconds']:.0f}s", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / f"{args.label}.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
