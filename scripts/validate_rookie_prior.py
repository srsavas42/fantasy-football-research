"""Walk-forward validation of the rookie draft-capital prior.

Fits the claim curves on training seasons only, decides promotion per
position/stream by walk-forward inside those seasons, and then scores the
resulting policy once on a holdout season that no fit or promotion decision has
seen. Selecting which streams to promote on the holdout itself would report a
number the model cannot reproduce out of sample.

    python scripts/validate_rookie_prior.py --holdout 2025
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from ffmodel.features.draft import ROOKIE_CLAIM_CURVES
from ffmodel.features.draft_calibration import (
    CLAIM_STREAMS,
    HAND_SET_CLAIM_CURVES,
    claim_from_curve,
    fit_rookie_priors,
    rookie_seasons,
)
from ffmodel.features.season_average import build_season_average_data


def _baseline(overall_pick, position: str, stream: str) -> float:
    """The hand-set curve this calibration has to beat."""
    return claim_from_curve(
        overall_pick, HAND_SET_CLAIM_CURVES.get((position, stream), (0.0, 60.0))
    )


def _errors(rookies: pd.DataFrame, position: str, stream: str, predict) -> tuple:
    column = CLAIM_STREAMS[stream]
    sub = rookies[rookies["position"].astype(str).eq(position)]
    if sub.empty or column not in sub.columns:
        return 0, 0.0, 0.0
    actual = pd.to_numeric(sub[column], errors="coerce").to_numpy(dtype=float)
    keep = np.isfinite(actual)
    if not keep.any():
        return 0, 0.0, 0.0
    predicted = np.array([predict(p) for p in sub["overall_pick"]], dtype=float)[keep]
    error = predicted - actual[keep]
    return int(keep.sum()), float(np.abs(error).sum()), float((error**2).sum())


def walk_forward_promotions(rows: pd.DataFrame, seasons) -> set:
    """Streams whose fit beats the shipped curve on both MAE and RMSE."""
    totals: dict = {}
    for season in seasons:
        fitted = fit_rookie_priors(rows[rows["season"].lt(season)])
        rookies = rookie_seasons(rows[rows["season"].eq(season)])
        for key, curve in fitted.items():
            position, stream = key
            n, base_ae, base_se = _errors(
                rookies, position, stream, lambda p: _baseline(p, position, stream)
            )
            _, fit_ae, fit_se = _errors(
                rookies, position, stream, lambda p, c=curve: claim_from_curve(p, c)
            )
            entry = totals.setdefault(key, [0, 0.0, 0.0, 0.0, 0.0])
            entry[0] += n
            entry[1] += base_ae
            entry[2] += fit_ae
            entry[3] += base_se
            entry[4] += fit_se
    promoted = set()
    for key, (n, base_ae, fit_ae, base_se, fit_se) in totals.items():
        if n < 10 or max(base_ae, fit_ae) / max(n, 1) < 1e-4:
            continue
        if fit_ae < base_ae and fit_se < base_se:
            promoted.add(key)
    return promoted


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--holdout", type=int, default=2025)
    parser.add_argument("--first-season", type=int, default=2015)
    args = parser.parse_args()

    rows = build_season_average_data(
        range(args.first_season - 1, args.holdout + 1), source="auto"
    ).player_rows
    train = rows[rows["season"].lt(args.holdout)]
    test = rows[rows["season"].eq(args.holdout)]

    promoted = walk_forward_promotions(
        train, range(args.holdout - 5, args.holdout)
    )
    print(f"promoted by walk-forward inside training: {sorted(promoted)}\n")

    rookies = rookie_seasons(test)
    print(f"holdout {args.holdout}: {len(rookies)} rookie seasons")
    print(f"{'pos':4s}{'stream':8s}{'n':>5s}{'MAE base':>10s}{'MAE ship':>10s}{'RMSE base':>11s}{'RMSE ship':>11s}")
    for key in sorted(ROOKIE_CLAIM_CURVES):
        position, stream = key
        n, base_ae, base_se = _errors(
            rookies, position, stream, lambda p: _baseline(p, position, stream)
        )
        _, ship_ae, ship_se = _errors(
            rookies,
            position,
            stream,
            lambda p, c=ROOKIE_CLAIM_CURVES[key]: claim_from_curve(p, c),
        )
        if not n or max(base_ae, ship_ae) / n < 1e-4:
            continue
        print(
            f"{position:4s}{stream:8s}{n:5d}{base_ae / n:10.4f}{ship_ae / n:10.4f}"
            f"{np.sqrt(base_se / n):11.4f}{np.sqrt(ship_se / n):11.4f}"
        )


if __name__ == "__main__":
    main()
