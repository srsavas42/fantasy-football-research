"""Every arm's disagreement with the board, scored the way a drafter uses it.

MAE says which forecast is closer on average. It does not say whether a forecast
is *usable at a draft table*, because a model that beats ADP by being a smoothed
copy of ADP tells you nothing you did not already have. What matters is whether
the direction of a disagreement predicts which way the board is wrong.

Two statistics per arm, against the ADP rank curve:

    corr    does the disagreement point the right way at all
    slope   how much real deviation a unit of disagreement buys

Slope is the more useful of the two. At 1.0 a disagreement is correctly sized:
if the arm says a player beats his board projection by twenty points, he does, on
average. Below 1.0 the arm overstates itself and should be shrunk toward the
board. Above 1.0 it is too timid and should be pushed further.

Reads the per-row means the flat-baseline run saves, so it costs nothing beyond
that run.

    python scripts/screen_directional_arms.py
"""
import warnings; warnings.filterwarnings("ignore")
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ARMS = ("pipeline", "flat_no_adp", "gbm_no_adp", "blend", "flat_ridge", "flat_gbm")
TIERS = (("top50", 1, 50), ("51_150", 51, 150), ("151_300", 151, 300))


def load(paths):
    blocks = []
    for path in paths:
        if not Path(path).exists():
            continue
        blob = json.loads(Path(path).read_text("utf-8"))
        for fold in blob["folds"].values():
            if "rows" in fold:
                blocks.append({k: np.array(v, dtype=float) for k, v in fold["rows"].items()})
    if not blocks:
        raise SystemExit("no per-row blocks found; re-run validate_flat_baseline.py")
    return {k: np.concatenate([b[k] for b in blocks]) for k in blocks[0]}


def main() -> int:
    rows = load([f"reports/flat_{y}.json" for y in (2023, 2024, 2025)])
    observed, adp, rank = rows["observed"], rows["adp"], rows["rank"]
    print(f"{len(observed)} drafted player-seasons\n")

    print(f"{'arm':14} {'MAE':>8} {'vs ADP':>8} {'corr':>8} {'slope':>8} {'p':>10}")
    print("-" * 62)
    base = np.abs(observed - adp).mean()
    print(f"{'adp':14} {base:8.2f} {'--':>8} {'--':>8} {'--':>8} {'--':>10}")
    for arm in ARMS:
        if arm not in rows:
            continue
        prediction = rows[arm]
        gap, error = prediction - adp, observed - adp
        keep = np.isfinite(gap) & np.isfinite(error)
        if np.std(gap[keep]) < 1e-9:
            continue
        r, p = stats.pearsonr(gap[keep], error[keep])
        slope = np.polyfit(gap[keep], error[keep], 1)[0]
        mae = np.abs(observed - prediction).mean()
        print(f"{arm:14} {mae:8.2f} {(mae - base) / base:+8.1%} {r:+8.3f} "
              f"{slope:+8.3f} {p:10.2g}")

    print("\n  slope 1.0 = correctly sized; below = overstated; above = too timid\n")
    print("by draft tier -- correlation of disagreement with the board's error")
    print(f"  {'arm':14}" + "".join(f"{t[0]:>12}" for t in TIERS))
    print("  " + "-" * (14 + 12 * len(TIERS)))
    for arm in ARMS:
        if arm not in rows:
            continue
        cells = []
        for _, low, high in TIERS:
            mask = np.isfinite(rank) & (rank >= low) & (rank <= high)
            gap, error = rows[arm][mask] - adp[mask], observed[mask] - adp[mask]
            ok = np.isfinite(gap) & np.isfinite(error)
            if ok.sum() < 30 or np.std(gap[ok]) < 1e-9:
                cells.append(f"{'--':>12}")
                continue
            cells.append(f"{stats.pearsonr(gap[ok], error[ok])[0]:+12.3f}")
        print(f"  {arm:14}" + "".join(cells))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
