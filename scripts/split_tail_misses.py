"""Task 34: is the tail deficit spread over every row, or concentrated?

A global width sweep put the missing mass in the volume layer and showed it is
shape rather than scale -- at a 2x stretch the 95% level lands on nominal while
the 80% level breaks. That sweep scales every row identically, so it cannot tell
"every row slightly too thin" from "a minority badly too thin". The two have
different fixes: the first is a dispersion parameter, the second is a mixture.

This splits the misses three ways.

* **PIT.** Uniform in the body with excess only in the end bins is a tail
  problem. A hump anywhere else is a location or width problem.
* **Magnitude.** How far outside the interval a miss lands, in half-widths. A
  slightly thin predictive misses by a little; a missing mixture component
  misses by a lot.
* **Covariates.** If the misses concentrate in rows whose role changed --
  low continuity, a new team, thin prior exposure -- the mixture reading is
  right and ``SeasonRegimeModel`` is the instrument. If they are flat across
  those splits, it is the count likelihoods' tails.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from ffmodel.evaluation.efficiency_posterior import observed_scoring_rows
from ffmodel.features.season_average import SeasonAverageData
from ffmodel.models.season_scoring import SeasonAverageScoringPipeline
from ffmodel.simulation.scoring import fantasy_points

_parser = argparse.ArgumentParser(description=__doc__)
_parser.add_argument("--holdout", type=int, default=2025)
_parser.add_argument("--scoring", default="ppr")
_parser.add_argument("--cache-dir", type=Path, default=Path(".cache/ffmodel-wf-2025"))
_parser.add_argument(
    "--cold-role-innovation",
    action="store_true",
    help="widen the role innovation for players with no prior role. Run with "
         "and without to check the fix lands on the rows it was built for "
         "rather than merely moving the aggregate",
)
_args = _parser.parse_args()

HOLDOUT = _args.holdout
SCORING = _args.scoring
SUFFIX = "_coldrole" if _args.cold_role_innovation else ""
OUTPUT = Path(f"scripts/validation_runs/tail_miss_split_{HOLDOUT}{SUFFIX}.json")


def binomial_z(misses: int, n: int, nominal: float) -> float:
    expected = (1.0 - nominal) * n
    return (misses - expected) / np.sqrt(n * nominal * (1.0 - nominal))


pr = pd.read_pickle(_args.cache_dir / "player_rows.pkl")
tr = pd.read_pickle(_args.cache_dir / "team_rows.pkl")
train = SeasonAverageData(
    tr[tr.season < HOLDOUT].copy(), pr[pr.season < HOLDOUT].copy()
)
test = SeasonAverageData(
    tr[tr.season == HOLDOUT].copy(), pr[pr.season == HOLDOUT].copy()
)

pipeline = SeasonAverageScoringPipeline()
pipeline.volume_model.cold_role_innovation = _args.cold_role_innovation
sample_kwargs = {"draws": 1000, "tune": 1000, "chains": 4}
pipeline.fit(
    train, volume_sample_kwargs=sample_kwargs, efficiency_sample_kwargs=sample_kwargs
)
print("fitted", flush=True)
prediction = pipeline.predict_samples(test, seed=42)

rows = prediction.player_rows.reset_index(drop=True)
observed = fantasy_points(observed_scoring_rows(rows), SCORING).to_numpy(dtype=float)
samples = np.asarray(prediction.fantasy_points[SCORING], dtype=float)
replacement = (
    pd.to_numeric(
        rows.get("is_replacement_player", pd.Series(0, index=rows.index)),
        errors="coerce",
    )
    .fillna(0)
    .ne(1)
    .to_numpy()
)
valid = np.isfinite(observed) & np.isfinite(samples).all(axis=1) & replacement

rows = rows.loc[valid].reset_index(drop=True)
observed = observed[valid]
samples = samples[valid]
n = len(observed)
print(f"scoring {n} rows on {HOLDOUT} {SCORING}\n")

lower95, upper95 = np.quantile(samples, [0.025, 0.975], axis=1)
lower80, upper80 = np.quantile(samples, [0.10, 0.90], axis=1)
below = observed < lower95
above = observed > upper95
miss95 = below | above
miss80 = (observed < lower80) | (observed > upper80)

report: dict[str, object] = {
    "holdout": HOLDOUT,
    "scoring": SCORING,
    "cold_role_innovation": bool(_args.cold_role_innovation),
    "n": n,
    "misses_95": int(miss95.sum()),
    "z95": float(binomial_z(int(miss95.sum()), n, 0.95)),
    "misses_80": int(miss80.sum()),
    "z80": float(binomial_z(int(miss80.sum()), n, 0.80)),
}
print(
    f"95%: {report['misses_95']} misses vs {0.05 * n:.1f} expected, "
    f"z={report['z95']:+.2f}   ({int(below.sum())} below, {int(above.sum())} above)"
)
print(
    f"80%: {report['misses_80']} misses vs {0.20 * n:.1f} expected, "
    f"z={report['z80']:+.2f}\n"
)
report["below_95"] = int(below.sum())
report["above_95"] = int(above.sum())

# --- PIT ------------------------------------------------------------------
pit = (samples < observed[:, None]).mean(axis=1)
counts, _ = np.histogram(pit, bins=10, range=(0.0, 1.0))
expected = n / 10.0
print("PIT histogram (uniform if calibrated)\n")
print(f"  {'bin':>10s} {'count':>6s} {'expected':>9s} {'z':>7s}")
for index, count in enumerate(counts):
    z = (count - expected) / np.sqrt(n * 0.1 * 0.9)
    bar = "#" * int(round(count / max(counts.max(), 1) * 30))
    print(
        f"  {index / 10:.1f}-{(index + 1) / 10:.1f} {count:>6d} {expected:>9.1f} "
        f"{z:>+7.2f}  {bar}"
    )
report["pit_counts"] = [int(c) for c in counts]

# --- how badly do the misses miss? ---------------------------------------
half_width = np.maximum((upper95 - lower95) / 2.0, 1e-9)
excess = np.where(
    below, (lower95 - observed) / half_width, (observed - upper95) / half_width
)
missed = excess[miss95]
print("\nHow far outside the 95% interval the misses land (in half-widths)\n")
for label, threshold in (
    ("within 0.25", 0.25),
    ("within 0.50", 0.50),
    ("within 1.00", 1.00),
    ("beyond 1.00", np.inf),
):
    if np.isinf(threshold):
        count = int((missed > 1.0).sum())
    else:
        count = int((missed <= threshold).sum())
    print(f"  {label:>12s} {count:>4d}  ({count / max(len(missed), 1):>5.1%})")
print(f"  median excess {np.median(missed):.3f}, max {missed.max():.3f}")
report["miss_excess_median"] = float(np.median(missed))
report["miss_excess_max"] = float(missed.max())
report["miss_excess_beyond_one"] = int((missed > 1.0).sum())


# --- covariate splits -----------------------------------------------------
def numeric(column: str) -> np.ndarray:
    return pd.to_numeric(
        rows.get(column, pd.Series(np.nan, index=rows.index)), errors="coerce"
    ).to_numpy(float)


continuity = numeric("prior_role_continuity")
snap = numeric("prior_snap_share")
experience = numeric("experience")
changed = numeric("prior_role_team_change")
splits: dict[str, list[tuple[str, np.ndarray]]] = {
    "position": [
        (pos, rows["position"].eq(pos).to_numpy()) for pos in ("QB", "RB", "WR", "TE")
    ],
    "prior role continuity": [
        ("missing", ~np.isfinite(continuity)),
        ("low   (<0.33)", continuity < 0.33),
        ("mid   (0.33-0.67)", (continuity >= 0.33) & (continuity < 0.67)),
        ("high  (>=0.67)", continuity >= 0.67),
    ],
    "prior snap share": [
        ("none/missing", ~np.isfinite(snap) | (snap <= 0.0)),
        ("thin  (0-0.25]", (snap > 0.0) & (snap <= 0.25)),
        ("mid   (0.25-0.60]", (snap > 0.25) & (snap <= 0.60)),
        ("heavy (>0.60)", snap > 0.60),
    ],
    "experience": [
        ("rookie", experience <= 0),
        ("second year", experience == 1),
        ("3-5", (experience >= 2) & (experience <= 4)),
        ("veteran (6+)", experience >= 5),
    ],
    "changed team": [
        ("stayed", changed == 0),
        ("changed", changed == 1),
    ],
}

print("\nMISS RATE BY ROW TYPE (95% level, nominal 5%)\n")
report["splits"] = {}
for name, bands in splits.items():
    print(f"  {name}")
    print(
        f"    {'band':22s} {'n':>4s} {'miss':>5s} {'rate':>7s} {'z':>7s} "
        f"{'below':>6s} {'above':>6s} {'med excess':>11s}"
    )
    report["splits"][name] = {}
    for label, mask in bands:
        mask = np.asarray(mask, dtype=bool)
        if mask.sum() < 5:
            continue
        m = int(miss95[mask].sum())
        rate = m / mask.sum()
        z = binomial_z(m, int(mask.sum()), 0.95)
        band_excess = excess[mask & miss95]
        print(
            f"    {label:22s} {int(mask.sum()):>4d} {m:>5d} {rate:>7.1%} {z:>+7.2f} "
            f"{int(below[mask].sum()):>6d} {int(above[mask].sum()):>6d} "
            f"{np.median(band_excess) if len(band_excess) else float('nan'):>11.3f}"
        )
        report["splits"][name][label] = {
            "n": int(mask.sum()),
            "misses": m,
            "rate": float(rate),
            "z": float(z),
            "below": int(below[mask].sum()),
            "above": int(above[mask].sum()),
        }

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
OUTPUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
print(f"\nwrote {OUTPUT}")
