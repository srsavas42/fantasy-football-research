"""Is the quarterback-quality effect real, or a receiver predicting himself?

``teammate_qb_quality_signal`` attaches the projected starter's prior-season
passing quality to every skill-position row on his team. For a receiver who kept
his quarterback, that composite is partly his own doing: his catches and yards
fed the passer's numbers last season, so the feature carries a shadow of his own
prior production and any gain may be autocorrelation wearing a teammate's name.

For a receiver who changed teams, the new quarterback's prior quality was earned
without him. That subset is the clean test, and it is where a genuine
cross-positional effect has to show up.

Three outcomes and what each means:

* **Both groups gain.** The effect is real and the confound, if present, is not
  carrying it.
* **Only movers gain.** Real, and stronger than the pooled number suggests,
  since the pooled number is diluted by players for whom the feature adds
  little new information.
* **Only stayers gain.** Circularity. The feature is a lagged self-signal, and
  the pooled improvement should be discarded rather than promoted.

Scored on total fantasy points, because that is what the package publishes and
because a receiving-efficiency metric would flatter a feature that only moves
receiving efficiency.
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
from ffmodel.evaluation.metrics import empirical_crps
from ffmodel.features.season_average import SeasonAverageData
from ffmodel.models.season_scoring import SeasonAverageScoringPipeline
from ffmodel.simulation.scoring import fantasy_points

RECEIVING_POSITIONS = ("RB", "WR", "TE")
# The gate's own floor. A verdict without one calls 0.01% a finding.
MATERIAL = 0.0025


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--holdout", type=int, default=2024)
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/ffmodel-wf-2025"))
    parser.add_argument("--scoring", default="ppr")
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    output = args.output or Path(
        f"scripts/validation_runs/teammate_split_{args.holdout}.json"
    )

    pr = pd.read_pickle(args.cache_dir / "player_rows.pkl")
    tr = pd.read_pickle(args.cache_dir / "team_rows.pkl")
    train = SeasonAverageData(
        tr[tr.season < args.holdout].copy(), pr[pr.season < args.holdout].copy()
    )
    test = SeasonAverageData(
        tr[tr.season == args.holdout].copy(), pr[pr.season == args.holdout].copy()
    )
    sample_kwargs = {"draws": args.draws, "tune": args.draws, "chains": 4}

    arms: dict[str, np.ndarray] = {}
    rows = None
    for label, enabled in (("baseline", False), ("teammate", True)):
        pipeline = SeasonAverageScoringPipeline(teammate_quality_features=enabled)
        pipeline.fit(
            train,
            volume_sample_kwargs=sample_kwargs,
            efficiency_sample_kwargs=sample_kwargs,
        )
        prediction = pipeline.predict_samples(test, seed=42)
        arms[label] = np.asarray(
            prediction.fantasy_points[args.scoring], dtype=float
        )
        rows = prediction.player_rows.reset_index(drop=True)
        print(f"fitted {label}", flush=True)

    observed = fantasy_points(observed_scoring_rows(rows), args.scoring).to_numpy(float)
    named = (
        pd.to_numeric(
            rows.get("is_replacement_player", pd.Series(0, index=rows.index)),
            errors="coerce",
        )
        .fillna(0)
        .ne(1)
        .to_numpy()
    )
    changed = (
        pd.to_numeric(rows.get("team_change"), errors="coerce").fillna(0).eq(1).to_numpy()
    )
    receiving = rows["position"].isin(RECEIVING_POSITIONS).to_numpy()
    valid = named & np.isfinite(observed) & receiving
    for samples in arms.values():
        valid &= np.isfinite(samples).all(axis=1)

    report: dict[str, object] = {
        "holdout": args.holdout,
        "scoring": args.scoring,
        "groups": {},
    }
    print(f"\nTEAMMATE QUALITY BY WHETHER THE PLAYER MOVED, {args.holdout} "
          f"{args.scoring}\n")
    print(f"  {'group':22s} {'n':>5s} {'base MAE':>9s} {'tm MAE':>8s} {'d MAE':>8s} "
          f"{'base CRPS':>10s} {'tm CRPS':>8s} {'d CRPS':>8s}")
    for label, mask in (
        ("changed team", valid & changed),
        ("stayed", valid & ~changed),
        ("all receiving", valid),
    ):
        if mask.sum() < 10:
            continue
        truth = observed[mask]
        entry: dict[str, float] = {"n": int(mask.sum())}
        for arm, samples in arms.items():
            drawn = samples[mask]
            entry[f"{arm}_mae"] = float(np.abs(truth - drawn.mean(axis=1)).mean())
            entry[f"{arm}_crps"] = float(empirical_crps(truth, drawn).mean())
        d_mae = (entry["teammate_mae"] - entry["baseline_mae"]) / entry["baseline_mae"]
        d_crps = (
            entry["teammate_crps"] - entry["baseline_crps"]
        ) / entry["baseline_crps"]
        entry["delta_mae"] = float(d_mae)
        entry["delta_crps"] = float(d_crps)
        report["groups"][label] = entry
        print(
            f"  {label:22s} {entry['n']:>5d} {entry['baseline_mae']:>9.3f} "
            f"{entry['teammate_mae']:>8.3f} {d_mae:>+7.2%} "
            f"{entry['baseline_crps']:>10.3f} {entry['teammate_crps']:>8.3f} "
            f"{d_crps:>+7.2%}"
        )

    groups = report["groups"]
    if "changed team" in groups and "stayed" in groups:
        # Signs alone are not a verdict. The first version of this keyed on the
        # sign of delta CRPS and duly announced a real effect from -0.11% and
        # -0.01%, with MAE moving the other way in both groups and 75 movers.
        # The gate's floor applies here for the same reason it applies there.
        def moved(group: dict[str, float]) -> int:
            crps, mae = group["delta_crps"], group["delta_mae"]
            if crps < -MATERIAL and mae < MATERIAL:
                return -1  # a gain, and point accuracy did not pay for it
            if crps > MATERIAL:
                return 1
            return 0

        movers = moved(groups["changed team"])
        stayers = moved(groups["stayed"])
        print()
        if movers < 0 and stayers < 0:
            verdict = "both groups gain: real, and the confound is not carrying it"
        elif movers < 0 <= stayers:
            verdict = "only movers gain: real, and the pooled number understates it"
        elif stayers < 0 <= movers:
            verdict = (
                "only stayers gain: circularity, the feature is a lagged "
                "self-signal and the pooled improvement should be discarded"
            )
        elif movers == 0 and stayers == 0:
            verdict = (
                f"neither group moves beyond {MATERIAL:.2%}: no effect to promote"
            )
        else:
            verdict = "both groups worsen: no effect to promote"
        report["verdict"] = verdict
        report["material_threshold"] = MATERIAL
        print(f"  {verdict}")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"\nwrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
