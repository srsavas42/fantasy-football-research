"""Task 31, continued: the tail deficit is not the efficiency layer. Whose is it?

Scaling the efficiency dispersion 0->3x moved PPR cov95 from 0.906 to 0.925
against a 0.950 nominal, at a rising MAE cost. Tripling a layer's noise and
recovering a third of a coverage gap is that layer being ruled out, not
implicated. Two candidates remain, and this tests both from one fit:

  (a) the volume draws. Widen the count draws around their own row means and
      rescore. If the deficit is volume's, a modest k closes it.
  (b) volume-efficiency dependence. The accepted path draws efficiency from
      independent marginals; ``draw_conditioned_efficiency`` evaluates the
      fitted efficiency means at each volume draw, so a player who draws a
      heavy workload also draws the efficiency that goes with it. Positive
      dependence fattens both tails of the product.
"""
import json
import warnings
from dataclasses import replace
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from ffmodel.evaluation.efficiency_posterior import score_fantasy_points_posterior
from ffmodel.features.season_average import SeasonAverageData
from ffmodel.models.season_scoring import SeasonAverageScoringPipeline
from ffmodel.simulation.season_scoring import score_volume_prediction

OUTPUT = Path("scripts/validation_runs/tail_attribution_2025.json")

pr = pd.read_pickle(".cache/ffmodel-wf-2025/player_rows.pkl")
tr = pd.read_pickle(".cache/ffmodel-wf-2025/team_rows.pkl")
train = SeasonAverageData(tr[tr.season < 2025].copy(), pr[pr.season < 2025].copy())
test = SeasonAverageData(tr[tr.season == 2025].copy(), pr[pr.season == 2025].copy())

pipeline = SeasonAverageScoringPipeline()
sample_kwargs = {"draws": 1000, "tune": 1000, "chains": 4}
pipeline.fit(
    train, volume_sample_kwargs=sample_kwargs, efficiency_sample_kwargs=sample_kwargs
)
print("fitted", flush=True)

volume = pipeline.volume_model.predict_samples(test, seed=42)

COUNT_FIELDS = ("pass_attempts", "targets", "carries")
RATE_FIELDS = tuple(
    f"{stream}_per_{basis}"
    for stream in COUNT_FIELDS
    for basis in ("team_game", "active_game")
)


def widen(prediction, k: float):
    """Volume draws stretched k-fold about each row's own posterior mean.

    Counts stay integers and non-negative; the per-game rates the efficiency
    exposures read are stretched by the same factor so the two stay coherent.
    """
    if k == 1.0:
        return prediction
    updates = {}
    for name in COUNT_FIELDS:
        draws = np.asarray(getattr(prediction, name), dtype=float)
        centre = draws.mean(axis=1, keepdims=True)
        updates[name] = np.rint(
            np.clip(centre + k * (draws - centre), 0.0, None)
        ).astype(int)
    for name in RATE_FIELDS:
        draws = np.asarray(getattr(prediction, name), dtype=float)
        centre = draws.mean(axis=1, keepdims=True)
        updates[name] = np.clip(centre + k * (draws - centre), 0.0, None)
    return replace(prediction, **updates)


results: dict[str, dict] = {"volume_width": {}, "dependence": {}}

print("\n(a) VOLUME WIDTH — PPR, 2025 out of sample\n")
print(f"{'volume width':>14s} {'cov80':>7s} {'cov95':>7s} {'MAE':>8s} {'CRPS':>8s}")
for k in (1.0, 1.25, 1.5, 2.0, 3.0):
    scored = score_volume_prediction(
        widen(volume, k), pipeline.efficiency_model, seed=42 + 10_000
    )
    summary = score_fantasy_points_posterior(scored, scoring="ppr")
    results["volume_width"][str(k)] = summary
    print(
        f"{k:>14.2f} {summary['coverage_80']:>7.3f} {summary['coverage_95']:>7.3f} "
        f"{summary['mae']:>8.3f} {summary['crps']:>8.3f}",
        flush=True,
    )

print("\n(b) VOLUME-EFFICIENCY DEPENDENCE — PPR, 2025 out of sample\n")
print(f"{'path':>22s} {'cov80':>7s} {'cov95':>7s} {'MAE':>8s} {'CRPS':>8s}")
for label, conditioned in (
    ("independent (accepted)", False),
    ("draw-conditioned", True),
):
    scored = score_volume_prediction(
        volume,
        pipeline.efficiency_model,
        draw_conditioned_efficiency=conditioned,
        seed=42 + 10_000,
    )
    summary = score_fantasy_points_posterior(scored, scoring="ppr")
    results["dependence"][label] = summary
    print(
        f"{label:>22s} {summary['coverage_80']:>7.3f} {summary['coverage_95']:>7.3f} "
        f"{summary['mae']:>8.3f} {summary['crps']:>8.3f}",
        flush=True,
    )

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
OUTPUT.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
print(f"\nwrote {OUTPUT}")
