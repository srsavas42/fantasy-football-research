"""Total-season scoring walk-forward, self-contained (no cached checkpoints).

validate_season_scoring_posteriors.py loads pre-fitted posteriors from
.cache/season-average-validation/..., which do not exist in a fresh container.
This fits both layers per holdout instead, so the gate can be re-run from data.
"""
import warnings; warnings.filterwarnings("ignore")
import json, os, sys, time
import numpy as np, pandas as pd

from ffmodel.features.season_average import SeasonAverageData
from ffmodel.models.season_scoring import SeasonAverageScoringPipeline
from ffmodel.evaluation.efficiency_posterior import score_fantasy_points_posterior

SCRATCH = "/tmp/claude-0/-home-user-fantasy-football-research/131829d3-0b2b-5048-ab68-b632c37f56e0/scratchpad"
label, draws, tune, chains = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])

pr = pd.read_pickle(f"{SCRATCH}/nfl_pr.pkl")
tr = pd.read_pickle(f"{SCRATCH}/nfl_tr.pkl")
kwargs = {"draws": draws, "tune": tune, "chains": chains}

out = {}
for holdout in (2022, 2023, 2024):
    started = time.perf_counter()
    train = SeasonAverageData(tr[tr.season < holdout].copy(), pr[pr.season < holdout].copy())
    test = SeasonAverageData(tr[tr.season == holdout].copy(), pr[pr.season == holdout].copy())
    pipeline = SeasonAverageScoringPipeline()
    if os.environ.get("FFMODEL_COUPLE_GATE") == "1":
        pipeline.volume_model.workload_model.couple_gate_to_availability = True
    pipeline.fit(train, volume_sample_kwargs=kwargs, efficiency_sample_kwargs=kwargs)
    prediction = pipeline.predict_samples(test, seed=42)
    fold = {
        scoring: score_fantasy_points_posterior(prediction, scoring=scoring)
        for scoring in ("standard", "half_ppr", "ppr")
    }
    fold["seconds"] = round(time.perf_counter() - started, 1)
    out[str(holdout)] = fold
    print(f"[{label}] holdout {holdout} done in {fold['seconds']}s "
          f"ppr cov95={fold['ppr']['coverage_95']:.3f}", flush=True)

json.dump(out, open(f"{SCRATCH}/scoring_{label}.json", "w"), indent=2, sort_keys=True)
print(f"[{label}] wrote scoring_{label}.json")
