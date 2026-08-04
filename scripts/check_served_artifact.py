"""The promoted default changes what a served artifact does, so check it serves.

cold_role_multiplier is fitted, not configured. A pipeline that saved the flag
and the mode but lost the number would load without complaint and quietly serve
one scale for rookies and starters alike -- the exact defect the flag exists to
fix, now invisible. The unit test pins this on a synthetic model; this pins it
on a real fit and a real prediction.
"""
import argparse
import tempfile
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from ffmodel.features.season_average import SeasonAverageData
from ffmodel.models.volume_season_average import SeasonAverageVolumePipeline

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--holdout", type=int, default=2025)
parser.add_argument("--cache-dir", type=Path, default=Path(".cache/ffmodel-wf-2025"))
# A round trip does not need a converged posterior, only a real one: this is
# checking that fitted metadata survives save and load, not what it contains.
parser.add_argument("--draws", type=int, default=250)
args = parser.parse_args()

pr = pd.read_pickle(args.cache_dir / "player_rows.pkl")
tr = pd.read_pickle(args.cache_dir / "team_rows.pkl")
train = SeasonAverageData(
    tr[tr.season < args.holdout].copy(), pr[pr.season < args.holdout].copy()
)
test = SeasonAverageData(
    tr[tr.season == args.holdout].copy(), pr[pr.season == args.holdout].copy()
)

pipe = SeasonAverageVolumePipeline().fit(
    train, draws=args.draws, tune=args.draws, chains=2
)
print("\ndefaults on the fitted pipeline:",
      pipe.cold_role_innovation, pipe.cold_role_scale_mode, flush=True)
for name in ("target", "carry"):
    m = getattr(pipe, f"{name}_model")
    print(f"  {name}: mode={m.cold_role_scale_mode} base={m.role_innovation_scale:.4f} "
          f"mult={m.cold_role_multiplier:.4f} -> cold "
          f"{m.role_innovation_scale * m.cold_role_multiplier:.4f}")

with tempfile.TemporaryDirectory() as tmp:
    path = pipe.save(Path(tmp) / "artifact")
    restored = SeasonAverageVolumePipeline.load(path)
    print("\nrestored:", restored.cold_role_innovation, restored.cold_role_scale_mode)
    for name in ("target", "carry"):
        a, b = getattr(pipe, f"{name}_model"), getattr(restored, f"{name}_model")
        assert b.cold_role_innovation == a.cold_role_innovation, name
        assert b.cold_role_scale_mode == a.cold_role_scale_mode, name
        assert np.isclose(b.cold_role_multiplier, a.cold_role_multiplier), name
        print(f"  {name}: multiplier {a.cold_role_multiplier:.6f} -> "
              f"{b.cold_role_multiplier:.6f}")
    before = pipe.predict_samples(test, seed=7).carries_per_team_game
    after = restored.predict_samples(test, seed=7).carries_per_team_game
    print("\nserved prediction identical after round trip:",
          np.array_equal(before, after))
