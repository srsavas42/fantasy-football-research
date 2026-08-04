"""The promoted default changes what a served artifact does, so check it serves.

cold_role_multiplier is fitted, not configured. A pipeline that saved the flag
and the mode but lost the number would load without complaint and quietly serve
one scale for rookies and starters alike -- the exact defect the flag exists to
fix, now invisible. The unit test pins this on a synthetic model; this pins it
on a real fit and a real prediction.
"""
import warnings; warnings.filterwarnings("ignore")
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from ffmodel.features.season_average import SeasonAverageData
from ffmodel.models.volume_season_average import SeasonAverageVolumePipeline

pr = pd.read_pickle(".cache/ffmodel-wf-2025/player_rows.pkl")
tr = pd.read_pickle(".cache/ffmodel-wf-2025/team_rows.pkl")
train = SeasonAverageData(tr[tr.season < 2025].copy(), pr[pr.season < 2025].copy())
test = SeasonAverageData(tr[tr.season == 2025].copy(), pr[pr.season == 2025].copy())

pipe = SeasonAverageVolumePipeline().fit(train, draws=250, tune=250, chains=2)
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
