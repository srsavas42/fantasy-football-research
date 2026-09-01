"""Do injury location and tissue type predict differently, or is it one effect?

The pooled reserve flag already earns its place at availability and at receiving
efficiency. The question here is different: whether that pooled effect is a
mixture whose parts behave differently, rather than one effect to be sliced
thinner. A knee and a hamstring cost different amounts of season and plausibly
different amounts of role on return.

Two cuts, both derivable from report_primary_injury:

location  the anatomical site as reported -- knee, ankle, hamstring, ...
type      the tissue, inferred from the site: muscle (hamstring, calf, quad,
          groin), joint (knee, ankle, shoulder, hip), tendon (achilles), head
          (concussion), bone/extremity (foot, toe, hand, ribs), illness

``type`` is inference, not data. The feed says "knee", never "ACL" against
"meniscus", so ligament-vs-cartilage is not recoverable here and the type cut is
coarser than the question deserves. It is worth measuring anyway because muscle
against joint against head is the split with a mechanism behind it.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import nflreadpy as nfl
pd.set_option("display.width", 250)

SEASONS = list(range(2016, 2026))
MUSCLE = ("hamstring", "calf", "quadricep", "quad", "groin", "thigh", "hip flexor")
JOINT = ("knee", "ankle", "shoulder", "hip", "wrist", "elbow", "neck", "back")
TENDON = ("achilles", "tendon")
HEAD = ("concussion", "head")
BONE = ("foot", "toe", "hand", "finger", "thumb", "rib", "chest", "abdomen", "collarbone")

def body_type(text: str) -> str:
    t = str(text).lower()
    if "not injury" in t or "personal" in t or "resting" in t: return "non_injury"
    if "illness" in t: return "illness"
    if any(k in t for k in HEAD): return "head"
    if any(k in t for k in TENDON): return "tendon"
    if any(k in t for k in MUSCLE): return "muscle"
    if any(k in t for k in JOINT): return "joint"
    if any(k in t for k in BONE): return "bone"
    return "other"

def location(text: str) -> str:
    t = str(text).lower()
    for k in ("knee","ankle","hamstring","shoulder","concussion","foot","hip",
              "groin","back","calf","toe","quadricep","achilles","ribs","neck"):
        if k in t: return "quad" if k == "quadricep" else k
    if "not injury" in t or "personal" in t or "resting" in t: return "non_injury"
    if "illness" in t: return "illness"
    return "other"

inj = nfl.load_injuries(seasons=SEASONS).to_pandas()
inj = inj[inj.position.isin(["QB","RB","WR","TE","FB","HB"])].copy()
inj["body"] = inj.report_primary_injury.fillna(inj.practice_primary_injury)
inj["site"] = inj.body.map(location)
inj["typ"] = inj.body.map(body_type)
inj = inj[~inj.site.isin(["non_injury"])]

# One row per player-season: the site reported in the most weeks that year, which
# is the season's dominant complaint rather than a single week's note.
sev = {"out": 3, "doubtful": 3, "questionable": 1}
inj["sev"] = inj.report_status.astype(str).str.lower().map(sev).fillna(0)
agg = (inj.groupby(["season","gsis_id","site","typ"])
          .agg(weeks=("week","nunique"), sev=("sev","sum")).reset_index())
agg = agg.sort_values(["season","gsis_id","sev","weeks"], ascending=[1,1,0,0])
season_injury = agg.drop_duplicates(["season","gsis_id"], keep="first")
print("player-seasons with a reported injury:", len(season_injury))
print()
print("=== cell counts by LOCATION ===")
print(season_injury.site.value_counts().to_string())
print()
print("=== cell counts by TYPE ===")
print(season_injury.typ.value_counts().to_string())
season_injury.to_pickle("/tmp/claude-0/-home-user-fantasy-football-research/4e1ed707-9a96-5493-b562-6226209c15ee/scratchpad/season_injury.pkl")
