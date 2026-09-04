"""Player-level test: does a receiver's aDOT move toward his new play-caller's?

Play-callers plainly differ in depth of target -- some want short throws and
run-after, some want shots downfield -- so the natural question is whether that
preference travels with the coach the way running-back usage does
(screen_coaching_tree_transfer.py finds rb_target_share transfers at +0.204).

The team-season version of this test is in that same screen and comes back null
on every depth shape. But n=121 there resolves only |r| > 0.18, so a real but
modest effect would be invisible. This is the well-powered version, and it asks
the question in the form the hypothesis is actually about.

The design holds fixed everything except the play-caller: the same receiver, on
the same team, in back-to-back seasons with a real target load in both, whose
scheme carrier changed between them. Controls are his own prior aDOT (mean
reversion) and the team's prior aDOT (whatever the offense was already doing).
The only thing left moving is who is calling the plays.

Result: nothing. partial r = -0.005 (p=0.95) against the incoming coach's
carried aDOT, +0.014 (p=0.86) against his carried YAC share. Those are flat
zeros, not merely non-significant, and the team-level test agrees.

The reading that fits both: a play-caller decides *who* gets the ball, which is
why running-back share travels with him, but not *how far it goes* -- that
needs a quarterback who will throw it there and a receiver who can get there,
and neither arrives in his luggage. Miami's short-and-YAC offense is not
separable from having signed Hill and Waddle.

Caveat worth keeping: this measures *realized* aDOT, which mixes play design
with execution and personnel. A designed-depth measure from route tracking
might transfer where realized depth does not; this repo has no such column.

    python scripts/screen_coach_adot_player_level.py
"""
import sys; sys.path.insert(0,'src'); sys.path.insert(0,'scripts')
import numpy as np, pandas as pd
from scipy import stats
from screen_coaching_tree_transfer import team_shapes
from ffmodel.data.coaching import load_scheme_lineage, load_scheme_sources

rows = pd.read_pickle(".cache/ffmodel-walkforward/player_rows.pkl")
rows = rows[pd.to_numeric(rows.get("is_replacement_player"),errors="coerce").fillna(0).ne(1)]
shapes = team_shapes(rows)

# Carried aDOT / yac_share of the scheme coach, from external prior stops.
lin = load_scheme_lineage()
role = lin.prior_role.astype(str).str.lower()
lin = lin[role.str.contains("offensive coordinator|head coach|quarterbacks coach",na=False)
          & ~role.str.contains("assistant|quality control|intern",na=False)]
lin = lin[lin.prior_team_code != lin.franchise_code]
st = lin.merge(shapes.rename(columns={"season":"prior_season","team":"prior_team_code"}),
               on=["prior_season","prior_team_code"], how="inner")
carried = {}
for m in ("team_adot","yac_share"):
    v = pd.to_numeric(st[m],errors="coerce"); ok=v.notna()
    g = st[ok].assign(_v=v[ok]).groupby(["season","franchise_code"],as_index=False)._v.mean()
    carried[m] = g.rename(columns={"_v":f"coach_{m}","franchise_code":"team"})

# Receivers with a real season in back-to-back years on the SAME team, so the
# player and roster are held roughly fixed and the play-caller is what moved.
w = rows[rows.position.isin(["WR","TE"])].copy()
w["_adot"] = pd.to_numeric(w.eff_rec_air_yds,errors="coerce")/w.targets.where(w.targets>0)
w = w[w.targets.ge(30) & w._adot.notna()][["season","team","player_key","_adot","targets"]]
prev = w.copy(); prev["season"] = prev.season+1
panel = w.merge(prev.rename(columns={"_adot":"_adot_prev","targets":"_targets_prev"}),
                on=["season","team","player_key"], how="inner")

src = load_scheme_sources().rename(columns={"franchise_code":"team"})
src = src.sort_values(["team","season"])
src["_prev_coach"] = src.groupby("team").scheme_coach_page_title.shift(1)
src["new_coach"] = (src.scheme_coach_page_title.notna() & src._prev_coach.notna()
                    & src.scheme_coach_page_title.ne(src._prev_coach))
panel = panel.merge(src[["season","team","new_coach"]], on=["season","team"], how="left")
for m in ("team_adot","yac_share"):
    panel = panel.merge(carried[m], on=["season","team"], how="left")

print(f"{len(panel)} same-team receiver-season pairs (>=30 targets both years)")
print(f"  of which the play-caller changed: {int(panel.new_coach.fillna(False).sum())}\n")

# Does the change in a receiver's aDOT follow the incoming coach's carried aDOT,
# controlling for his own prior aDOT (mean reversion) and the team's prior aDOT?
tp = shapes.rename(columns={"team":"team"})[["season","team","team_adot"]].copy()
tp["season"] = tp.season+1
panel = panel.merge(tp.rename(columns={"team_adot":"team_adot_prev"}), on=["season","team"], how="left")

sub = panel[panel.new_coach.fillna(False)].dropna(subset=["coach_team_adot","team_adot_prev"])
print(f"=== receivers whose play-caller changed, with a carried coach aDOT: n={len(sub)} ===")
y = sub._adot.to_numpy(float)
D = np.column_stack([sub._adot_prev.to_numpy(float), sub.team_adot_prev.to_numpy(float), np.ones(len(sub))])
for name in ("coach_team_adot","coach_yac_share"):
    x = pd.to_numeric(sub[name],errors="coerce").to_numpy(float)
    at = np.isfinite(x)
    if at.sum() < 40: print(f"  {name}: too few ({at.sum()})"); continue
    Dm=D[at]
    xr = x[at]-Dm@np.linalg.lstsq(Dm,x[at],rcond=None)[0]
    yr = y[at]-Dm@np.linalg.lstsq(Dm,y[at],rcond=None)[0]
    r = float(np.corrcoef(xr,yr)[0,1]); n=int(at.sum())
    p = 2*stats.norm.sf(abs(np.arctanh(r)*np.sqrt(max(n-3-Dm.shape[1],1))))
    print(f"  {name:<20} n={n:<5} partial r={r:+.4f}  p={p:.3f}   (controls: own prior aDOT, team prior aDOT)")
