"""Is there enough season-long opponent-defence signal to be worth modelling?

Two things have to be true before this is worth building, and the second is the
one that usually fails for a season-long projection.

**It has to be knowable in August.** The schedule is, so opponent identity is
fine; opponent *quality* cannot come from the season being projected. This uses
each opponent's prior-season defence, which is what a drafter actually has.

**It has to vary.** Over seventeen games a schedule is close to balanced, and
the week-to-week swing that makes matchups matter for a weekly projection
averages out. If the spread in season-long opponent strength is small next to
the spread in the thing being predicted, there is no room for the feature to
work however real the underlying effect is.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy import stats
import nflreadpy as nfl
pd.set_option("display.width", 220)

SEASONS = list(range(2015, 2026))
wk = nfl.load_player_stats(seasons=SEASONS).to_pandas()
wk = wk[wk.season_type.eq("REG")] if "season_type" in wk.columns else wk

# Defence allowed, by the team that was defending, in that season.
d = wk.groupby(["season", "opponent_team"], dropna=False).agg(
    rec_yds=("receiving_yards", "sum"), tgts=("targets", "sum"),
    rush_yds=("rushing_yards", "sum"), carries=("carries", "sum"),
).reset_index().rename(columns={"opponent_team": "team"})
d["def_ypt_allowed"] = d.rec_yds / d.tgts.replace(0, np.nan)
d["def_ypc_allowed"] = d.rush_yds / d.carries.replace(0, np.nan)
d = d[["season", "team", "def_ypt_allowed", "def_ypc_allowed"]]

# Schedule: who each team plays. Opponent quality is taken from season-1.
sched = nfl.load_schedules(seasons=SEASONS).to_pandas()
sched = sched[sched.game_type.eq("REG")]
long = pd.concat([
    sched[["season", "home_team", "away_team"]].rename(
        columns={"home_team": "team", "away_team": "opp"}),
    sched[["season", "away_team", "home_team"]].rename(
        columns={"away_team": "team", "home_team": "opp"}),
], ignore_index=True)
prior_def = d.copy(); prior_def["season"] = prior_def.season + 1
long = long.merge(
    prior_def.rename(columns={"team": "opp"}), on=["season", "opp"], how="left")
sos = long.groupby(["season", "team"], dropna=False).agg(
    sos_ypt=("def_ypt_allowed", "mean"), sos_ypc=("def_ypc_allowed", "mean"),
    games=("opp", "size"),
).reset_index()

print("=== how much does season-long opponent strength actually vary? ===")
for col, lab in (("sos_ypt", "opp yds/target allowed"), ("sos_ypc", "opp yds/carry allowed")):
    v = sos[col].dropna()
    print(f"  {lab:26} mean {v.mean():.3f}  sd {v.std():.3f}  "
          f"cv {v.std()/v.mean():.1%}  range {v.min():.2f}-{v.max():.2f}")

pr = pd.read_pickle("/home/user/fantasy-football-research/.cache/ffmodel-2026/player_rows.pkl")
p = pr[pr.season.isin(SEASONS)].copy()
p = p[pd.to_numeric(p.get("is_replacement_player"), errors="coerce").fillna(0).ne(1)]
print("\n=== for comparison, the spread in what we are predicting ===")
for col, lab in (("rec_yards_per_target", "player yds/target"), ("rush_yards_per_carry", "player yds/carry")):
    v = pd.to_numeric(p[col], errors="coerce")
    v = v[v.gt(0)].dropna()
    print(f"  {lab:26} mean {v.mean():.3f}  sd {v.std():.3f}  cv {v.std()/v.mean():.1%}")

m = p.merge(sos, on=["season", "team"], how="left")
print("\n=== does opponent strength predict the efficiency residual? ===")
print("   residual = log(observed / prior); leakage-safe SOS from prior-season defence")
for ycol, pcol, scol, expo, floor, lab in (
    ("rec_yards_per_target", "prior_rec_yards_per_target", "sos_ypt", "targets", 25, "yds/target"),
    ("rush_yards_per_carry", "prior_rush_yards_per_carry", "sos_ypc", "rush_att", 25, "yds/carry"),
):
    o = pd.to_numeric(m[ycol], errors="coerce")
    q = pd.to_numeric(m[pcol], errors="coerce")
    e = pd.to_numeric(m[expo], errors="coerce").fillna(0)
    s = pd.to_numeric(m[scol], errors="coerce")
    ok = o.gt(0.1) & q.gt(0.1) & e.ge(floor) & s.notna()
    resid = np.log(o[ok] / q[ok]); x = s[ok]
    r, pv = stats.pearsonr(x, resid)
    # what an r that size buys, in the units the response is measured in
    implied = r * resid.std() * 100
    print(f"  {lab:12} r={r:+.3f}  p={pv:.3g}  n={ok.sum():>5}   "
          f"1sd of SOS moves the residual {implied:+.1f}%")
