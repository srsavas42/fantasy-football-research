"""Two questions the SOS screen raised: is its sign real, and is O-line better?

The rushing SOS correlation came out significant and *backwards* -- a schedule
of opponents who allowed more yards last year went with a lower residual, when
an easier schedule should raise it. The obvious confound is the player's own
prior. Divisions are played twice a year, so schedule strength is autocorrelated
across seasons; a soft slate last year inflates the prior this year's residual is
measured against, and pure mean reversion then reads as a matchup effect with
the wrong sign. Controlling for last season's SOS separates the two.

O-line is screened on the same terms. It has a better prior than SOS for one
structural reason: the efficiency layer carries *no* team-level covariate at all
-- prior_availability, prior_snap_share, age, experience, team_change,
cold_start and nothing else -- so blocking quality is not merely under-weighted,
it is absent. And unlike a schedule, it does not average out over seventeen
games.

Proxies, both leakage-safe from the prior season:
  pass protection   team sack rate (already computed in team_rows)
  run blocking      team rush yards over expected per attempt (NGS)
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy import stats
import nflreadpy as nfl
pd.set_option("display.width", 220)

SEASONS = list(range(2016, 2026))
pr = pd.read_pickle("/home/user/fantasy-football-research/.cache/ffmodel-2026/player_rows.pkl")
tr = pd.read_pickle("/home/user/fantasy-football-research/.cache/ffmodel-2026/team_rows.pkl")
p = pr[pr.season.isin(SEASONS)].copy()
p = p[pd.to_numeric(p.get("is_replacement_player"), errors="coerce").fillna(0).ne(1)]

# ---- run blocking: team rush yards over expected per attempt, prior season ----
ngs = nfl.load_nextgen_stats(seasons=SEASONS, stat_type="rushing").to_pandas()
ngs = ngs[ngs.week.eq(0)] if (ngs.week == 0).any() else ngs
team_ryoe = ngs.groupby(["season", "team_abbr"], dropna=False).agg(
    ryoe=("rush_yards_over_expected_per_att", "mean")).reset_index()
team_ryoe = team_ryoe.rename(columns={"team_abbr": "team"})
team_ryoe["season"] = team_ryoe.season + 1          # prior season -> this season

# ---- pass protection: team sack rate, prior season (already in team_rows) ----
sack = tr[["season", "team", "prior_sack_rate"]].copy()

m = p.merge(team_ryoe, on=["season", "team"], how="left")
m = m.merge(sack, on=["season", "team"], how="left")

print("=== dispersion: does the O-line proxy vary more than a schedule does? ===")
print("   (SOS was cv 2.4% on yds/target allowed, 3.8% on yds/carry allowed)")
for col, lab in (("ryoe", "team rush yds over exp/att"), ("prior_sack_rate", "team sack rate")):
    v = pd.to_numeric(m[col], errors="coerce").dropna()
    if len(v) == 0:
        print(f"  {lab:30} no coverage"); continue
    print(f"  {lab:30} mean {v.mean():+.3f}  sd {v.std():.3f}  "
          f"range {v.min():+.2f}..{v.max():+.2f}  n={len(v)}")

print("\n=== does the O-line proxy predict the efficiency residual? ===")
for ycol, pcol, xcol, expo, floor, lab in (
    ("rush_yards_per_carry", "prior_rush_yards_per_carry", "ryoe", "rush_att", 25, "yds/carry ~ run block"),
    ("rec_yards_per_target", "prior_rec_yards_per_target", "prior_sack_rate", "targets", 25, "yds/target ~ sack rate"),
    ("pass_yards_per_attempt", "prior_pass_yards_per_attempt", "prior_sack_rate", "pass_att", 50, "yds/att ~ sack rate"),
):
    o = pd.to_numeric(m[ycol], errors="coerce"); q = pd.to_numeric(m[pcol], errors="coerce")
    e = pd.to_numeric(m[expo], errors="coerce").fillna(0); x = pd.to_numeric(m[xcol], errors="coerce")
    ok = o.gt(0.1) & q.gt(0.1) & e.ge(floor) & x.notna()
    if ok.sum() < 50:
        print(f"  {lab:26} too few rows ({ok.sum()})"); continue
    resid = np.log(o[ok] / q[ok])
    r, pv = stats.pearsonr(x[ok], resid)
    print(f"  {lab:26} r={r:+.3f}  p={pv:.3g}  n={ok.sum():>5}   "
          f"1sd moves residual {r*resid.std()*100:+.1f}%")
