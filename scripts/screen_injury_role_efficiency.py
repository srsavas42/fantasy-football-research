"""Does a recent injury predict role and efficiency, net of prior role?

The share and efficiency layers read six covariates -- prior_availability,
prior_snap_share, age, experience, team_change, cold_start -- and no reserve or
injury flag. Injury status reaches the availability layer only. This screens
whether that is a real omission.

Which group answers that is the whole point. ``prior_availability`` is already a
covariate in both layers, so a gap measured on the "prior year under 0.6
availability" group is something the model can already see and may already have
learned. ``roster_reserve`` is not a covariate in either layer, so a gap on the
week-1 reserve group is invisible to them, and that is the group to read.

Result on 2015-2025: week-1 reserve players take 26% fewer carries and 17%
fewer targets per game than their own prior role implies, and gain 10% fewer
yards per target. Career injury history shows nothing -- the signal is recent
injury, not durability.

A screen, not a validation: these layers are fitted, so a raw residual gap does
not prove the fitted model misses it. The walk-forward arm is the test.

Role test with the prior put on the same per-game footing as the outcome.

prior_*_role is a season share, so for a player who missed half of last season
it is roughly half his per-game role. Dividing it by last season's availability
puts both sides of the residual in per-game terms; without that the test
guarantees a positive residual for anyone who was hurt.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy import stats
pd.set_option("display.width", 250)

pr = pd.read_pickle("/home/user/fantasy-football-research/.cache/ffmodel-2026/player_rows.pkl")
d = pr[pr.season.lt(2026)].copy()
d = d[pd.to_numeric(d.get("is_replacement_player"), errors="coerce").fillna(0).ne(1)]

av_now   = pd.to_numeric(d.snap_availability, errors="coerce").clip(0.05, 1.0)
av_prior = pd.to_numeric(d.prior_availability, errors="coerce").clip(0.05, 1.0)

def role_resid(share_col, prior_col):
    now   = pd.to_numeric(d[share_col],  errors="coerce") / av_now
    prior = pd.to_numeric(d[prior_col],  errors="coerce") / av_prior
    ok = now.gt(0.01) & prior.gt(0.01)
    return np.log(now.where(ok) / prior.where(ok))

def eff_resid(obs, prior, floor=0.01):
    o = pd.to_numeric(d[obs], errors="coerce"); p = pd.to_numeric(d[prior], errors="coerce")
    ok = o.gt(floor) & p.gt(floor)
    return np.log(o.where(ok) / p.where(ok))

d["r_carry"]  = role_resid("carry_share",  "prior_carry_share")
d["r_target"] = role_resid("target_share", "prior_target_share")
d["r_ypc"]   = eff_resid("rush_yards_per_carry", "prior_rush_yards_per_carry")
d["r_ypt"]   = eff_resid("rec_yards_per_target", "prior_rec_yards_per_target")
d["r_catch"] = eff_resid("rec_catch_rate",       "prior_rec_catch_rate")

healthy = ~d.roster_reserve.gt(0) & av_prior.ge(0.6)
groups = {
    "week-1 IR":        d.roster_injured_reserve.gt(0),
    "any week-1 res.":  d.roster_reserve.gt(0),
    "prior yr <0.6 av": av_prior.lt(0.6),
    "injury hist >4wk": pd.to_numeric(d.get("prior_injury_out_weeks_3yr"), errors="coerce").gt(4),
}
metrics = [("r_carry","role carries/gm"),("r_target","role targets/gm"),
           ("r_ypc","eff yds/carry"),("r_ypt","eff yds/target"),("r_catch","eff catch rate")]

print("Per-game role residual, both sides availability-normalised.")
print(f"ref = healthy (n={int(healthy.sum())})\n")
print(f"{'group':20}{'metric':20}{'delta':>9}{'t':>7}{'n':>6}")
for label, mask in groups.items():
    for col, lab in metrics:
        a = d.loc[mask, col].dropna(); b = d.loc[healthy, col].dropna()
        if len(a) < 15: continue
        t, p = stats.ttest_ind(a, b, equal_var=False)
        print(f"{label:20}{lab:20}{a.mean()-b.mean():>+9.3f}{t:>7.2f}{len(a):>6}{' *' if p<0.05 else ''}")
