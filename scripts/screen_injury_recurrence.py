"""Does injury recurrence cost role beyond playing time and age, and does
injury touch efficiency at all?

A binary reserve flag was rejected at the role layer because playing time
mediates it: told how many snaps a player took, the model gains nothing from
being told why. Recurrence is a different quantity and behaves differently.

Recurrence vs carry role, partial correlation as controls are added:

    none                       -0.162   p 9.4e-09
    snaps                      -0.156   p 3.3e-08
    age + experience           -0.147   p 1.9e-07
    snaps + age + experience   -0.141   p 5.7e-07

It barely moves. A player with four or more injury episodes in three years takes
less of his team's carries than his own prior role implies, and neither lost
playing time nor age explains it. Targets behave differently -- age and
experience absorb more than half (-0.115 to -0.051) -- so the durable finding is
carries.

Severity, by contrast, was mostly an artifact. Grouping by weeks missed showed a
dramatic gradient until prior availability was floored, at which point the worst
band flipped sign on targets: prior per-game role divides by prior availability,
so a tiny denominator inflates the baseline and manufactures exactly the shape a
real effect would have.

Efficiency is close to nothing. Against recurrence the sign is *positive* --
yards per carry +0.053, yards per target +0.037 -- which is survivorship rather
than support for the hypothesis, since a player who keeps getting hurt and keeps
playing is good enough to be kept. Against last season's absence receiving
efficiency is mildly negative (-0.056 yards per target, -0.048 catch rate) and
rushing is flat. Every one of these is under |r| = 0.06 against recurrence-role
at 0.14.

A screen, not a validation. The walk-forward arm is the test.


A player with four injury episodes in three years is likely older and declining,
and age and experience are already covariates in both layers. If the recurrence
gradient is age wearing an injury costume, the model already has it.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy import stats
pd.set_option("display.width", 250)

pr = pd.read_pickle("/home/user/fantasy-football-research/.cache/ffmodel-2026/player_rows.pkl")
d = pr[pr.season.lt(2026)].copy()
d = d[pd.to_numeric(d.get("is_replacement_player"), errors="coerce").fillna(0).ne(1)]
av_now = pd.to_numeric(d.snap_availability, errors="coerce")
av_pr  = pd.to_numeric(d.prior_availability, errors="coerce")
ok_base = av_pr.ge(0.60) & av_now.ge(0.15)

def role(share, prior):
    now = pd.to_numeric(d[share], errors="coerce") / av_now.clip(0.05, 1.0)
    old = pd.to_numeric(d[prior], errors="coerce") / av_pr.clip(0.05, 1.0)
    return np.log(now.where(now.gt(0.01) & old.gt(0.01) & ok_base) / old)

def eff(obs, prior, floor=0.01):
    o = pd.to_numeric(d[obs], errors="coerce"); p = pd.to_numeric(d[prior], errors="coerce")
    return np.log(o.where(o.gt(floor) & p.gt(floor)) / p)

d["rc"] = role("carry_share", "prior_carry_share")
d["rt"] = role("target_share", "prior_target_share")
d["rs"] = np.log((av_now / av_pr).where(ok_base))
d["e_ypc"]   = eff("rush_yards_per_carry", "prior_rush_yards_per_carry")
d["e_ypt"]   = eff("rec_yards_per_target", "prior_rec_yards_per_target")
d["e_catch"] = eff("rec_catch_rate", "prior_rec_catch_rate")
ep  = pd.to_numeric(d.prior_injury_episode_count_3yr, errors="coerce")
age = pd.to_numeric(d.age, errors="coerce")
exp = pd.to_numeric(d.experience, errors="coerce")

def partial(y, x, controls, frame):
    s = frame[[y, x] + controls].dropna()
    if len(s) < 60: return None
    Z = np.column_stack([np.ones(len(s))] + [s[c].to_numpy() for c in controls])
    def r(v):
        v = s[v].to_numpy()
        return v - Z @ np.linalg.lstsq(Z, v, rcond=None)[0]
    xr, yr = r(x), r(y)
    rr, pp = stats.pearsonr(xr, yr)
    return rr, pp, len(s)

frame = d.assign(ep=ep, age=age, exp=exp)
print("=== recurrence vs role, adding controls one at a time ===")
print(f"{'outcome':9}{'controls':34}{'partial r':>11}{'p':>11}{'n':>7}")
for y, lab in (("rc","carry"), ("rt","target")):
    for controls, cl in (
        ([], "none"),
        (["rs"], "snaps"),
        (["age","exp"], "age + experience"),
        (["rs","age","exp"], "snaps + age + experience"),
    ):
        out = partial(y, "ep", controls, frame)
        if out: print(f"{lab:9}{cl:34}{out[0]:>+11.3f}{out[1]:>11.2g}{out[2]:>7}")

print("\n=== EFFICIENCY: never tested before now ===")
print("residual log(observed / prior efficiency), by recurrence band")
frame["ep_band"] = pd.cut(ep, [-0.1,0.5,1.5,3.5,99], labels=["0","1","2-3","4+"])
print(f"{'episodes':10}{'yds/carry':>18}{'yds/target':>18}{'catch rate':>18}")
for band, blk in frame.groupby("ep_band", observed=True):
    cells = []
    for c in ("e_ypc","e_ypt","e_catch"):
        v = blk[c].dropna()
        cells.append(f"{v.mean():+.3f} (n={len(v)})" if len(v) >= 25 else "--")
    print(f"{str(band):10}" + "".join(f"{c:>18}" for c in cells))

print("\nefficiency vs recurrence, controlling for age and experience")
print(f"{'outcome':14}{'partial r':>11}{'p':>11}{'n':>7}")
for y, lab in (("e_ypc","yds/carry"), ("e_ypt","yds/target"), ("e_catch","catch rate")):
    out = partial(y, "ep", ["age","exp"], frame)
    if out: print(f"{lab:14}{out[0]:>+11.3f}{out[1]:>11.2g}{out[2]:>7}")

print("\nefficiency vs last season's weeks missed, controlling for age and experience")
frame["missed"] = ((1 - av_pr) * 17).round()
for y, lab in (("e_ypc","yds/carry"), ("e_ypt","yds/target"), ("e_catch","catch rate")):
    out = partial(y, "missed", ["age","exp"], frame)
    if out: print(f"{lab:14}{out[0]:>+11.3f}{out[1]:>11.2g}{out[2]:>7}")
