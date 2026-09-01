"""Why the reserve arm was null: the flag fires mostly where nothing can move.

roster_reserve is 1 for a fringe roster body and for a starter returning from
injury alike, and 68% of flagged rows held under 2% of their team's work the
year before. A single fitted coefficient is therefore estimated mostly on
players whose volume cannot move, and then applied to the few whose can.

The flag does not even point the same way across those groups. Role residual for
reserve players against healthy ones, within bands of the role they held the
season before:

    prior role        carry (reserve vs healthy)   target
    fringe 2-8%       +0.463 vs +0.401             +0.059 vs +0.196
    rotation 8-20%    -0.346 vs +0.047             -0.244 vs -0.077
    starter 20%+      -0.394 vs -0.161             -0.162 vs -0.061

Among fringe players being on a reserve list costs nothing. Among players with a
real role it costs 0.2 to 0.4 in log share. Averaging those into one coefficient
is what the rejected arm did, and it explains the null without needing the
effect to be absent.

It also explains why recurrence survived where the flag did not: episode count
rises with role rather than falling with it -- mean prior role runs 0.030, 0.093,
0.118, 0.163 across the bands -- because a player has to play to get hurt, and
has to be worth keeping to stay in the league while hurt. The recurrence signal
is concentrated in exactly the population the binary flag dilutes.

The population that matters is small and that bounds what any rerun can show:
roughly 30 to 50 rows a season hold a real role and a reserve flag at once.


roster_reserve fires for a fringe roster body and for a starter returning from
injury alike. If most flagged rows are the former, one fitted coefficient is
mostly describing marginal players, and it cannot say anything useful about the
few whose volume could actually move.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy import stats
pd.set_option("display.width", 240)

pr = pd.read_pickle("/home/user/fantasy-football-research/.cache/ffmodel-2026/player_rows.pkl")
d = pr[pr.season.lt(2026)].copy()
d = d[pd.to_numeric(d.get("is_replacement_player"), errors="coerce").fillna(0).ne(1)]
res = pd.to_numeric(d.roster_reserve, errors="coerce").fillna(0).gt(0)
prior_carry  = pd.to_numeric(d.prior_carry_share, errors="coerce").fillna(0)
prior_target = pd.to_numeric(d.prior_target_share, errors="coerce").fillna(0)
prior_any = np.maximum(prior_carry, prior_target)

print(f"roster_reserve = 1 on {int(res.sum())} rows ({res.mean():.1%} of the frame)")
print("\ncomposition of those rows by the role they held the year before:")
band = pd.cut(prior_any, [-0.001, 0.02, 0.08, 0.20, 1.0],
              labels=["none (<2%)", "fringe 2-8%", "rotation 8-20%", "starter 20%+"])
comp = pd.crosstab(band[res], columns="n")
comp["share of flagged"] = comp["n"] / comp["n"].sum()
print(comp.to_string())

print("\n=== does the flag mean the same thing in each band? ===")
av_now = pd.to_numeric(d.snap_availability, errors="coerce")
av_pr  = pd.to_numeric(d.prior_availability, errors="coerce")
ok = av_pr.ge(0.60) & av_now.ge(0.15)
def role(share, prior):
    now = pd.to_numeric(d[share], errors="coerce") / av_now.clip(0.05, 1.0)
    old = pd.to_numeric(d[prior], errors="coerce") / av_pr.clip(0.05, 1.0)
    return np.log(now.where(now.gt(0.01) & old.gt(0.01) & ok) / old)
d["rc"] = role("carry_share", "prior_carry_share")
d["rt"] = role("target_share", "prior_target_share")
d["band"] = band

print("mean role residual, reserve vs healthy, within each prior-role band")
print(f"{'prior role':18}{'carry: res':>14}{'carry: healthy':>16}{'target: res':>14}{'target: healthy':>17}")
for b in band.cat.categories:
    m = d.band.eq(b)
    row = [b]
    for col in ("rc", "rt"):
        a = d.loc[m & res, col].dropna(); h = d.loc[m & ~res, col].dropna()
        row.append(f"{a.mean():+.3f} (n={len(a)})" if len(a) >= 12 else "--")
        row.append(f"{h.mean():+.3f}" if len(h) >= 12 else "--")
    print(f"{row[0]:18}{row[1]:>14}{row[2]:>16}{row[3]:>14}{row[4]:>17}")

print("\n=== how much of the FLAGGED population can move at all? ===")
for col, share_col, lab in (("rc","carry_share","carry"), ("rt","target_share","target")):
    flagged = d.loc[res, col].dropna()
    movable = d.loc[res & prior_any.gt(0.08), col].dropna()
    print(f"  {lab}: {len(flagged)} flagged rows with a residual, "
          f"{len(movable)} ({len(movable)/max(len(flagged),1):.0%}) held a real role beforehand")

print("\n=== same question for the recurrence signal ===")
ep = pd.to_numeric(d.prior_injury_episode_count_3yr, errors="coerce")
print("episode count is nonzero on", f"{ep.gt(0).mean():.1%}", "of rows")
print("mean prior role by episode band (is recurrence concentrated in bench players?):")
epb = pd.cut(ep, [-0.1,0.5,1.5,3.5,99], labels=["0","1","2-3","4+"])
print(pd.DataFrame({"mean prior role": prior_any.groupby(epb, observed=True).mean().round(4),
                    "n": prior_any.groupby(epb, observed=True).size()}).to_string())
