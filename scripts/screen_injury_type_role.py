"""Knee against hamstring, injured-to-injured, with the obvious confound removed.

The 'no injury reported' group is not a control: its mean availability is 0.412
against 0.65-0.79 for every injured group, so it is loaded with fringe players
who never reach an injury report at all. Comparing an injured group to it
measures roster status as much as injury. Injured-against-injured is the
comparison that survives.

Knee and hamstring are the sharp pair -- both common, both serious-sounding,
opposite return profiles in the raw cut. The confound to rule out is position
mix: if knees skew toward backs and hamstrings toward receivers, a position
effect wears the injury's clothes.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy import stats
pd.set_option("display.width", 250)

si = pd.read_pickle("/tmp/claude-0/-home-user-fantasy-football-research/4e1ed707-9a96-5493-b562-6226209c15ee/scratchpad/season_injury.pkl")
pr = pd.read_pickle("/home/user/fantasy-football-research/.cache/ffmodel-2026/player_rows.pkl")
pr = pr[pr.season.lt(2026)].copy()
pr = pr[pd.to_numeric(pr.get("is_replacement_player"), errors="coerce").fillna(0).ne(1)]
key = "player_id" if "player_id" in pr.columns else "gsis_id"
d = pr.merge(si.rename(columns={"gsis_id": key}), on=["season", key], how="left")
d["site"] = d.site.fillna("none"); d["typ"] = d.typ.fillna("none")
d = d.sort_values([key, "season"])
d["prev_site"] = d.groupby(key)["site"].shift(1)
d["prev_typ"] = d.groupby(key)["typ"].shift(1)
av = pd.to_numeric(d.snap_availability, errors="coerce")
av_pr = pd.to_numeric(d.prior_availability, errors="coerce")
ok = av_pr.ge(0.35) & av.ge(0.15)

def role(share, prior):
    now = pd.to_numeric(d[share], errors="coerce") / av.clip(0.05, 1.0)
    old = pd.to_numeric(d[prior], errors="coerce") / av_pr.clip(0.05, 1.0)
    return np.log(now.where(now.gt(0.01) & old.gt(0.01) & ok) / old)
d["r_carry"] = role("carry_share", "prior_carry_share")
d["r_tgt"]   = role("target_share", "prior_target_share")

print("=== position mix, is the contrast confounded? ===")
mix = pd.crosstab(d.prev_site, d.position, normalize="index")
print((mix.loc[[s for s in ["knee","ankle","hamstring","calf","foot"] if s in mix.index]]
       * 100).round(1).to_string())

print("\n=== injured vs injured, prior-season site ===")
print("residual role next season; * marks p<0.05 against the knee group")
for col, lab in (("r_carry","carries/gm"), ("r_tgt","targets/gm")):
    print(f"\n-- {lab} --")
    ref = d.loc[d.prev_site.eq("knee"), col].dropna()
    print(f"{'site':12}{'mean':>9}{'n':>6}{'vs knee':>10}{'p':>9}")
    for site in ["knee","ankle","hamstring","calf","groin","foot","shoulder","concussion"]:
        v = d.loc[d.prev_site.eq(site), col].dropna()
        if len(v) < 25: continue
        if site == "knee":
            print(f"{site:12}{v.mean():>+9.3f}{len(v):>6}{'--':>10}{'--':>9}")
        else:
            t_, p_ = stats.ttest_ind(v, ref, equal_var=False)
            print(f"{site:12}{v.mean():>+9.3f}{len(v):>6}{v.mean()-ref.mean():>+10.3f}{p_:>9.3f}")

print("\n=== controlling for position and age (partial correlation) ===")
print("muscle=1 vs joint=0, among players injured the previous season")
sub = d[d.prev_typ.isin(["muscle","joint"])].copy()
sub["is_muscle"] = sub.prev_typ.eq("muscle").astype(float)
sub["age_n"] = pd.to_numeric(sub.age, errors="coerce")
for pos_label, mask in (("all skill", sub.position.notna()),
                        ("RB only", sub.position.eq("RB")),
                        ("WR only", sub.position.eq("WR"))):
    for col, lab in (("r_carry","carries/gm"), ("r_tgt","targets/gm")):
        s = sub[mask][[col, "is_muscle", "age_n"]].dropna()
        if len(s) < 60: continue
        Z = np.column_stack([np.ones(len(s)), s.age_n.to_numpy()])
        r = lambda v: v - Z @ np.linalg.lstsq(Z, v, rcond=None)[0]
        rr, pp = stats.pearsonr(r(s.is_muscle.to_numpy()), r(s[col].to_numpy()))
        print(f"  {pos_label:10} {lab:12} r={rr:+.3f}  p={pp:.3g}  n={len(s)}")
