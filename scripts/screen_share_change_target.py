"""Would projecting the change in share help, and is the fit really diluted?

Two questions that came out of asking whether the share models should target a
change rather than a level, so that players with no role before their injury
stop washing the signal out.

**Scoring the change is the same number.** The prior share is known at predict
time, so it cancels: (pred - s0) - (act - s0) = pred - act. MAE on the change
equals MAE on the level to ten decimal places. The model is also already a
change model in structure -- eta is log(prior_role) plus exposure plus X beta,
so the prior is the anchor and the covariates are the deviation.

Scoring it as a *ratio* is not inert, but it pushes the wrong way: log(pred/act)
treats 1% to 2% the same as 20% to 40%, so it would weight no-role players more
heavily, not less.

**The fit is not diluted the way I described it.** The likelihood is a
Multinomial on counts, so a player's weight is his volume. I had said 68% of
reserve-flagged rows hold no role and therefore drag the coefficient; the first
half is true and the second does not follow.

What the counts actually say, 2016-2025 skill positions:

    rows with under 2% prior role   46% of rows, holding 20.8% of carries
    reserve AND role-holding        148 rows total, 15 a season, 1.5% of carries

So the informative population -- a player with a role to lose who is also
flagged -- carries about one and a half percent of the likelihood, at fifteen
rows a season. That is the binding constraint, and it is not an encoding
problem. It explains the four nulls at this layer without any of them needing to
be a mistake: a bare flag, the flag split by reserve kind, the flag interacted
with role, and recurrence all failed, and the interaction arm's -1.66% MAE on
two folds of three is what a real but under-powered effect looks like.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd

rng = np.random.default_rng(0)
s0 = rng.random(2000) * 0.3
act = np.clip(s0 + rng.normal(0, 0.05, 2000), 0, 1)
pred = np.clip(s0 + rng.normal(0, 0.05, 2000), 0, 1)
print("MAE on level :", f"{np.abs(pred - act).mean():.10f}")
print("MAE on change:", f"{np.abs((pred - s0) - (act - s0)).mean():.10f}")

pr = pd.read_pickle("/home/user/fantasy-football-research/.cache/ffmodel-2026/player_rows.pkl")
d = pr[pr.season.between(2016, 2025)]
d = d[pd.to_numeric(d.get("is_replacement_player"), errors="coerce").fillna(0).ne(1)]
prior = np.maximum(
    pd.to_numeric(d.prior_carry_share, errors="coerce").fillna(0),
    pd.to_numeric(d.prior_target_share, errors="coerce").fillna(0),
)
res = pd.to_numeric(d.roster_reserve, errors="coerce").fillna(0).gt(0)
for col, lab in (("rush_att", "carries"), ("targets", "targets")):
    c = pd.to_numeric(d[col], errors="coerce").fillna(0)
    norole, inf = prior.lt(0.02), res & prior.ge(0.08)
    print(f"\n{lab}: <2% prior role is {norole.mean():.0%} of rows, {c[norole].sum()/c.sum():.1%} of volume")
    print(f"   reserve and role-holding: {int(inf.sum())} rows, "
          f"{int(inf.sum())//10} a season, {c[inf].sum()/c.sum():.1%} of volume")
