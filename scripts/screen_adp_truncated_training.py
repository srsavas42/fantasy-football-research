"""Should the pipeline train only on the players who matter for a draft?

Evaluating on the drafted population is right and is already done -- the full
pipeline reports ppr_drafted alongside ppr_overall. Training on it is a
different proposition, and three things argue against it.

**It breaks the volume layer structurally.** Role shares are a softmax
allocation across a team's roster, and the roster is what makes them sum to one.
Restricting to drafted players leaves 7.4 of 18.8 players per team-season, and
their carry shares sum to 0.855 on average (sd 0.122, as low as 0.324). The
model would be asked to allocate a total that is neither one nor constant across
teams, which is not the same model fitted on less data -- it is a different
target.

**It selects on a forecast of the response.** ADP is a prediction of the thing
being predicted, so conditioning the training set on it truncates asymmetrically:
131 player-seasons finished top-100 without being drafted and would be discarded,
while 402 drafted players finished outside the top 300 and would be kept. The
rows lost are the low-prior/high-outcome corner -- exactly the examples that
teach the model when a role *increases*, which the reliability work identified as
the remaining volume error. The rows kept are the high-prior/low-outcome corner,
which teaches the opposite lesson.

**It discards 61% of the data.** Drafted rows are 2,575 of 6,575.

**But the premise behind the question is real.** The relationships genuinely
differ by tier:

    response                 tier        n     slope     r2
    rec_yards_per_target     drafted   1353    +0.669   25.9%
    rec_yards_per_target     undrafted  473    +0.454    6.9%
    rec_catch_rate           drafted   1353    +0.854   44.6%
    rec_catch_rate           undrafted  473    +0.776   23.7%
    rush_yards_per_carry     drafted    776    +0.589   10.9%
    rush_yards_per_carry     undrafted  154    +0.552    7.3%

Part of that gap is restricted range rather than real heterogeneity -- undrafted
players have compressed priors, which deflates r-squared on its own -- but the
slopes differ too, and a slope is not a range artifact.

The remedy for heterogeneity is an interaction or a weight, both of which keep
the rows while letting the coefficients differ. Truncation is the one option
that throws the data away to get the same flexibility. Note also that the
efficiency ridge already weights by exposure, clipped to [0.25, 5.0], which is a
soft version of caring more about the players who matter.

    python scripts/screen_adp_truncated_training.py
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy import stats

CACHE = "/home/user/fantasy-football-research/.cache/ffmodel-2026"


def main() -> int:
    pr = pd.read_pickle(f"{CACHE}/player_rows.pkl")
    p = pr[pr.season.between(2015, 2025)].copy()
    p = p[pd.to_numeric(p.get("is_replacement_player"), errors="coerce").fillna(0).ne(1)]
    drafted = pd.to_numeric(p.get("adp_drafted"), errors="coerce").fillna(0).eq(1)

    def num(name):
        return (
            pd.to_numeric(p[name], errors="coerce").fillna(0)
            if name in p.columns
            else pd.Series(0.0, index=p.index)
        )

    print("=== what training on the drafted set would discard ===")
    print(f"  drafted rows {int(drafted.sum())} of {len(p)} ({drafted.mean():.1%})")
    print(f"  players per team-season: all {p.groupby(['season','team']).size().mean():.1f}"
          f", drafted only {p[drafted].groupby(['season','team']).size().mean():.1f}")

    print("\n=== the structural problem: shares are a softmax over the roster ===")
    p["team_carries"] = p.groupby(["season", "team"]).rush_att.transform("sum")
    p["carry_share"] = p.rush_att / p["team_carries"].replace(0, np.nan)
    kept = p[drafted].groupby(["season", "team"]).carry_share.sum()
    print(f"  full roster sums to 1.000 by construction")
    print(f"  drafted subset sums to {kept.mean():.3f} "
          f"(sd {kept.std():.3f}, min {kept.min():.3f})")

    print("\n=== the selection problem: ADP forecasts the response ===")
    p["ppr"] = (
        num("rec_yds") * 0.1 + num("rush_yds") * 0.1 + num("receptions")
        + num("eff_rec_td") * 6 + num("eff_rush_td") * 6
    )
    rank = p.groupby("season").ppr.rank(ascending=False)
    print(f"  top-100 finishers never drafted (lost):        {int(((~drafted) & rank.le(100)).sum())}")
    print(f"  drafted, finished outside top-300 (kept):      {int((drafted & rank.gt(300)).sum())}")

    print("\n=== but the premise is real: the relationships differ by tier ===")
    for response, prior, exposure, floor in (
        ("rush_yards_per_carry", "prior_rush_yards_per_carry", "rush_att", 30),
        ("rec_yards_per_target", "prior_rec_yards_per_target", "targets", 30),
        ("rec_catch_rate", "prior_rec_catch_rate", "targets", 30),
    ):
        y = pd.to_numeric(p[response], errors="coerce")
        q = pd.to_numeric(p[prior], errors="coerce")
        e = pd.to_numeric(p[exposure], errors="coerce").fillna(0)
        ok = y.notna() & q.notna() & e.ge(floor)
        for label, mask in (("drafted", ok & drafted), ("undrafted", ok & ~drafted)):
            if mask.sum() < 80:
                print(f"  {response:22} {label:10} n={int(mask.sum()):5d}  too few")
                continue
            slope, _ = np.polyfit(q[mask], y[mask], 1)
            r = stats.pearsonr(q[mask], y[mask])[0]
            print(f"  {response:22} {label:10} n={int(mask.sum()):5d}  "
                  f"slope {slope:+.3f}  r2 {r * r:5.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
