"""Phase 3A diagnostics: cross-season volume projection.

Three checks, all offline on the legacy CSVs:
  1. Does vacated opportunity actually correlate with returning teammates'
     realized change in share? (Sanity for the key feature.)
  2. A walk-forward backtest: train on transitions <= year N, predict N+1
     season shares, and compare MAE against two baselines the model must beat:
       - persistence  : next = prior share
       - shrink       : next = mean of prior and position mean
  3. A breakout leaderboard for a chosen transition, to eyeball.

Run: python scripts/validate_crossseason.py
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from ffmodel.features import crossseason as cs
from ffmodel.models import volume_season as vs
from ffmodel.projections import season_volume as sv

SEASONS = [2014, 2015, 2016, 2017, 2018, 2019, 2020]
FIT_KW = dict(draws=600, tune=600, chains=2)


def vacated_signal(trans: pd.DataFrame) -> None:
    print("1) Opportunity signals vs returning RBs' realized Δ(carry share):")
    rb = trans[trans["position"] == "RB"].copy()
    rb["delta_carry"] = rb["next_carry_share"] - rb["carry_share"]
    sub = rb.dropna(subset=["vacated_carry_share", "delta_carry"])

    def rho(x):
        return sub[x].rank().corr(sub["delta_carry"].rank())

    # Net opportunity (vacated - incoming competition) should track realized
    # change better than vacated alone: competition is the other half.
    print(f"   RB: n={len(sub):,}")
    print(f"     Spearman(vacated_carry            -> Δ) = {rho('vacated_carry_share'):+.3f}")
    print(f"     Spearman(incoming_comp_carry      -> Δ) = {rho('incoming_comp_carry'):+.3f}")
    print(f"     Spearman(net_carry_opportunity    -> Δ) = {rho('net_carry_opportunity'):+.3f}\n")


def backtest(trans: pd.DataFrame) -> None:
    print("2) Walk-forward backtest — next-season target share (WR/TE/RB).")
    print("   Target share is very persistent, so matching persistence on MAE")
    print("   while producing calibrated 80% intervals is the win here.")
    print(f"   {'test':<12}{'model':>8}{'persist':>9}{'80%cov':>8}{'n':>7}")
    for test_tr in sorted(trans["transition"].unique())[2:]:
        train = trans[trans["transition"] < test_tr]
        test = trans[trans["transition"] == test_tr]
        if len(train) < 200 or len(test) < 20:
            continue
        model = vs.fit_target_share(train, **FIT_KW)
        q = model.predict_quantiles(test, qs=(0.1, 0.5, 0.9))
        y = test["next_target_share"].to_numpy()
        persist = test["target_share"].to_numpy()
        cov = float(np.mean((y >= q["p10"].to_numpy()) & (y <= q["p90"].to_numpy())))
        print(f"   {test_tr:<12}{_mae(y, q['p50'].to_numpy()):>8.4f}"
              f"{_mae(y, persist):>9.4f}{cov:>8.2f}{len(test):>7}")
    print()


def leaderboard(trans: pd.DataFrame, test_tr: str = "2019->2020") -> None:
    train = trans[trans["transition"] < test_tr]
    test = trans[trans["transition"] == test_tr]
    mt = vs.fit_target_share(train, **FIT_KW)
    mc = vs.fit_carry_share(train, **FIT_KW)
    rep = sv.breakout_report(test, mt, mc, threshold=0.05)
    print(f"3) Breakout leaderboard {test_tr} (top 10 by P(Δopp share > +0.05)):")
    cols = ["player_name", "position", "team_next", "prior_opp_share",
            "proj_opp_p50", "exp_delta", "p_breakout", "p_decline"]
    print(rep.head(10)[cols].to_string(index=False))


def _mae(y, yhat):
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(yhat))))


def main() -> None:
    trans = cs.build_transitions(SEASONS, source="legacy")
    print(f"Transitions: {len(trans):,} returning player-seasons "
          f"({trans['transition'].nunique()} year-pairs)\n")
    vacated_signal(trans)
    backtest(trans)
    leaderboard(trans)


if __name__ == "__main__":
    main()
