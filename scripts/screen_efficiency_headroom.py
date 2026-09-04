"""Which efficiency responses have room left, and what actually reaches it.

This is the diagnostic that should have run before any of the covariate
screening in this line of work, and would have redirected most of it.

Two measurements. Split-half reliability -- each response computed on a player's
odd weeks and his even weeks, correlated across players within a season and
Spearman-Brown corrected back to full length -- gives the fraction of an observed
season rate that is signal rather than sampling noise. With that in hand the
year-over-year correlation can be disattenuated, and the most any model can
explain of next season's *observed* rate is r_yoy^2 / reliability.

    response                reliability   r_yoy   ceiling   design   headroom
    rec_yards_per_target          42.7%   0.398    37.1%    27.1%     +10.0%
    rush_td_rate                  37.6%   0.265    18.6%     9.3%      +9.3%
    rec_catch_rate                62.9%   0.565    50.7%    42.6%      +8.1%
    pass_completion_rate          60.3%   0.401    26.6%    19.9%      +6.7%
    rec_td_rate                   26.7%   0.176    11.6%     6.7%      +4.9%
    pass_td_rate                  45.2%   0.333    24.5%    19.8%      +4.7%
    rush_yards_per_carry          52.9%   0.355    23.8%    22.2%      +1.6%
    pass_yards_per_attempt        56.6%   0.383    25.9%    24.8%      +1.1%

``rush_yards_per_carry`` is where nearly all of this session's covariate work
went -- opponent defence, three O-line proxies, team RYOE, team pressure rate,
quarterback carry share -- and it had 1.6% of variance left to find. Seven
failures on the most nearly-exhausted response in the layer is not seven
independent facts about football.

The ceiling is for a predictor built from a *player's history*, not from one
season, which is the second measurement here: an exposure-weighted prior over a
trailing window against the one-year prior every response currently uses.

    response                  1yr     2yr     3yr  career   best gain
    rec_yards_per_target    15.4%   20.2%   23.7%   19.7%      +8.2%
    rush_yards_per_carry    12.0%   16.0%   17.2%   20.0%      +8.0%
    rec_catch_rate          34.5%   39.1%   39.6%   39.9%      +5.5%
    pass_yards_per_attempt  10.5%   12.2%   11.6%   13.8%      +3.3%
    rush_td_rate             6.7%    9.1%    8.2%    5.4%      +2.4%
    rec_td_rate              3.5%    4.1%    5.6%    5.2%      +2.2%

Every response improves, several by more than any covariate screened in this
session. It is also the explanation for the covariate failures rather than a
separate finding: the disattenuated year-over-year correlation of true talent is
about 0.93 on yards per target, so talent barely moves and almost all of the
one-year prior's error is measurement noise in the prior itself. A covariate
cannot fix noise in the input it is being added alongside; more history can.

Note the shape differs by response. Receiving and rushing yards want three years
or a career; the touchdown rates peak at two and get worse beyond it, which is
what a genuinely drifting quantity looks like and matches their low reliability.
So the window is a per-response choice, not a constant.

    python scripts/screen_efficiency_headroom.py
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy import stats
import nflreadpy as nfl

SEASONS = list(range(2012, 2026))

# response -> (numerator, denominator, minimum exposure per half)
RESPONSES = {
    "rec_catch_rate":         ("receptions", "targets", 15),
    "rec_yards_per_target":   ("receiving_yards", "targets", 15),
    "rec_td_rate":            ("receiving_tds", "targets", 15),
    "rush_yards_per_carry":   ("rushing_yards", "carries", 15),
    "rush_td_rate":           ("rushing_tds", "carries", 15),
    "pass_yards_per_attempt": ("passing_yards", "attempts", 40),
    "pass_completion_rate":   ("completions", "attempts", 40),
    "pass_td_rate":           ("passing_tds", "attempts", 40),
}

# What the shipping design explains, measured by the covariate screens.
DESIGN = {
    "rec_catch_rate": 0.426, "rec_yards_per_target": 0.271, "rec_td_rate": 0.067,
    "rush_yards_per_carry": 0.222, "rush_td_rate": 0.093,
    "pass_yards_per_attempt": 0.248, "pass_completion_rate": 0.199,
    "pass_td_rate": 0.198,
}


def _weekly() -> pd.DataFrame:
    d = nfl.load_player_stats(seasons=SEASONS).to_pandas()
    if "season_type" in d:
        d = d[d.season_type.eq("REG")]
    d["week"] = pd.to_numeric(d.week, errors="coerce")
    return d[d.week.notna()]


def reliability_table(d: pd.DataFrame) -> None:
    d = d.copy()
    d["half"] = np.where(d.week.astype(int) % 2 == 1, "odd", "even")
    print(f"{'response':24} {'reliability':>11} {'r_yoy':>7} {'ceiling':>9} "
          f"{'design':>7} {'headroom':>9}")
    print("-" * 72)
    for name, (num, den, floor) in RESPONSES.items():
        for column in (num, den):
            d[column] = pd.to_numeric(d.get(column), errors="coerce").fillna(0.0)
        halves = d.groupby(["season", "player_id", "half"], as_index=False)[[num, den]].sum()
        wide = halves.pivot_table(
            index=["season", "player_id"], columns="half", values=[num, den]
        )
        wide.columns = [f"{a}_{b}" for a, b in wide.columns]
        wide = wide[wide[f"{den}_odd"].ge(floor) & wide[f"{den}_even"].ge(floor)]
        if len(wide) < 200:
            print(f"{name:24} too few players ({len(wide)})")
            continue
        half_r, _ = stats.pearsonr(
            wide[f"{num}_odd"] / wide[f"{den}_odd"],
            wide[f"{num}_even"] / wide[f"{den}_even"],
        )
        # Spearman-Brown: two half-length measurements to one full-length one.
        rel = 2 * half_r / (1 + half_r)

        full = halves.groupby(["season", "player_id"], as_index=False)[[num, den]].sum()
        full = full[full[den].ge(2 * floor)]
        full["rate"] = full[num] / full[den]
        nxt = full.copy()
        nxt["season"] = nxt.season - 1
        pair = full.merge(nxt, on=["season", "player_id"], suffixes=("", "_next"))
        r_yoy, _ = stats.pearsonr(pair.rate, pair.rate_next)
        ceiling = r_yoy**2 / rel if rel > 0 else np.nan
        design = DESIGN.get(name, np.nan)
        print(f"{name:24} {rel:11.1%} {r_yoy:+7.3f} {ceiling:9.1%} "
              f"{design:7.1%} {ceiling - design:+9.1%}")


def window_table(d: pd.DataFrame) -> None:
    print(f"\n{'response':24} {'1yr':>8} {'2yr':>8} {'3yr':>8} {'career':>8} {'gain':>8}")
    print("-" * 70)
    for name, (num, den, floor) in RESPONSES.items():
        floor = floor * 2
        for column in (num, den):
            d[column] = pd.to_numeric(d.get(column), errors="coerce").fillna(0.0)
        seasons = (
            d.groupby(["season", "player_id"], as_index=False)[[num, den]]
            .sum().sort_values(["player_id", "season"])
        )
        scores = {}
        for label, window in (("1yr", 1), ("2yr", 2), ("3yr", 3), ("career", 99)):
            prior, following = [], []
            for _, block in seasons.groupby("player_id"):
                block = block.reset_index(drop=True)
                for i in range(1, len(block)):
                    low = max(0, i - window)
                    numerator = block[num][low:i].sum()
                    denominator = block[den][low:i].sum()
                    need = floor * min(window, i)
                    if denominator >= need and block[den][i] >= floor:
                        prior.append(numerator / denominator)
                        following.append(block[num][i] / block[den][i])
            scores[label] = (
                stats.pearsonr(prior, following)[0] ** 2 if len(prior) >= 200 else np.nan
            )
        best = max(
            (v for k, v in scores.items() if k != "1yr" and np.isfinite(v)),
            default=np.nan,
        )
        print(f"{name:24} {scores['1yr']:8.1%} {scores['2yr']:8.1%} "
              f"{scores['3yr']:8.1%} {scores['career']:8.1%} "
              f"{best - scores['1yr']:+8.1%}")


def main() -> int:
    weekly = _weekly()
    print("=== how much of each response is signal, and how much is left to find ===\n")
    reliability_table(weekly)
    print("\n=== does more of a player's history beat the one season the layer uses? ===")
    window_table(weekly)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
