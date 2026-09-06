"""What is a fantasy week's second, third and fourth deadline actually worth?

A week is not one deadline. Every player locks when his own game kicks off, so a
manager keeps deciding around whoever has not started: the Sunday afternoon flex
is open while the morning games are being played, and Monday night is open all
weekend. The environment now resolves a week that way. This measures whether it
matters.

The answer turns entirely on **what changes between decision points**, and only
one thing does: the score. Nobody learns anything about a player's Sunday
afternoon by watching Sunday morning. So three policies are run in the same
seats, paired on seed:

``once``
    The trained agent, deciding at the first deadline and never again. The
    baseline, and exactly what it did before this existed.
``chase``
    The same agent, which at the last decision point swaps toward volatility
    when it is behind and toward steadiness when it is ahead. The realistic use
    of the extra deadlines, and the only one available without new data.
``late-oracle``
    The same agent, which at every decision point after the first knows what the
    players who have not yet started are really about to score. Not a
    competitor and not really a ceiling on *sequencing*: only two clubs play
    before Sunday, so by the second decision point almost the whole roster is
    still movable and this is close to a full oracle. What it measures is how
    much room a better projection has if it can be applied late -- which is
    worth knowing, and is a different question from what the extra deadlines buy
    on their own.

    python scripts/validate_slots.py --seasons 2023 2024 2025 --seeds 20
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from ffmodel.league.agent import LinearAgent, Scaler, as_waiver_policy
from ffmodel.league.config import LeagueConfig
from ffmodel.league.env import FantasyLeagueEnv, run_episode
from ffmodel.league.features import build_feature_tables, build_volatility
from ffmodel.league.kickoff import build_kickoff_slots
from ffmodel.league.policies import EwmaPolicy, SeasonPolicy
from ffmodel.league.pool import build_player_pool


class ChaseTheWin(LinearAgent):
    """Tilt toward or away from volatility once the game is decidable.

    The trigger is the **projected** final margin, not the raw one. Forty points
    behind on Sunday morning with six starters left is an ordinary week; forty
    behind with one left is a decision, and a policy that cannot tell those
    apart will reshuffle its lineup every week for no reason. So the deficit is
    estimated as what is already banked plus what each side's remaining starters
    are worth, using this agent's own projection for its own players and the
    same per-starter average for the opponent's, which is what the scoreboard's
    "yet to play" count supports.

    Behind on that estimate, buy variance. Ahead, sell it. Scaled by the size of
    the projected gap, so a one-point edge does not trigger a reshuffle.
    """

    reactive = True

    def __init__(
        self, *args, volatility=None, strength=1.0, residual=True,
        buy_only=False, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self._volatility = volatility
        self.strength = float(strength)
        # Volatility correlates +0.64 with the agent's own score: the players
        # who swing hardest are mostly the good ones. Tilting on it raw
        # therefore does not express "prefer risk", it expresses "prefer good
        # players" when behind and "bench your best" when ahead -- which is why
        # the raw version gains points and loses games. Residualising against
        # the score leaves the part of volatility that is actually risk.
        self.residual = bool(residual)
        # And there is a case for only ever buying: being ahead late is already
        # a won game more often than not, so trading expectation for certainty
        # has little to buy and a starter to lose.
        self.buy_only = bool(buy_only)

    def _spread(self, keys, week, scale, values):
        """Volatility, rescaled into the units the ranking is expressed in.

        These are not naturally comparable: the agent's score is a linear
        combination of standardised features and lives within about one unit,
        while volatility is fantasy points and runs to fifteen. Added raw, the
        tilt is three times the entire spread of the score and simply replaces
        the ranking with a volatility ranking. Standardising it makes
        ``strength`` dimensionless -- the fraction of a score-standard-deviation
        the tilt can move a player at a full deficit.
        """
        index = pd.MultiIndex.from_arrays([list(keys), [int(week)] * len(keys)])
        spread = np.nan_to_num(
            self._volatility.reindex(index).to_numpy(float), nan=0.0
        )
        if self.residual and values.std() > 1e-9 and spread.std() > 1e-9:
            slope = np.cov(values, spread)[0, 1] / values.var()
            spread = spread - slope * (values - values.mean())
        if spread.std() < 1e-9:
            return np.zeros_like(spread)
        return (spread - spread.mean()) / spread.std() * scale

    def score(self, player_keys, history, week, board, state=None):
        base = self.values(player_keys, week)
        if state is None or self._volatility is None or not state.remaining:
            return base

        # What the rest of the week is worth to each side.
        mine = [base.get(key, 0.0) for key in state.playable]
        mine.sort(reverse=True)
        per_starter = float(np.mean(mine[: state.remaining])) if mine else 0.0
        projected = state.points + per_starter * state.remaining
        opponent = state.opponent_points + per_starter * state.opponent_remaining

        deficit = opponent - projected
        if abs(deficit) < 3.0:
            return base
        if deficit <= 0 and self.buy_only:
            return base
        direction = 1.0 if deficit > 0 else -1.0
        weight = direction * self.strength * min(abs(deficit) / 25.0, 1.0)
        values = np.fromiter((base.get(k, 0.0) for k in player_keys), float)
        scale = values.std() if values.size else 0.0
        spread = self._spread(list(player_keys), week, scale, values)
        return {
            key: base.get(key, 0.0) + weight * value
            for key, value in zip(player_keys, spread)
        }


class LateOracle(LinearAgent):
    """Perfect knowledge of whoever has not kicked off yet. The ceiling."""

    reactive = True

    def __init__(self, *args, truth=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._truth = truth

    def score(self, player_keys, history, week, board, state=None):
        if state is None or state.slot == 0 or self._truth is None:
            return self.values(player_keys, week)
        out = self.values(player_keys, week)
        for key in player_keys:
            if key in state.playable:
                value = self._truth.get((key, int(week)))
                if value is not None and np.isfinite(value):
                    out[key] = float(value)
        return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent", type=Path, default=Path("artifacts/league_agent.json"))
    parser.add_argument("--seasons", type=int, nargs="+", default=[2023, 2024, 2025])
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument(
        "--strengths", type=float, nargs="+", default=[0.25, 0.5, 1.0],
        help="how hard to tilt toward variance; swept so a null is a null "
             "rather than a badly-chosen knob",
    )
    parser.add_argument("--output", type=Path, default=Path("reports/league_slots.json"))
    args = parser.parse_args(argv)

    saved = json.loads(args.agent.read_text("utf-8"))
    theta = np.asarray(saved["theta"], float)
    scaler = Scaler.from_dict(saved["scaler"])

    pool = build_player_pool(args.seasons)
    tables = build_feature_tables(pool, args.seasons)
    volatility = {s: build_volatility(pool, s) for s in args.seasons}
    kickoffs = {s: build_kickoff_slots(args.seasons, s) for s in args.seasons}
    truth = {
        s: pool[pool["season"] == s].set_index(["player_key", "week"])["points"].to_dict()
        for s in args.seasons
    }
    config = LeagueConfig()
    print(
        f"{len(args.seasons)} seasons x {args.seeds} seeds; "
        f"weeks span "
        f"{np.mean([kickoffs[s].slots_in(w) for s in args.seasons for w in config.weeks]):.1f} "
        "kickoff slots on average"
    )

    rows = []
    for season in args.seasons:
        table = tables[season]
        for seed in range(args.seeds):
            contenders = ["once", "late-oracle"] + [
                f"{kind}{s:g}"
                for s in args.strengths
                for kind in ("chase", "chase-resid", "chase-buy")
            ]
            for name in contenders:
                env = FantasyLeagueEnv(
                    pool, season=season, config=config, seed=seed,
                    kickoffs=kickoffs[season],
                    opponent=SeasonPolicy(table=table),
                    roster_valuation=EwmaPolicy(table=table),
                )
                if name == "once":
                    agent = LinearAgent.from_parameters(theta, table, scaler)
                elif name.startswith("chase"):
                    kind, _, level = name.rpartition("-") if "-" in name else ("chase", "", "")
                    strength = float(name.split("chase")[-1].lstrip("-resiuybd"))
                    agent = ChaseTheWin.from_parameters(
                        theta, table, scaler, volatility=volatility[season],
                        strength=strength,
                        residual=name.startswith(("chase-resid", "chase-buy")),
                        buy_only=name.startswith("chase-buy"),
                    )
                else:
                    agent = LateOracle.from_parameters(
                        theta, table, scaler, truth=truth[season]
                    )
                result = run_episode(env, agent, waiver_policy=as_waiver_policy(agent))
                rows.append(
                    {
                        "policy": name, "season": season, "seed": seed,
                        "wins": result.wins, "points": result.total_points,
                    }
                )

    frame = pd.DataFrame(rows)
    print("\n=== averaged over every seat ===")
    print(
        frame.groupby("policy")[["wins", "points"]].mean().round(3).to_string()
    )

    wide = {
        m: frame.pivot_table(index=["season", "seed"], columns="policy", values=m)
        for m in ("wins", "points")
    }
    print("\n=== against deciding once, paired on seed ===")
    print(f"  {'policy':14s} {'wins':>7s} {'SE':>6s} {'t':>7s}   {'points':>8s} {'t':>7s}")
    for name in sorted(set(frame["policy"]) - {"once"}):
        dw = wide["wins"][name] - wide["wins"]["once"]
        dp = wide["points"][name] - wide["points"]["once"]
        sew = dw.std(ddof=1) / np.sqrt(len(dw))
        sep = dp.std(ddof=1) / np.sqrt(len(dp))
        print(
            f"  {name:14s} {dw.mean():+7.3f} {sew:6.3f} "
            f"{dw.mean() / sew if sew else float('nan'):+7.2f}   "
            f"{dp.mean():+8.1f} {dp.mean() / sep if sep else float('nan'):+7.2f}"
        )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps({"episodes": frame.to_dict("records")}, indent=2, default=str),
            "utf-8",
        )
        print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
