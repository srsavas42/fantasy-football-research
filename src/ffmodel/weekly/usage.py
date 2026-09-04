"""How a player's role moves, fitted as a process rather than assumed away.

Every simulator in this package so far has held usage frozen. The recursive one
carries a player's *scoring* forward — a drawn week changes his points average and
therefore next week's projection — but his share of his team's carries and
targets, and his share of its snaps, stay at whatever they were in week ``w`` for
all seventeen simulated games. That is why its spread had to be topped up with a
fitted drift term: role uncertainty is most of what is unknown about a season and
none of it was being generated.

This fits the missing process. A share is bounded in [0, 1], so it is modelled in
logit space, where the dynamics are close to linear and the innovation close to
Gaussian:

    logit(s_t) = a + b * logit(s_{t-1}) + c * L_{t-1} + e

``b`` is persistence — how much of last week's role carries into this one — and
``c`` is reversion toward ``L``, the player's own long-run level rather than the
population's. Reverting to the population mean would drag every star toward
average and every backup up toward it, which is not what roles do. Both
coefficients and the innovation scale are fitted per position, because a running
back's carry share and a receiver's target share do not move the same way.

Two shares are simulated: snap share, which is the broadest statement of role,
and the player's primary opportunity share — carries for backs, targets for
everyone else.

**On double counting.** The magnitude model's residual pool was fitted on real
weeks, where realised usage already differed from lagged usage; some of the
spread being added here is therefore spread the residual already contained.
Rather than argue about how much, the drift term downstream is re-fitted with
this process switched on. If usage simulation generates the role uncertainty that
was previously being injected, the fitted drift falls toward zero on its own, and
that is a measurement rather than a claim.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

POSITIONS = ("QB", "RB", "WR", "TE")

# Shares are squeezed off the boundary before the logit: a week with zero targets
# is a real observation and must not become negative infinity.
EPSILON = 0.005

# Below this many usable transitions a position's process is not identified and
# the pooled fit is used instead.
MIN_TRANSITIONS = 400

# The long-run level a share reverts toward, as an exponentially weighted mean.
# Deliberately slow -- this is the player's standing role, not his recent form,
# and the recent form is already the other term in the regression.
LEVEL_HALFLIFE = 12.0
LEVEL_ALPHA = 1.0 - 0.5 ** (1.0 / LEVEL_HALFLIFE)


def logit(share: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(share, dtype=float), EPSILON, 1.0 - EPSILON)
    return np.log(clipped / (1.0 - clipped))


def expit(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(np.asarray(value, dtype=float), -30.0, 30.0)))


def primary_share_column(position: pd.Series) -> pd.Series:
    """Carries for backs, targets for everyone else."""
    return np.where(position.eq("RB"), "rush_share", "target_share")


@dataclass
class ShareDynamics:
    """One fitted AR(1)-with-reversion process, for one share and one position."""

    intercept: float = 0.0
    persistence: float = 0.0
    reversion: float = 0.0
    innovation: float = 0.0

    def step(
        self, previous: np.ndarray, level: np.ndarray, noise: np.ndarray
    ) -> np.ndarray:
        return (
            self.intercept
            + self.persistence * previous
            + self.reversion * level
            + self.innovation * noise
        )


@dataclass
class UsageProcess:
    """Fitted share dynamics, keyed by (share name, position)."""

    by_key: dict = field(default_factory=dict)
    pooled: dict = field(default_factory=dict)

    @staticmethod
    def observed_shares(frame: pd.DataFrame) -> pd.DataFrame:
        """Realised shares per played week, plus the lagged level they revert to."""
        work = frame[frame["played"].eq(1)].copy()
        with np.errstate(divide="ignore", invalid="ignore"):
            for name, numerator, denominator in (
                ("target_share", "targets", "team_targets"),
                ("rush_share", "rush_att", "team_rush_att"),
            ):
                top = pd.to_numeric(work[numerator], errors="coerce").to_numpy(float)
                bottom = pd.to_numeric(work[denominator], errors="coerce").to_numpy(float)
                work[name] = np.divide(
                    top, bottom, out=np.full(len(work), np.nan), where=bottom > 0
                )
        work["snap_share"] = pd.to_numeric(
            work.get("snap_share", np.nan), errors="coerce"
        )
        work["primary_share"] = np.where(
            work["position"].eq("RB"), work["rush_share"], work["target_share"]
        )
        return work.sort_values(["player_key", "season", "week"], kind="mergesort")

    def fit(self, frame: pd.DataFrame) -> "UsageProcess":
        work = self.observed_shares(frame)
        for name in ("primary_share", "snap_share"):
            values = pd.to_numeric(work[name], errors="coerce")
            grouped = values.groupby(
                [work["player_key"], work["season"]], sort=False
            )
            previous = grouped.shift(1)
            # The level is a slow average of what came before, so it is a
            # statement about his standing role at the time, not hindsight.
            level = (
                values.groupby(work["player_key"], sort=False)
                .apply(lambda s: s.ewm(alpha=LEVEL_ALPHA, adjust=True).mean().shift(1))
                .droplevel(0)
                .reindex(work.index)
            )
            usable = values.notna() & previous.notna() & level.notna()
            y = logit(values[usable].to_numpy())
            x_previous = logit(previous[usable].to_numpy())
            x_level = logit(level[usable].to_numpy())
            position = work.loc[usable, "position"].to_numpy()

            self.pooled[name] = self._solve(y, x_previous, x_level)
            for spot in POSITIONS:
                want = position == spot
                if want.sum() < MIN_TRANSITIONS:
                    continue
                self.by_key[(name, spot)] = self._solve(
                    y[want], x_previous[want], x_level[want]
                )
        return self

    @staticmethod
    def _solve(y: np.ndarray, previous: np.ndarray, level: np.ndarray) -> ShareDynamics:
        design = np.column_stack([np.ones(len(y)), previous, level])
        beta, *_ = np.linalg.lstsq(design, y, rcond=None)
        residual = y - design @ beta
        return ShareDynamics(
            intercept=float(beta[0]),
            persistence=float(beta[1]),
            reversion=float(beta[2]),
            innovation=float(residual.std(ddof=3)) if len(y) > 3 else 0.0,
        )

    def get(self, name: str, position: str) -> ShareDynamics:
        return self.by_key.get((name, position), self.pooled.get(name, ShareDynamics()))

    def summary(self) -> pd.DataFrame:
        rows = []
        for (name, spot), dynamics in sorted(self.by_key.items()):
            rows.append(
                {
                    "share": name,
                    "position": spot,
                    "persistence": dynamics.persistence,
                    "reversion": dynamics.reversion,
                    "innovation": dynamics.innovation,
                }
            )
        return pd.DataFrame(rows)


# Horizons used to separate a wandering level from mean-reverting noise. The
# short ones are dominated by the stationary part; the long ones are where a
# random walk shows itself.
VARIANCE_HORIZONS = (1, 2, 4, 6, 8, 10, 12)

# Below this the long-horizon variance is estimated from too few pairs to
# support a slope, and the random-walk term is reported as zero.
MIN_PAIRS = 300


def estimate_random_walk(
    frame: pd.DataFrame, column: str = "primary_share"
) -> tuple[float, pd.DataFrame]:
    """How much of a player's role wanders permanently, by variance ratio.

    A mean-reverting process and a random walk look identical one week apart and
    completely different ten weeks apart, which is what makes the horizon the
    identifying variable. For an AR(1) around a fixed level the variance of the
    ``h``-week change flattens out as ``h`` grows -- there is a ceiling, set by
    how far the process ever strays from its level. For a random walk it keeps
    growing, linearly in ``h``, forever.

    So measure ``Var(logit s_{t+h} - logit s_t)`` within a player-season across
    several horizons and read the two components off the shape:

        Var(h) = ceiling * (1 - rho^h) + h * sigma_rw^2

    The slope of the long-horizon end estimates ``sigma_rw^2`` per week directly,
    without having to pin down the stationary part. Fitted on the horizons from
    six weeks out, where the stationary term has essentially finished growing and
    anything still accumulating is the walk.

    This is the quantity the previous round was missing. A stationary process
    contributes about ``sqrt(G)`` to a ``G``-game total and a wandering level
    contributes ``G``, which is why an AR(1) fitted around a slowly-updating
    anchor generated so little season-total spread and a flat scalar drift
    generated the right amount by accident.
    """
    work = UsageProcess.observed_shares(frame)
    values = pd.to_numeric(work[column], errors="coerce")
    usable = values.notna()
    logits = pd.Series(np.nan, index=work.index)
    logits[usable] = logit(values[usable].to_numpy())

    rows = []
    grouped = logits.groupby([work["player_key"], work["season"]], sort=False)
    for horizon in VARIANCE_HORIZONS:
        change = grouped.diff(horizon)
        clean = change.dropna()
        if len(clean) < MIN_PAIRS:
            continue
        rows.append(
            {"horizon": horizon, "pairs": int(len(clean)), "variance": float(clean.var(ddof=1))}
        )
    table = pd.DataFrame(rows)
    if len(table) < 3:
        return 0.0, table

    tail = table[table["horizon"] >= 6]
    if len(tail) < 2:
        return 0.0, table
    design = np.column_stack([np.ones(len(tail)), tail["horizon"].to_numpy(float)])
    beta, *_ = np.linalg.lstsq(design, tail["variance"].to_numpy(float), rcond=None)
    slope = float(beta[1])
    return (float(np.sqrt(slope)) if slope > 0 else 0.0), table
