"""Prior-only features: everything here is a function of strictly earlier weeks.

A weekly panel makes leakage very easy and almost invisible. An expanding mean
that forgets to shift includes the week being predicted, which on a
seventeen-game series is a large share of the average and produces a model that
validates beautifully and cannot be used. So the shift is structural here rather
than something each feature has to remember: every history column is produced by
:func:`_prior`, which applies its statistic and then lags it by one row within
the group. There is no path through this module that reaches the current week.

What the history is built from, and why:

**Player history spans the career, not the season.** Week 1 of a fourth season
knows about the previous three. This is the whole reason the weekly layer exists
-- at season level the median player has one prior observation, and here he has
however many weeks he has played.

**Two averages of points, not one.** The mean over all rostered weeks includes
the zeros and answers "what does starting him cost me on average". The mean over
weeks he played excludes them and answers "what is he worth when he suits up".
The first is the right target for a lineup decision and the second is the right
input to a hurdle model, and they are very different numbers for an injury-prone
player. Carrying only the pooled one throws away the distinction the response
turns on.

**Team context is an exponentially weighted average across seasons.** An offense
in week 1 has no current-season history, and a within-season expanding mean
would leave it undefined exactly where the draft-adjacent decisions are made.
Ordering team-weeks by (season, week) and decaying across the boundary gives
week 1 last year's late-season offense, which is the best available answer and
is never missing.

The decay rate is fixed a priori at a four-game half-life rather than tuned. The
season layer's analogous constant is 0.50 on a one-year step; four games is the
same instinct at this cadence, and choosing it by holdout score would spend the
holdout on a nuisance parameter.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Half-life of four games, fixed a priori. See the module docstring.
HISTORY_HALFLIFE = 4.0
HISTORY_ALPHA = 1.0 - 0.5 ** (1.0 / HISTORY_HALFLIFE)

# Team offences move more slowly than a single player's week-to-week usage, and
# this average has to reach across a season boundary to be defined in week 1.
TEAM_HALFLIFE = 8.0
TEAM_ALPHA = 1.0 - 0.5 ** (1.0 / TEAM_HALFLIFE)

# Where "early season" ends, for the purpose of letting the draft board matter
# more while a player's current-season history is still thin. Four games is a
# quarter of a season and matches the horizon buckets the evaluation reports, so
# it is not a third boundary invented for one feature.
EARLY_SEASON_WEEKS = 4

PLAYER_ORDER = ["player_key", "season", "week"]

# Volume columns whose lagged rate the models are allowed to read.
VOLUME_COLUMNS = ("targets", "rush_att", "pass_att", "receptions")

FEATURE_COLUMNS = (
    "prior_weeks",
    "prior_games",
    "prior_play_rate",
    "recent_play_rate",
    "weeks_since_played",
    "prior_points_mean",
    "prior_points_recent",
    "prior_points_given_played",
    "prior_points_recent_given_played",
    "prior_points_sd",
    "prior_targets_recent",
    "prior_rush_att_recent",
    "prior_pass_att_recent",
    "prior_target_share_recent",
    "prior_rush_share_recent",
    "team_plays_recent",
    "team_points_recent",
    "team_pass_att_recent",
    "team_rush_att_recent",
    "defense_points_allowed_recent",
    "def_rush_att_allowed",
    "def_rush_yds_allowed",
    "def_rush_ypc_allowed",
    "def_rush_epa_allowed",
    "def_targets_allowed",
    "def_rec_yds_allowed",
    "def_rec_epa_allowed",
    "own_def_rush_epa_allowed",
    "own_def_rec_epa_allowed",
    "own_def_rec_yds_allowed",
)

# Prospective rather than historical: the market's forecast of the game about to
# be played. Not produced by :func:`_prior` because they need no lag -- a
# closing line is published before kickoff and is legitimately known at decision
# time. They are attached in :mod:`ffmodel.weekly.frame`.
MARKET_COLUMNS = (
    "spread",
    "game_total",
    "implied_team_total",
    "implied_opponent_total",
)


def _prior(
    frame: pd.DataFrame,
    keys: list[str],
    values: pd.Series,
    *,
    how: str,
    alpha: float | None = None,
) -> pd.Series:
    """A history statistic of ``values``, lagged one row inside each group.

    The lag is applied here and only here. ``how`` is ``"mean"`` (expanding),
    ``"ewm"`` (exponentially weighted), ``"sum"``, ``"count"`` or ``"std"``.
    """
    grouped = values.groupby([frame[k] for k in keys], sort=False)
    if how == "mean":
        statistic = grouped.expanding().mean()
    elif how == "sum":
        statistic = grouped.expanding().sum()
    elif how == "count":
        statistic = grouped.expanding().count()
    elif how == "std":
        statistic = grouped.expanding().std()
    elif how == "ewm":
        if alpha is None:
            raise ValueError("an exponentially weighted statistic needs an alpha")
        statistic = grouped.ewm(alpha=alpha, adjust=True).mean()
    else:
        raise ValueError(f"unknown history statistic {how!r}")
    # A grouped expanding/ewm returns a (group, original position) MultiIndex,
    # whose row order is by group rather than by frame. Dropping the group
    # levels and reindexing puts every value back on the row it came from.
    # Positional reassignment happens to work while the frame is sorted by the
    # grouping key and silently misaligns the moment it is not, which is not a
    # property worth depending on.
    if isinstance(statistic.index, pd.MultiIndex):
        statistic = statistic.droplevel(list(range(statistic.index.nlevels - 1)))
    statistic = statistic.reindex(frame.index)
    return statistic.groupby([frame[k] for k in keys], sort=False).shift(1)


def _weeks_since_played(frame: pd.DataFrame) -> pd.Series:
    """Panel weeks since this player last recorded a line, lagged.

    Distinguishes a player returning from four weeks out from one who played
    last week, which the play-rate averages cannot: both can sit at 0.75.
    """
    played = frame["played"].to_numpy(int)
    keys = frame["player_key"].to_numpy()
    out = np.full(len(frame), np.nan)
    gap = 0
    started = False
    previous = None
    for i in range(len(frame)):
        if keys[i] != previous:
            gap, started, previous = 0, False, keys[i]
        out[i] = gap if started else np.nan
        if played[i] == 1:
            gap, started = 1, True
        elif started:
            gap += 1
    return pd.Series(out, index=frame.index)


def _team_history(panel: pd.DataFrame) -> pd.DataFrame:
    """Lagged team-week offence, decayed across the season boundary."""
    weeks = (
        panel[
            [
                "season",
                "week",
                "team",
                "team_plays",
                "team_points",
                "team_pass_att",
                "team_rush_att",
            ]
        ]
        .drop_duplicates(subset=["season", "week", "team"])
        .sort_values(["team", "season", "week"], kind="mergesort")
        .reset_index(drop=True)
    )
    out = weeks[["season", "week", "team"]].copy()
    for column, name in (
        ("team_plays", "team_plays_recent"),
        ("team_points", "team_points_recent"),
        ("team_pass_att", "team_pass_att_recent"),
        ("team_rush_att", "team_rush_att_recent"),
    ):
        out[name] = _prior(
            weeks, ["team"], weeks[column].astype(float), how="ewm", alpha=TEAM_ALPHA
        )
    return out


def _defense_phase_history(panel: pd.DataFrame) -> pd.DataFrame:
    """Lagged run and pass defence, kept apart, in volume *and* efficiency.

    "Good against the run" is two claims and a running back cares about both.
    A defence can hold rushing yards down because it is hard to run on, or
    because game script means nobody runs on it -- and those point in opposite
    directions for a back's workload. So the volume conceded (carries, targets)
    and the efficiency conceded (yards per carry, EPA per play) are separate
    columns rather than one points-allowed aggregate.

    Splitting by phase is the other half. Points allowed to running backs mixes
    a defence's run front with its coverage of backs out of the backfield, and a
    defence is routinely good at one and poor at the other. Keyed on the defence
    alone rather than defence-by-position, because that is what a phase is.
    """
    rows = panel[panel["opponent"].notna()].copy()
    per_game = (
        rows.groupby(["season", "week", "opponent"], as_index=False)
        .agg(
            rush_att=("rush_att", "sum"),
            rush_yds=("rush_yds", "sum"),
            rush_epa=("rush_epa", "sum"),
            targets=("targets", "sum"),
            rec_yds=("rec_yds", "sum"),
            rec_epa=("rec_epa", "sum"),
        )
        .rename(columns={"opponent": "defense"})
        .sort_values(["defense", "season", "week"], kind="mergesort")
        .reset_index(drop=True)
    )
    # Efficiency is formed before lagging, so each is a ratio of two numbers
    # from the same game rather than a lagged total over a lagged count.
    with np.errstate(divide="ignore", invalid="ignore"):
        per_game["rush_ypc"] = np.divide(
            per_game["rush_yds"], per_game["rush_att"],
            out=np.full(len(per_game), np.nan), where=per_game["rush_att"] > 0,
        )
        per_game["rush_epa_play"] = np.divide(
            per_game["rush_epa"], per_game["rush_att"],
            out=np.full(len(per_game), np.nan), where=per_game["rush_att"] > 0,
        )
        per_game["rec_epa_play"] = np.divide(
            per_game["rec_epa"], per_game["targets"],
            out=np.full(len(per_game), np.nan), where=per_game["targets"] > 0,
        )
    out = per_game[["season", "week", "defense"]].copy()
    for column, name in (
        ("rush_att", "def_rush_att_allowed"),
        ("rush_yds", "def_rush_yds_allowed"),
        ("rush_ypc", "def_rush_ypc_allowed"),
        ("rush_epa_play", "def_rush_epa_allowed"),
        ("targets", "def_targets_allowed"),
        ("rec_yds", "def_rec_yds_allowed"),
        ("rec_epa_play", "def_rec_epa_allowed"),
    ):
        out[name] = _prior(
            per_game,
            ["defense"],
            per_game[column].astype(float),
            how="ewm",
            alpha=TEAM_ALPHA,
        )
    return out


def _defense_history(panel: pd.DataFrame) -> pd.DataFrame:
    """Lagged points a defence has allowed to each position.

    The matchup adjustment every lineup column in the sport is built on, in its
    strongest reasonable form: not a season-long rank but a recency-weighted
    average, on the same decay as everything else, lagged so the week being
    predicted is not in it. Points allowed are summed over the opposing players
    at that position, which is the quantity the folk version is reaching for.

    Keyed on (defence, position), so a defence that is soft against tight ends
    and hard against receivers is described as such rather than averaged into one
    number.
    """
    rows = panel[panel["opponent"].notna()].copy()
    allowed = (
        rows.groupby(["season", "week", "opponent", "position"], as_index=False)[
            "points"
        ]
        .sum()
        .rename(columns={"opponent": "defense", "points": "points_allowed"})
        .sort_values(["defense", "position", "season", "week"], kind="mergesort")
        .reset_index(drop=True)
    )
    allowed["defense_points_allowed_recent"] = _prior(
        allowed,
        ["defense", "position"],
        allowed["points_allowed"].astype(float),
        how="ewm",
        alpha=TEAM_ALPHA,
    )
    return allowed[
        ["season", "week", "defense", "position", "defense_points_allowed_recent"]
    ]


def add_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Attach every prior-only feature to the panel, in panel order."""
    frame = panel.sort_values(PLAYER_ORDER, kind="mergesort").reset_index(drop=True)
    keys = ["player_key"]

    played = frame["played"].astype(float)
    points = frame["points"].astype(float)

    frame["prior_weeks"] = _prior(frame, keys, pd.Series(1.0, index=frame.index),
                                  how="count").fillna(0.0)
    frame["prior_games"] = _prior(frame, keys, played, how="sum").fillna(0.0)
    frame["prior_play_rate"] = _prior(frame, keys, played, how="mean")
    frame["recent_play_rate"] = _prior(
        frame, keys, played, how="ewm", alpha=HISTORY_ALPHA
    )
    frame["weeks_since_played"] = _weeks_since_played(frame)

    frame["prior_points_mean"] = _prior(frame, keys, points, how="mean")
    frame["prior_points_recent"] = _prior(
        frame, keys, points, how="ewm", alpha=HISTORY_ALPHA
    )
    frame["prior_points_sd"] = _prior(frame, keys, points, how="std")

    # Averages over the weeks he actually played. Masking with NaN rather than
    # zero is what makes these conditional: pandas' expanding and ewm statistics
    # skip missing entries, so a week he sat out does not enter the mean at all.
    played_points = points.where(frame["played"].eq(1))
    frame["prior_points_given_played"] = _prior(
        frame, keys, played_points, how="mean"
    )
    frame["prior_points_recent_given_played"] = _prior(
        frame, keys, played_points, how="ewm", alpha=HISTORY_ALPHA
    )

    for column in VOLUME_COLUMNS:
        if column == "receptions":
            continue
        masked = frame[column].astype(float).where(frame["played"].eq(1))
        frame[f"prior_{column}_recent"] = _prior(
            frame, keys, masked, how="ewm", alpha=HISTORY_ALPHA
        )

    # Shares are formed before lagging, so each is a ratio of two numbers from
    # the same week rather than a lagged count over a lagged denominator.
    with np.errstate(divide="ignore", invalid="ignore"):
        target_share = np.divide(
            frame["targets"].to_numpy(float),
            frame["team_targets"].to_numpy(float),
            out=np.full(len(frame), np.nan),
            where=frame["team_targets"].to_numpy(float) > 0,
        )
        rush_share = np.divide(
            frame["rush_att"].to_numpy(float),
            frame["team_rush_att"].to_numpy(float),
            out=np.full(len(frame), np.nan),
            where=frame["team_rush_att"].to_numpy(float) > 0,
        )
    for name, values in (
        ("prior_target_share_recent", target_share),
        ("prior_rush_share_recent", rush_share),
    ):
        masked = pd.Series(values, index=frame.index).where(frame["played"].eq(1))
        frame[name] = _prior(frame, keys, masked, how="ewm", alpha=HISTORY_ALPHA)

    if "adp_rank" in frame.columns:
        rank = pd.to_numeric(frame["adp_rank"], errors="coerce")
        drafted = pd.to_numeric(
            frame.get("adp_drafted", rank.notna().astype(float)), errors="coerce"
        ).fillna(0.0)
        # Undrafted is a value, not a gap: a player the board declined to rank is
        # placed one past the deepest rank it published rather than imputed to
        # the middle of it, which would assert an average draft position for
        # somebody nobody drafted. This is the season layer's encoding.
        deepest = float(rank.max()) if rank.notna().any() else 300.0
        frame["adp_log_rank"] = np.log(rank.fillna(deepest + 1.0).clip(lower=1.0))
        frame["adp_drafted"] = drafted
        early = frame["week"].le(EARLY_SEASON_WEEKS).astype(float)
        frame["adp_log_rank_early"] = frame["adp_log_rank"] * early
        frame["adp_drafted_early"] = frame["adp_drafted"] * early

    team = _team_history(frame)
    frame = frame.merge(team, on=["season", "week", "team"], how="left")

    defense = _defense_history(frame)
    frame = frame.merge(
        defense,
        left_on=["season", "week", "opponent", "position"],
        right_on=["season", "week", "defense", "position"],
        how="left",
    ).drop(columns=["defense"], errors="ignore")

    phase = _defense_phase_history(frame)
    frame = frame.merge(
        phase,
        left_on=["season", "week", "opponent"],
        right_on=["season", "week", "defense"],
        how="left",
    ).drop(columns=["defense"], errors="ignore")

    # The player's *own* defence, from the same table. A team that cannot get
    # off the field plays from behind and throws to keep up, which is a fact
    # about its offence's volume and is invisible in the offence's own history
    # until it has already happened.
    own = phase.rename(
        columns={
            column: column.replace("def_", "own_def_")
            for column in phase.columns
            if column.startswith("def_")
        }
    )
    frame = frame.merge(
        own,
        left_on=["season", "week", "team"],
        right_on=["season", "week", "defense"],
        how="left",
    ).drop(columns=["defense"], errors="ignore")

    return frame


def relevant_population(frame: pd.DataFrame, *, min_points: float = 4.0) -> pd.Series:
    """Rows a fantasy manager would plausibly be deciding about, from prior data.

    The season layer learned this the expensive way: fitting and scoring on every
    rostered player, most of whom are fringe, cost 3.6 CRPS points on the players
    who actually get drafted and accounted for most of an apparent deficit
    against the draft board. A pooled weekly metric has the same defect and worse
    -- roughly half the panel is a third-string player whose zero nobody needed a
    model to forecast, and a forecast of zero for all of them would score well.

    The filter reads only lagged columns, so it defines a population without
    looking at the outcome. ``min_points`` is on the recency-weighted average of
    weeks he played: a threshold on what he is worth when he suits up, not on
    whether he has been suiting up, so an injured starter stays in the population
    he belongs to.
    """
    recent = pd.to_numeric(frame["prior_points_recent_given_played"], errors="coerce")
    seen = pd.to_numeric(frame["prior_games"], errors="coerce").fillna(0.0)
    return (recent >= min_points) & (seen >= 4)
