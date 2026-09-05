"""Kickers and team defenses: the two starting slots the panel never modelled.

The skill panel is built out of opportunity share -- targets, carries, snaps --
and none of those words mean anything for a kicker or a defense. So these two get
their own panel and their own features rather than a position dummy bolted onto
a design whose columns they would all read as missing. What they share with the
skill panel is the *interface*: the same row schema, the same walk-forward, the
same estimator protocol, so a start/sit or draft agent can concatenate all six
positions into one table of comparable points.

**Why each one is a different problem.**

A kicker's week is an opportunity count times a conversion rate, and the two have
very different persistence. How many field goals he attempts is a property of his
offence -- one that drives well and stalls inside the 30 kicks a lot -- and it is
about as predictable as any team quantity. Whether he makes them is close to
unpredictable at a one-week horizon; kicker accuracy is famously nearly all
noise, which is why the model leans on volume and the implied team total and
treats the leg itself as a small correction rather than the signal. Distance is
the exception that is real: a kicker trusted from 50 gets attempts a
short-legged one never sees, so ``prior_fg_long_recent`` and the long-tier
attempt rate are carried separately.

A defense's week is mostly **the other team's offence**. The event half -- sacks,
takeaways, touchdowns -- is a modest and noisy count. The points-allowed half is
a step function of the opponent's final score and dominates the variance. That
makes ``implied_opponent_total``, straight off the closing line, the single most
informative column in this frame, and it is the reason a DST projection is much
more a projection of the opponent than of the defense. The panel carries the
defense's own recent form too, but the line is doing most of the work and the
fitted coefficients say so.

**Availability.** A team defense is never inactive, so its hurdle is degenerate
and it is fitted as a plain regression on played team-weeks. A kicker is
essentially always active when rostered -- clubs carry exactly one -- but not
quite always, and a mid-season replacement is a real event, so kickers keep the
hurdle.

**Weather.** This is where the physical conditions have their most credible claim
in the whole package: wind moves a kicked ball much further off line than a
thrown one, and a 50-yard attempt into a gale is a different proposition from the
same attempt in a dome. :mod:`ffmodel.weekly.weather` is attached here for the
same measured treatment it gets on the skill panel, with the same caveat that the
two readings are recorded at the game and so report a ceiling.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ffmodel.data import ingest
from ffmodel.simulation.scoring import defense_points, kicker_points
from ffmodel.weekly.features import TEAM_ALPHA, _prior
from ffmodel.weekly.fitting import LocalResiduals, Logistic, Ridge
from ffmodel.weekly.restofseason import (
    OFFSET,
    _beta_concentration,
    _persistent_sd,
)
from ffmodel.weekly.frame import (
    CONTRACT_STATUSES,
    FIRST_PANEL_SEASON,
    _market_lines,
    _opponent_map,
)

# Half-life on a specialist's own history. Four games, matching the skill
# panel's selected decay: fixed a priori there and carried across here rather
# than re-tuned, so no holdout is spent on it in either place.
SPECIALIST_HALFLIFE = 4.0
SPECIALIST_ALPHA = 1.0 - 0.5 ** (1.0 / SPECIALIST_HALFLIFE)

KICKER_STATS = (
    "fg_att",
    "fg_made",
    "fg_long",
    "pat_att",
    "pat_made",
    "fg_made_40_49",
    "fg_att_50_plus",
)

DEFENSE_STATS = (
    "def_sacks",
    "def_interceptions",
    "fumble_recovery_opp",
    "def_tds",
    "points_allowed",
    "yards_allowed",
)


def _team_scores(seasons: Iterable[int]) -> pd.DataFrame:
    """(season, week, team) -> points this team allowed, from the schedule.

    A defense's largest scoring component is not in its own box score. It is the
    other side's final score, which lives on the schedule row.
    """
    empty = pd.DataFrame(columns=["season", "week", "team", "points_allowed", "points_for"])
    try:
        schedule = ingest.load_schedules(list(seasons))
    except Exception:
        return empty
    if schedule.empty:
        return empty
    needed = {"season", "week", "home_team", "away_team", "home_score", "away_score"}
    if not needed.issubset(schedule.columns):
        return empty
    if "game_type" in schedule.columns:
        schedule = schedule[schedule["game_type"] == "REG"]

    frames = []
    for side, other in (("home_team", "away_team"), ("away_team", "home_team")):
        mine = "home_score" if side == "home_team" else "away_score"
        theirs = "away_score" if side == "home_team" else "home_score"
        block = schedule[["season", "week", side, mine, theirs]].copy()
        block.columns = ["season", "week", "team", "points_for", "points_allowed"]
        frames.append(block)
    out = pd.concat(frames, ignore_index=True)
    for column in ("season", "week"):
        out[column] = pd.to_numeric(out[column], errors="coerce").astype("Int64")
    for column in ("points_for", "points_allowed"):
        out[column] = pd.to_numeric(out[column], errors="coerce")
    out["team"] = out["team"].astype(str)
    return out.dropna(subset=["season", "week"]).drop_duplicates(
        subset=["season", "week", "team"]
    )


def _kicker_stats(seasons: Iterable[int]) -> pd.DataFrame:
    """Per-kicker per-week lines with points already scored."""
    seasons = sorted({int(s) for s in seasons})
    stats = ingest.load_kicking(seasons)
    if stats.empty:
        return pd.DataFrame()
    stats = stats.copy()

    stats["points"] = kicker_points(stats).astype(float)
    # Long-range *attempts* rather than makes: a coach's willingness to send him
    # out from 50 is the durable part, and whether it went through is not.
    for made, missed, name in (
        ("fg_made_50_59", "fg_missed_50_59", "_50s"),
        ("fg_made_60_", "fg_missed_60_", "_60s"),
    ):
        for column in (made, missed):
            if column not in stats.columns:
                stats[column] = 0.0
    stats["fg_att_50_plus"] = (
        pd.to_numeric(stats["fg_made_50_59"], errors="coerce").fillna(0.0)
        + pd.to_numeric(stats["fg_missed_50_59"], errors="coerce").fillna(0.0)
        + pd.to_numeric(stats["fg_made_60_"], errors="coerce").fillna(0.0)
        + pd.to_numeric(stats["fg_missed_60_"], errors="coerce").fillna(0.0)
    )
    for column in KICKER_STATS:
        if column not in stats.columns:
            stats[column] = 0.0
        stats[column] = pd.to_numeric(stats[column], errors="coerce").fillna(0.0)

    keep = ["season", "week", "team", "player_id", "points", *KICKER_STATS]
    if "player_display_name" in stats.columns:
        stats["player_name"] = stats["player_display_name"]
        keep.append("player_name")
    stats = stats.rename(columns={"recent_team": "team"})
    return stats.reindex(columns=keep)


def build_kicker_panel(seasons: Iterable[int]) -> pd.DataFrame:
    """One row per rostered kicker per team-week, zeros included.

    Membership comes from the weekly roster for the same reason the skill panel
    uses it: a kicker who was on the roster and did not kick is a zero the model
    has to be able to predict, and a kicker who was not on the roster is not a
    row at all.
    """
    seasons = sorted({int(s) for s in seasons})
    too_early = [s for s in seasons if s < FIRST_PANEL_SEASON]
    if too_early:
        raise ValueError(
            f"weekly rosters start in {FIRST_PANEL_SEASON}; cannot build honest "
            f"zeros for {too_early}"
        )

    rosters = ingest.load_weekly_rosters(seasons)
    rosters = rosters[rosters.get("position").astype(str) == "K"].copy()
    if "game_type" in rosters.columns:
        rosters = rosters[rosters["game_type"] == "REG"]
    # The same contract filter the skill panel uses. Without it the frame fills
    # with practice-squad and released kickers, who are not zeros anyone has to
    # predict -- they are not on the roster in the sense the decision means.
    status = rosters.get("status")
    if status is not None:
        rosters = rosters[status.isin(CONTRACT_STATUSES)]
    rosters = rosters.rename(
        columns={"gsis_id": "player_id", "full_name": "player_name"}
    )
    members = rosters.reindex(
        columns=["season", "week", "team", "player_id", "player_name"]
    ).dropna(subset=["season", "week", "team", "player_id"])
    for column in ("season", "week"):
        members[column] = pd.to_numeric(members[column], errors="coerce").astype(int)
    members = members.drop_duplicates(subset=["season", "week", "team", "player_id"])

    stats = _kicker_stats(seasons)
    if stats.empty:
        return pd.DataFrame()
    for column in ("season", "week"):
        stats[column] = pd.to_numeric(stats[column], errors="coerce").astype("Int64")

    panel = members.merge(
        stats.drop(columns=["player_name"], errors="ignore"),
        on=["season", "week", "team", "player_id"],
        how="left",
    )
    # A kicker on the roster in a week his club had a bye is not a zero -- there
    # was no game to kick in. Team-weeks that produced no lines anywhere are byes.
    played_weeks = (
        stats[["season", "week", "team"]].drop_duplicates().assign(team_played=1)
    )
    panel = panel.merge(played_weeks, on=["season", "week", "team"], how="left")
    panel = panel[panel["team_played"] == 1].drop(columns=["team_played"])

    panel["played"] = panel["points"].notna().astype(int)
    panel["points"] = panel["points"].fillna(0.0)
    for column in KICKER_STATS:
        panel[column] = pd.to_numeric(panel[column], errors="coerce").fillna(0.0)
    panel["position"] = "K"
    panel["player_key"] = panel["player_id"].astype(str)
    return panel.sort_values(["player_key", "season", "week"], kind="mergesort").reset_index(
        drop=True
    )


def _canonical_team_codes(seasons: Iterable[int]) -> dict[tuple[int, str], str]:
    """(season, any club code) -> the code that season's schedule uses.

    nflverse is not internally consistent about relocated franchises.
    ``team_stats`` labels them by their **modern** code in every season --
    2016 Chargers are ``LAC``, 2016 Raiders are ``LV`` -- while ``schedules``
    labels them by the code in use **at the time**, ``SD`` and ``OAK``. Merging
    the two on the club code therefore drops those franchises entirely from
    exactly the seasons they had not yet moved in: two clubs in 2016 and one in
    2017 through 2019, silently, on an inner join.

    ``team_id`` is the franchise identity and survives a move (``OAK`` and
    ``LV`` are both 2520), so it is the bridge. The schedule's own codes are
    treated as canonical, because the opponent map, the weather join and the
    rest of this package all key off the schedule.
    """
    import nflreadpy as nfl

    seasons = sorted({int(s) for s in seasons})
    teams = nfl.load_teams().to_pandas()
    if not {"team_abbr", "team_id"}.issubset(teams.columns):
        raise ValueError("nflverse team table lacks team_abbr/team_id")
    teams = teams.dropna(subset=["team_abbr", "team_id"])
    by_abbr = dict(zip(teams["team_abbr"].astype(str), teams["team_id"].astype(str)))

    schedule = ingest.load_schedules(seasons)
    if "game_type" in schedule.columns:
        schedule = schedule[schedule["game_type"] == "REG"]

    mapping: dict[tuple[int, str], str] = {}
    for season, block in schedule.groupby("season"):
        season = int(season)
        codes = set(block["home_team"].astype(str)) | set(block["away_team"].astype(str))
        # Franchise id -> the code this season actually used.
        canonical = {by_abbr.get(code): code for code in codes if by_abbr.get(code)}
        for code, franchise in by_abbr.items():
            target = canonical.get(franchise)
            if target is not None:
                mapping[(season, code)] = target
    return mapping


def build_defense_panel(seasons: Iterable[int]) -> pd.DataFrame:
    """One row per team-week. The 'player' is the club.

    There is no roster question here and no honest zero to construct: every team
    that played fielded a defense, and its fantasy week exists whether or not it
    did anything memorable.
    """
    seasons = sorted({int(s) for s in seasons})
    stats = ingest.load_team_stats(seasons, summary_level="week")
    if stats.empty:
        return pd.DataFrame()
    if "season_type" in stats.columns:
        stats = stats[stats["season_type"] == "REG"]
    stats = stats.copy()
    for column in ("season", "week"):
        stats[column] = pd.to_numeric(stats[column], errors="coerce").astype("Int64")
    stats["team"] = stats["team"].astype(str)

    # Relabel the stats feed onto the schedule's codes before merging, or
    # every relocated franchise silently disappears from the seasons before it
    # moved. See `_canonical_team_codes`.
    codes = _canonical_team_codes(seasons)
    if codes:
        keys = list(zip(stats["season"].astype(int), stats["team"].astype(str)))
        stats["team"] = [codes.get(key, key[1]) for key in keys]

    scores = _team_scores(seasons)
    panel = stats.merge(scores, on=["season", "week", "team"], how="inner")

    # Yards allowed is the opponent's offensive yardage, so it is this frame
    # read from the other side rather than a column of its own.
    gained = panel.reindex(columns=["season", "week", "team"]).copy()
    for column in ("passing_yards", "rushing_yards"):
        gained[column] = pd.to_numeric(panel.get(column), errors="coerce").fillna(0.0)
    gained["yards_for"] = gained.pop("passing_yards") + gained.pop("rushing_yards")
    opponents = _opponent_map(seasons)
    panel = panel.merge(opponents, on=["season", "week", "team"], how="left")
    panel = panel.merge(
        gained.rename(columns={"team": "opponent", "yards_for": "yards_allowed"}),
        on=["season", "week", "opponent"],
        how="left",
    )

    panel["points"] = defense_points(panel).astype(float)
    for column in DEFENSE_STATS:
        if column not in panel.columns:
            panel[column] = 0.0
        panel[column] = pd.to_numeric(panel[column], errors="coerce")
    panel["position"] = "DST"
    panel["player_id"] = panel["team"]
    panel["player_key"] = panel["team"]
    panel["player_name"] = panel["team"] + " D/ST"
    panel["played"] = 1

    keep = [
        "season",
        "week",
        "team",
        "opponent",
        "player_id",
        "player_key",
        "player_name",
        "position",
        "points",
        "played",
        *DEFENSE_STATS,
    ]
    return (
        panel.reindex(columns=keep)
        .dropna(subset=["points"])
        .sort_values(["player_key", "season", "week"], kind="mergesort")
        .reset_index(drop=True)
    )


# Half-life on the *league* baseline, in team-weeks. Long, because this column
# is meant to track an era rather than a hot streak: roughly a season and a
# half, so a rule change is fully absorbed within one season of it and a single
# windy Sunday is not mistaken for one.
LEAGUE_HALFLIFE = 24.0
LEAGUE_ALPHA = 1.0 - 0.5 ** (1.0 / LEAGUE_HALFLIFE)


def add_league_baseline(panel: pd.DataFrame, column: str = "points") -> pd.DataFrame:
    """Attach the lagged league-wide mean, so the fit can see the era it is in.

    Every other feature in this module is about one specialist. This one is
    about the league, and it exists because kicker scoring is not stationary.

    The 2024 dynamic kickoff and the 2025 touchback spot moved average starting
    field position forward, and the measured consequence is that drives reach
    field-goal range more often: attempts per played game rose 1.93 to 2.03
    between 2016-2023 and 2024-2025 while accuracy (0.845 to 0.848) and extra
    points (2.32 to 2.30) did not move. More chances, not better kicking. Points
    per played game went 7.53 to 8.12.

    A model whose level features are a within-player history cannot see that.
    Its career mean stays anchored in the old era, and the walk-forward showed
    exactly the resulting signature: the kicker projection under-shot by 0.12
    points in the last third of 2023 and by 0.91 and 1.08 in 2024 and 2025.

    The column is a weighted mean over **prior** team-weeks, pooled across every
    specialist and carried across the season boundary, so week 1 of a season
    inherits the level the previous season ended at. It is lagged like
    everything else here: the week being predicted is never in its own baseline.

    **It was measured and it does not work.** As a rung it costs CRPS (2.7266 to
    2.7308) and nearly doubles the under-projection it was built to remove
    (-0.235 to -0.420), buying only a little MAE. The reason is visible in the
    column itself and is not a tuning problem:

    - It **lags the shift it exists to track**. Against the realised level it is
      off by +0.36 and +0.31 in 2024 and 2025, against a +-0.13 error in every
      season before them. A trailing average cannot lead a discontinuity.
    - The coefficient is **extrapolated, not estimated**. Across 2016-2022 this
      column has a standard deviation of 0.103; the 2024-2025 shift is 0.271,
      2.6 times the variation any coefficient on it was ever fitted from.

    The general statement is that **the size of a rule change cannot be learned
    from data that predates it**, and no lagged feature evades that. Nor does
    waiting a season: 2025 is fitted with a full new-era season in training and
    its weeks 1-4 are *worse* than 2024's (-1.76 against -1.56), because 2025
    moved the touchback spot again and is its own new era.

    What would work is an external prior on the size of the effect, which is the
    kind of thing ``data/manual/`` exists for in this package. The function is
    kept because the rung is the evidence for that conclusion, and because the
    baseline itself is the right diagnostic to look at when scoring drifts.
    """
    frame = panel.sort_values(["season", "week"], kind="mergesort").reset_index(drop=True)
    played = pd.to_numeric(frame.get("played", 1), errors="coerce").fillna(1.0)
    values = pd.to_numeric(frame[column], errors="coerce").where(played == 1.0)

    weekly = (
        pd.DataFrame({"season": frame["season"], "week": frame["week"], "value": values})
        .groupby(["season", "week"], as_index=False)["value"]
        .mean()
        .sort_values(["season", "week"], kind="mergesort")
        .reset_index(drop=True)
    )
    weekly["_key"] = 0
    weekly["league_points_recent"] = _prior(
        weekly, ["_key"], weekly["value"].astype(float), how="ewm", alpha=LEAGUE_ALPHA
    )
    return frame.merge(
        weekly[["season", "week", "league_points_recent"]],
        on=["season", "week"],
        how="left",
    )


def _history(panel: pd.DataFrame, columns: Iterable[str], suffix: str = "_recent"):
    """Lagged, recency-weighted history of each column, per specialist."""
    out = {}
    for column in columns:
        values = pd.to_numeric(panel[column], errors="coerce").astype(float)
        out[f"prior_{column}{suffix}"] = _prior(
            panel, ["player_key"], values, how="ewm", alpha=SPECIALIST_ALPHA
        )
    return out


def add_kicker_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Feature layer for the kicker panel."""
    frame = panel.sort_values(
        ["player_key", "season", "week"], kind="mergesort"
    ).reset_index(drop=True)

    points = pd.to_numeric(frame["points"], errors="coerce").astype(float)
    frame["prior_points_mean"] = _prior(frame, ["player_key"], points, how="mean")
    frame["prior_points_recent"] = _prior(
        frame, ["player_key"], points, how="ewm", alpha=SPECIALIST_ALPHA
    )
    played = pd.to_numeric(frame["played"], errors="coerce").fillna(0).astype(float)
    frame["prior_play_rate"] = _prior(frame, ["player_key"], played, how="mean")
    frame["prior_games"] = _prior(frame, ["player_key"], played, how="sum")

    # Volume and leg, kept apart: attempts are the offence's, conversion is his,
    # and only the first of the two carries reliably to next Sunday.
    for name, series in _history(frame, KICKER_STATS).items():
        frame[name] = series
    made = pd.to_numeric(frame["fg_made"], errors="coerce").astype(float)
    attempts = pd.to_numeric(frame["fg_att"], errors="coerce").astype(float)
    frame["prior_fg_pct_recent"] = _prior(
        frame,
        ["player_key"],
        (made / attempts.where(attempts > 0)).astype(float),
        how="ewm",
        alpha=SPECIALIST_ALPHA,
    )
    # What he is worth on a week he kicks, as opposed to blended with the weeks
    # he did not. `relevant_population` reads this column, so the shared
    # walk-forward defines the same population here as on the skill panel.
    frame["prior_points_recent_given_played"] = _prior(
        frame,
        ["player_key"],
        points.where(played == 1.0),
        how="ewm",
        alpha=SPECIALIST_ALPHA,
    )
    return frame


def add_defense_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Feature layer for the team-defense panel.

    Two blocks. The defense's own recent form, and -- carried explicitly because
    it is the larger half of the response -- what the **opponent's offence** has
    been doing. The latter is a lagged read of the other club's scoring, joined
    through the opponent map, and it is the backward-looking companion to the
    implied opponent total the closing line supplies prospectively.
    """
    frame = panel.sort_values(
        ["player_key", "season", "week"], kind="mergesort"
    ).reset_index(drop=True)

    points = pd.to_numeric(frame["points"], errors="coerce").astype(float)
    frame["prior_points_mean"] = _prior(frame, ["player_key"], points, how="mean")
    frame["prior_points_recent"] = _prior(
        frame, ["player_key"], points, how="ewm", alpha=SPECIALIST_ALPHA
    )
    for name, series in _history(frame, DEFENSE_STATS).items():
        frame[name] = series

    # What this week's opponent has scored lately, lagged inside its own history
    # and then joined onto the defense facing it.
    offence = (
        frame[["season", "week", "team", "points_allowed"]]
        .drop_duplicates(subset=["season", "week", "team"])
        .sort_values(["team", "season", "week"], kind="mergesort")
        .reset_index(drop=True)
    )
    scores = _team_scores(sorted(frame["season"].unique().tolist()))
    offence = offence.drop(columns=["points_allowed"]).merge(
        scores[["season", "week", "team", "points_for"]],
        on=["season", "week", "team"],
        how="left",
    )
    offence["opp_points_for_recent"] = _prior(
        offence,
        ["team"],
        pd.to_numeric(offence["points_for"], errors="coerce").astype(float),
        how="ewm",
        alpha=TEAM_ALPHA,
    )
    frame = frame.merge(
        offence[["season", "week", "team", "opp_points_for_recent"]].rename(
            columns={"team": "opponent"}
        ),
        on=["season", "week", "opponent"],
        how="left",
    )
    # A defense plays every week its club does, so "given played" is the same
    # quantity as the plain recency level. Both names are carried because the
    # shared walk-forward reads the first and the ladder reads the second.
    frame["prior_points_recent_given_played"] = frame["prior_points_recent"]
    frame["prior_play_rate"] = 1.0
    frame["prior_games"] = _prior(
        frame,
        ["player_key"],
        pd.Series(1.0, index=frame.index),
        how="sum",
    )
    return frame


def attach_market(panel: pd.DataFrame) -> pd.DataFrame:
    """Closing spread and totals, the prospective half of both frames."""
    frame = panel.copy()
    lines = _market_lines(sorted(frame["season"].unique().tolist()))
    if lines.empty:
        for column in ("spread", "game_total", "implied_team_total", "implied_opponent_total"):
            frame[column] = np.nan
        return frame
    return frame.merge(lines, on=["season", "week", "team"], how="left")


# --------------------------------------------------------------------------
# Estimators
#
# The same protocol the skill-panel ladder uses -- ``fit(frame, target)`` then
# ``predict_samples(frame, draws, seed)`` -- so ``evaluate.walk_forward`` scores
# these without knowing they are a different position.
# --------------------------------------------------------------------------

# The league's current level, so an era shift is a feature rather than a bias.
LEAGUE_FEATURES = ("league_points_recent",)

KICKER_HISTORY_FEATURES = (
    "prior_points_mean",
    "prior_points_recent",
    "prior_fg_att_recent",
    "prior_fg_made_recent",
    "prior_pat_att_recent",
    "prior_fg_pct_recent",
    "prior_fg_long_recent",
    "prior_fg_att_50_plus_recent",
    "prior_fg_made_40_49_recent",
)

# What the offence is expected to do. For a kicker this is the volume driver:
# extra points come from touchdowns and field goals come from drives that stall,
# so the implied team total prices both, in opposite directions.
MARKET_FEATURES = (
    "spread",
    "game_total",
    "implied_team_total",
    "implied_opponent_total",
)

DEFENSE_HISTORY_FEATURES = (
    "prior_points_mean",
    "prior_points_recent",
    "prior_def_sacks_recent",
    "prior_def_interceptions_recent",
    "prior_fumble_recovery_opp_recent",
    "prior_def_tds_recent",
    "prior_points_allowed_recent",
    "prior_yards_allowed_recent",
)

# The other half of a DST week, and the larger one.
DEFENSE_OPPONENT_FEATURES = ("opp_points_for_recent",)

SPECIALIST_WEATHER_FEATURES = (
    "roof_indoor",
    "wx_temp",
    "wx_wind",
    "wx_wind_high",
    "wx_freezing",
    "wx_missing",
)

# Roof alone. Known when the schedule is published, so a gain here is shippable
# as it stands; a gain that needs `wx_temp`/`wx_wind` needs a forecast behind it.
# Splitting the two is the only way to tell which kind of gain a weather rung
# has actually found.
SPECIALIST_ROOF_FEATURES = ("roof_indoor",)

RIDGE_PENALTY = 10.0
LOGISTIC_PENALTY = 5.0


def _matrix(
    frame: pd.DataFrame, columns: tuple[str, ...], medians: pd.Series | None = None
) -> tuple[np.ndarray, pd.Series]:
    """Design matrix with training medians filled in, plus a missing flag."""
    block = frame.reindex(columns=list(columns)).apply(pd.to_numeric, errors="coerce")
    if medians is None:
        medians = block.median()
    medians = medians.fillna(0.0)
    missing = block.isna().all(axis=1).astype(float)
    filled = block.fillna(medians).fillna(0.0)
    design = np.column_stack([filled.to_numpy(float), missing.to_numpy()])
    return design, medians


@dataclass
class SpecialistClimatology:
    """The position's mean week, resampled. The floor every rung has to clear."""

    name: str = "climatology"
    pool: np.ndarray | None = None

    def fit(self, frame: pd.DataFrame, target: np.ndarray) -> "SpecialistClimatology":
        values = np.asarray(target, float)
        self.pool = values[np.isfinite(values)]
        return self

    def predict_samples(
        self, frame: pd.DataFrame, draws: int = 800, seed: int = 0
    ) -> np.ndarray:
        rng = np.random.default_rng(seed)
        picks = rng.integers(0, len(self.pool), size=(len(frame), draws))
        return self.pool[picks]


@dataclass
class SpecialistHistory:
    """One column: his own recency-weighted average. The persistence baseline."""

    column: str = "prior_points_recent"
    name: str = "recency-mean"
    fallback: float = 0.0
    residuals: LocalResiduals | None = None

    def fit(self, frame: pd.DataFrame, target: np.ndarray) -> "SpecialistHistory":
        target = np.asarray(target, float)
        fitted = self._level(frame)
        self.fallback = float(np.nanmedian(target))
        fitted = np.where(np.isfinite(fitted), fitted, self.fallback)
        self.residuals = LocalResiduals.fit(fitted, target)
        return self

    def _level(self, frame: pd.DataFrame) -> np.ndarray:
        return pd.to_numeric(frame[self.column], errors="coerce").to_numpy(float)

    def predict_samples(
        self, frame: pd.DataFrame, draws: int = 800, seed: int = 0
    ) -> np.ndarray:
        rng = np.random.default_rng(seed)
        fitted = self._level(frame)
        fitted = np.where(np.isfinite(fitted), fitted, self.fallback)
        drawn = fitted[:, None] + self.residuals.draw(fitted, draws, rng)
        return drawn


@dataclass
class SpecialistModel:
    """Hurdle for kickers, plain regression for defenses.

    ``use_hurdle`` is the whole difference. A kicker can be inactive and the
    availability half is worth fitting; a team defense plays every week its club
    does, so a hurdle there would be a logistic regression on a constant. The
    flag is set by the caller rather than inferred, so the choice is visible.
    """

    name: str
    history: tuple[str, ...]
    use_market: bool = False
    use_weather: bool = False
    use_roof: bool = False
    use_opponent: bool = False
    use_league: bool = False
    use_hurdle: bool = True
    availability: object = None
    magnitude: object = None
    residuals: LocalResiduals | None = None
    availability_medians: pd.Series | None = None
    magnitude_medians: pd.Series | None = None

    @property
    def features(self) -> tuple[str, ...]:
        return (
            self.history
            + (MARKET_FEATURES if self.use_market else ())
            + (SPECIALIST_WEATHER_FEATURES if self.use_weather else ())
            + (SPECIALIST_ROOF_FEATURES if self.use_roof else ())
            + (DEFENSE_OPPONENT_FEATURES if self.use_opponent else ())
            + (LEAGUE_FEATURES if self.use_league else ())
        )

    def fit(self, frame: pd.DataFrame, target: np.ndarray) -> "SpecialistModel":
        target = np.asarray(target, float)
        played = pd.to_numeric(frame["played"], errors="coerce").fillna(0).to_numpy(int)

        if self.use_hurdle:
            design, self.availability_medians = _matrix(frame, self.features)
            self.availability = Logistic.fit(design, played, penalty=LOGISTIC_PENALTY)

        # Magnitude is fitted on the weeks he actually played, so it estimates
        # what a week is worth rather than a blend of that and his absences.
        on_field = (played == 1) & np.isfinite(target)
        block = frame.loc[on_field]
        design, self.magnitude_medians = _matrix(block, self.features)
        self.magnitude = Ridge.fit(design, target[on_field], penalty=RIDGE_PENALTY)
        fitted = self.magnitude.predict(design)
        self.residuals = LocalResiduals.fit(fitted, target[on_field])
        return self

    def predict_samples(
        self, frame: pd.DataFrame, draws: int = 800, seed: int = 0
    ) -> np.ndarray:
        rng = np.random.default_rng(seed)
        design, _ = _matrix(frame, self.features, self.magnitude_medians)
        fitted = self.magnitude.predict(design)
        drawn = fitted[:, None] + self.residuals.draw(fitted, draws, rng)

        if self.use_hurdle:
            available, _ = _matrix(frame, self.features, self.availability_medians)
            probability = self.availability.predict_proba(available)
            plays = rng.random((len(frame), draws)) < probability[:, None]
            # A week he does not play is a zero, not a small number. The atom is
            # the point of the hurdle.
            drawn = np.where(plays, drawn, 0.0)
        return drawn


def kicker_ladder() -> list:
    """Rungs for the kicker response, in reporting order."""
    return [
        SpecialistClimatology(),
        SpecialistHistory(),
        SpecialistModel(name="kicker-history", history=KICKER_HISTORY_FEATURES),
        SpecialistModel(
            name="kicker+market", history=KICKER_HISTORY_FEATURES, use_market=True
        ),
        SpecialistModel(
            name="kicker+market+league",
            history=KICKER_HISTORY_FEATURES,
            use_market=True,
            use_league=True,
        ),
        SpecialistModel(
            name="kicker+market+roof",
            history=KICKER_HISTORY_FEATURES,
            use_market=True,
            use_roof=True,
        ),
        SpecialistModel(
            name="kicker+market+weather",
            history=KICKER_HISTORY_FEATURES,
            use_market=True,
            use_weather=True,
        ),
    ]


def defense_ladder() -> list:
    """Rungs for the team-defense response, in reporting order."""
    return [
        SpecialistClimatology(),
        SpecialistHistory(),
        SpecialistModel(
            name="defense-history",
            history=DEFENSE_HISTORY_FEATURES,
            use_hurdle=False,
        ),
        SpecialistModel(
            name="defense+opponent",
            history=DEFENSE_HISTORY_FEATURES,
            use_opponent=True,
            use_hurdle=False,
        ),
        SpecialistModel(
            name="defense+opponent+market",
            history=DEFENSE_HISTORY_FEATURES,
            use_opponent=True,
            use_market=True,
            use_hurdle=False,
        ),
        SpecialistModel(
            name="defense+opponent+market+weather",
            history=DEFENSE_HISTORY_FEATURES,
            use_opponent=True,
            use_market=True,
            use_weather=True,
            use_hurdle=False,
        ),
    ]


# --------------------------------------------------------------------------
# Model 2: rest of season
#
# The same two constructions the skill panel uses, and for the same reason. A
# total over the remaining games is not a week's variance times the number of
# games: most of what is unknown about a kicker's rest of season is unknown
# about *him* -- whether he keeps the job, whether the offence keeps stalling in
# field-goal range -- and that does not average out over ten games because it is
# the same unknown in each. Drawing the player once per draw and then playing his
# games against that draw is what puts the correlation back.
#
# Game script is deliberately absent from both. A closing line is published for
# one game; this response spans up to seventeen, and this week's spread says
# nothing about week twelve's. Only the season-long part of the context travels,
# which for these two panels means the history block and nothing else.
# --------------------------------------------------------------------------


@dataclass
class SpecialistDirectTotal:
    """Ridge on the remaining total, with the games-remaining offset. The control."""

    name: str
    history: tuple[str, ...]
    model: Ridge | None = None
    residuals: LocalResiduals | None = None
    medians: pd.Series | None = None

    def _design(self, frame: pd.DataFrame) -> np.ndarray:
        design, medians = _matrix(frame, self.history, self.medians)
        if self.medians is None:
            self.medians = medians
        offset = pd.to_numeric(frame[OFFSET], errors="coerce").fillna(1.0).to_numpy(float)
        level = (
            pd.to_numeric(frame["prior_points_recent"], errors="coerce")
            .fillna(0.0)
            .to_numpy(float)
        )
        # The offset enters directly and interacted with the per-game level, so
        # "points per game times games left" is expressible rather than having to
        # be approximated additively.
        return np.column_stack([design, offset, offset * level])

    def fit(self, frame: pd.DataFrame, target: np.ndarray) -> "SpecialistDirectTotal":
        self.medians = None
        design = self._design(frame)
        target = np.asarray(target, float)
        keep = np.isfinite(target)
        self.model = Ridge.fit(design[keep], target[keep], penalty=RIDGE_PENALTY)
        fitted = self.model.predict(design[keep])
        self.residuals = LocalResiduals.fit(fitted, target[keep])
        return self

    def predict_samples(
        self, frame: pd.DataFrame, draws: int = 800, seed: int = 0
    ) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("fit before predicting")
        rng = np.random.default_rng(seed)
        fitted = self.model.predict(self._design(frame))
        drawn = fitted[:, None] + self.residuals.draw(fitted, draws, rng)
        return np.maximum(drawn, 0.0)


@dataclass
class SpecialistSeason:
    """Draw the specialist once, then play out his remaining games against him.

    Two latents per draw, both held fixed across the games inside it:

    ``pi``
        A latent play rate from a Beta whose mean is the fitted availability and
        whose concentration is estimated from how much realized play counts
        over-disperse relative to Binomial. Switched off for defenses, which play
        every week their club does.

    ``lambda``
        A latent per-game level around the fitted magnitude, with a standard
        deviation estimated from the covariance between two different weeks of
        the same specialist -- the part of the residual that belongs to him
        rather than to either week.

    ``persistent=False`` removes both and draws the weeks independently, which is
    the construction whose intervals are too narrow. It is kept as the ablation
    that demonstrates the point rather than as a candidate.
    """

    name: str
    history: tuple[str, ...]
    use_hurdle: bool = True
    persistent: bool = True
    availability: object = None
    magnitude: object = None
    residuals: LocalResiduals | None = None
    availability_medians: pd.Series | None = None
    magnitude_medians: pd.Series | None = None
    concentration: float = 40.0
    level_sd: float = 0.0

    def fit(self, frame: pd.DataFrame, target: np.ndarray) -> "SpecialistSeason":
        played = pd.to_numeric(frame["played"], errors="coerce").fillna(0).to_numpy(int)
        points = pd.to_numeric(frame["points"], errors="coerce").to_numpy(float)

        if self.use_hurdle:
            design, self.availability_medians = _matrix(frame, self.history)
            self.availability = Logistic.fit(design, played, penalty=LOGISTIC_PENALTY)
            probability = self.availability.predict_proba(design)
            self.concentration = _beta_concentration(
                played, probability, frame["player_key"].to_numpy()
            )

        on_field = (played == 1) & np.isfinite(points)
        block = frame.loc[on_field]
        design, self.magnitude_medians = _matrix(block, self.history)
        self.magnitude = Ridge.fit(design, points[on_field], penalty=RIDGE_PENALTY)
        fitted = self.magnitude.predict(design)
        self.residuals = LocalResiduals.fit(fitted, points[on_field])
        self.level_sd = _persistent_sd(
            points[on_field] - fitted, block["player_key"].to_numpy()
        )
        return self

    def predict_samples(
        self, frame: pd.DataFrame, draws: int = 800, seed: int = 0
    ) -> np.ndarray:
        if self.magnitude is None:
            raise RuntimeError("fit before predicting")
        rng = np.random.default_rng(seed)
        rows = len(frame)
        games = (
            pd.to_numeric(frame[OFFSET], errors="coerce").fillna(1.0).to_numpy(float)
        )

        design, _ = _matrix(frame, self.history, self.magnitude_medians)
        level = self.magnitude.predict(design)

        if self.use_hurdle:
            available, _ = _matrix(frame, self.history, self.availability_medians)
            probability = np.clip(self.availability.predict_proba(available), 1e-4, 1 - 1e-4)
        else:
            probability = np.ones(rows)

        # One draw of the specialist himself, shared by every game in that draw.
        if self.persistent and self.use_hurdle and self.concentration > 0:
            alpha = probability[:, None] * self.concentration
            beta = (1.0 - probability[:, None]) * self.concentration
            play_rate = rng.beta(alpha, beta, size=(rows, draws))
        else:
            play_rate = np.repeat(probability[:, None], draws, axis=1)

        if self.persistent and self.level_sd > 0:
            drawn_level = level[:, None] + rng.normal(
                0.0, self.level_sd, size=(rows, draws)
            )
        else:
            drawn_level = np.repeat(level[:, None], draws, axis=1)

        # Play the remaining games against the drawn specialist. Each game is
        # played or not on its own Bernoulli draw of the latent rate, and a game
        # he plays gets the drawn level plus that week's own noise.
        #
        # The games are accumulated one at a time rather than simulated at full
        # participation and scaled by the played fraction. Scaling would average
        # the noise over every remaining game before shrinking it, which makes
        # the total's variance too small by exactly the amount the missed weeks
        # should have removed -- the error this construction exists to avoid.
        out = np.zeros((rows, draws), dtype=float)
        maximum = int(np.nanmax(games)) if rows else 0
        for count in range(1, maximum + 1):
            want = games == count
            if not want.any():
                continue
            block_level = drawn_level[want]
            # Bucket the residual pool by the row's own drawn level, once per
            # block rather than once per game: the pool is indexed by fitted
            # value and a specialist's level does not move between his games
            # inside a single draw.
            centre = block_level.mean(axis=1)
            total = np.zeros_like(block_level)
            for _ in range(count):
                noise = self.residuals.draw(centre, draws, rng)
                # No floor at zero. A specialist's week genuinely can be
                # negative -- a kicker who misses a field goal and an extra
                # point scores -2, a defense that concedes 35 with no takeaways
                # scores -4 -- and the residual pool is empirical, so
                # ``level + noise`` already reproduces that tail. Clipping it
                # away would delete real mass from the low side of every game
                # and push the sum of ten of them several points high, which is
                # a bias that grows with the horizon.
                week = block_level + noise
                played_this_week = rng.random(week.shape) < play_rate[want]
                total += np.where(played_this_week, week, 0.0)
            out[want] = total
        return out


def kicker_season_ladder() -> list:
    """Rest-of-season rungs for kickers."""
    return [
        SpecialistDirectTotal(name="direct-total", history=KICKER_HISTORY_FEATURES),
        SpecialistSeason(
            name="season-independent",
            history=KICKER_HISTORY_FEATURES,
            persistent=False,
        ),
        SpecialistSeason(name="season-hierarchical", history=KICKER_HISTORY_FEATURES),
    ]


def defense_season_ladder() -> list:
    """Rest-of-season rungs for team defenses.

    The opponent column travels here in a way the closing line does not: it is a
    lagged read of clubs' scoring rather than a forecast of one game, so it still
    means something ten weeks out. It is nonetheless a weaker feature over a long
    horizon than over one week, since a defense's remaining schedule averages
    over many opponents rather than facing the one the column describes.
    """
    history = DEFENSE_HISTORY_FEATURES + DEFENSE_OPPONENT_FEATURES
    return [
        SpecialistDirectTotal(name="direct-total", history=history),
        SpecialistSeason(
            name="season-independent",
            history=history,
            use_hurdle=False,
            persistent=False,
        ),
        SpecialistSeason(
            name="season-hierarchical", history=history, use_hurdle=False
        ),
    ]
