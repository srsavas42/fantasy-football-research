"""Leakage-safe injury history and expected recovery features.

The season-average model projects from a Week-1 / current roster snapshot.
Historical nflverse injury reports identify injury occurrence, official game
status, practice participation, and body area.  A recovery episode is labeled
by the number of later regular-season roster weeks unavailable before a return
to ``ACT`` status.  That is deliberately an expected *availability* duration,
not a medical prognosis.

For a projected season ``Y``, all prior-burden features and empirical recovery
tables use reports and outcomes from seasons strictly before ``Y``.  A current
injury snapshot can be supplied for a live projection; historical builds use
the official Week-1 report, aligned with the existing Week-1 roster cutoff.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ffmodel.data.wikipedia_coaching import team_identity
from ffmodel.features import crossseason
from ffmodel.features.volume import MODEL_POSITIONS, opportunity_position


INJURY_AVAILABILITY_FEATURES = (
    "injury_history_available",
    "prior_injury_report_weeks_3yr",
    "prior_injury_out_weeks_3yr",
    "prior_injury_episode_count_3yr",
    "prior_injury_mean_recovery_weeks_3yr",
    "prior_injury_weeks_since_last",
    "current_injury_snapshot_available",
    "current_injury_reported",
    "current_injury_severity",
    "current_injury_practice_severity",
    "current_injury_expected_recovery_weeks",
)

_RECOVERY_GROUP_STRENGTH = 12.0
_PLAYER_RECOVERY_STRENGTH = 3.0


def _empty_reports() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "season",
            "team",
            "week",
            "player_key",
            "position",
            "injury_reported",
            "injury_severity",
            "injury_practice_severity",
            "injury_body_group",
        ]
    )


def load_live_injury_snapshot(
    season: int,
    *,
    cutoff_week: int = 1,
    snapshot_at=None,
    refresh: bool = False,
    cache_dir=None,
) -> pd.DataFrame:
    """Archive and label a current Sleeper injury snapshot for one projection.

    Sleeper exposes present-day player status rather than historical states. Its
    loader therefore refuses to fabricate a past snapshot on a cache miss; this
    helper keeps that safeguard and only annotates the archived data with the
    projected season and cutoff week used by the feature builder.
    """
    from ffmodel.data import sleeper

    snapshot = sleeper.load_players(
        snapshot_at=snapshot_at,
        refresh=refresh,
        cache_dir=cache_dir,
    ).copy()
    snapshot["season"] = int(season)
    snapshot["week"] = int(cutoff_week)
    return snapshot


def _input_report_seasons(injuries: pd.DataFrame | None) -> set[int]:
    """Seasons for which an injury-feed artifact was supplied, not its events."""
    if injuries is None or injuries.empty or "season" not in injuries:
        return set()
    values = pd.to_numeric(injuries["season"], errors="coerce").dropna()
    return set(values.astype(int).unique())


def _normalise_team(values: pd.Series, seasons: pd.Series) -> pd.Series:
    normalized = []
    for team, season in zip(values, seasons):
        code = str(team).upper().strip()
        if code in {"", "NAN", "NONE", "<NA>"}:
            normalized.append("UNK")
            continue
        try:
            normalized.append(team_identity(code, int(season)).franchise_code)
        except KeyError:
            # Live player directories also contain free agents and stale team
            # values. They cannot form a historical recovery episode, but their
            # player-level current-injury signal is still valid.
            normalized.append(code)
    return pd.Series(normalized, index=values.index, dtype="object")


def _status_severity(value: object) -> float:
    text = str(value).strip().upper()
    if text in {"OUT", "IR", "PUP", "NFI", "COV"} or "INJURED RESERVE" in text:
        return 3.0
    if text == "DOUBTFUL":
        return 2.0
    if text == "QUESTIONABLE":
        return 1.0
    return 0.0


def _practice_severity(value: object) -> float:
    text = str(value).strip().upper()
    if "DID NOT" in text or text == "DNP":
        return 2.0
    if "LIMITED" in text:
        return 1.0
    return 0.0


def _injury_body_group(value: object) -> str:
    text = str(value).strip().lower()
    if not text or text in {"nan", "none", "<na>"}:
        return "unknown"
    if any(token in text for token in ("not injury", "personal", "resting")):
        return "non_injury"
    if any(
        token in text
        for token in (
            "ankle",
            "achilles",
            "calf",
            "foot",
            "toe",
            "knee",
            "hamstring",
            "groin",
            "hip",
            "quad",
            "thigh",
            "leg",
        )
    ):
        return "lower_body"
    if any(
        token in text
        for token in (
            "shoulder",
            "elbow",
            "wrist",
            "hand",
            "finger",
            "arm",
            "bicep",
            "tricep",
            "pectoral",
            "chest",
        )
    ):
        return "upper_body"
    if any(token in text for token in ("concussion", "head", "neck", "eye", "jaw")):
        return "head_neck"
    if any(token in text for token in ("back", "abdomen", "oblique", "rib", "core")):
        return "core"
    if any(token in text for token in ("illness", "ill", "covid", "migraine")):
        return "illness"
    return "other"


def normalise_injury_reports(injuries: pd.DataFrame | None) -> pd.DataFrame:
    """Return one normalized injury state per player/team/season/week.

    The function accepts both nflverse historical report columns and the
    corresponding current Sleeper fields.  Fields that cannot identify a
    player are discarded rather than joined by a fuzzy player name.
    """
    if injuries is None or injuries.empty:
        return _empty_reports()
    out = injuries.copy()
    if "season" not in out:
        out["season"] = np.nan
    if "week" not in out:
        out["week"] = np.nan
    if "team" not in out:
        out["team"] = np.nan
    if "gsis_id" in out:
        out["player_id"] = out["gsis_id"]
    elif "player_id" not in out:
        out["player_id"] = pd.NA
    if "full_name" in out and "player_name" not in out:
        out["player_name"] = out["full_name"]
    if "player_name" not in out:
        out["player_name"] = pd.NA
    if "position" not in out:
        out["position"] = pd.NA
    out["season"] = pd.to_numeric(out["season"], errors="coerce")
    out["week"] = pd.to_numeric(out["week"], errors="coerce")
    out = out[out["season"].notna() & out["week"].notna()].copy()
    if "game_type" in out:
        game_type = out["game_type"].astype("string").str.upper().fillna("REG")
        out = out[game_type.eq("REG")].copy()
    if out.empty:
        return _empty_reports()
    out["season"] = out["season"].astype(int)
    out["week"] = out["week"].astype(int)
    out["position"] = opportunity_position(out["position"]).astype(str)
    out["player_key"] = crossseason.player_key(out)
    valid_id = out.get("player_id", pd.Series(pd.NA, index=out.index)).notna()
    out = out[valid_id & out["position"].isin(MODEL_POSITIONS)].copy()
    if out.empty:
        return _empty_reports()
    out["team"] = _normalise_team(out["team"].astype(str), out["season"])

    report_status = out.get(
        "report_status", out.get("injury_status", pd.Series(pd.NA, index=out.index))
    )
    practice_status = out.get(
        "practice_status",
        out.get("practice_participation", pd.Series(pd.NA, index=out.index)),
    )
    body = out.get(
        "report_primary_injury",
        out.get("injury_body_part", pd.Series(pd.NA, index=out.index)),
    )
    practice_body = out.get(
        "practice_primary_injury", pd.Series(pd.NA, index=out.index)
    )
    body = pd.Series(body, index=out.index).combine_first(
        pd.Series(practice_body, index=out.index)
    )
    out["injury_severity"] = pd.Series(report_status, index=out.index).map(
        _status_severity
    )
    out["injury_practice_severity"] = pd.Series(practice_status, index=out.index).map(
        _practice_severity
    )
    out["injury_body_group"] = pd.Series(body, index=out.index).map(
        _injury_body_group
    )
    out["injury_reported"] = (
        out["injury_severity"].gt(0)
        | out["injury_practice_severity"].gt(0)
    ) & out["injury_body_group"].ne("non_injury")
    out = out[out["injury_reported"]].copy()
    if out.empty:
        return _empty_reports()

    keys = ["season", "team", "week", "player_key"]
    out = out.sort_values(
        keys + ["injury_severity", "injury_practice_severity"],
        ascending=[True, True, True, True, False, False],
    ).drop_duplicates(keys)
    return out[
        [
            "season",
            "team",
            "week",
            "player_key",
            "position",
            "injury_reported",
            "injury_severity",
            "injury_practice_severity",
            "injury_body_group",
        ]
    ].reset_index(drop=True)


def _normalise_roster_availability(weekly_rosters: pd.DataFrame | None) -> pd.DataFrame:
    if weekly_rosters is None or weekly_rosters.empty:
        return pd.DataFrame(
            columns=["season", "team", "week", "player_key", "roster_active"]
        )
    out = weekly_rosters.copy()
    required = {"season", "team", "week"}
    if not required <= set(out.columns):
        return pd.DataFrame(
            columns=["season", "team", "week", "player_key", "roster_active"]
        )
    if "gsis_id" in out:
        out["player_id"] = out["gsis_id"]
    elif "player_id" not in out:
        return pd.DataFrame(
            columns=["season", "team", "week", "player_key", "roster_active"]
        )
    if "full_name" in out and "player_name" not in out:
        out["player_name"] = out["full_name"]
    if "player_name" not in out:
        out["player_name"] = pd.NA
    if "position" not in out:
        out["position"] = pd.NA
    out["season"] = pd.to_numeric(out["season"], errors="coerce")
    out["week"] = pd.to_numeric(out["week"], errors="coerce")
    out = out[out["season"].notna() & out["week"].notna()].copy()
    if "game_type" in out:
        game_type = out["game_type"].astype("string").str.upper().fillna("REG")
        out = out[game_type.eq("REG")].copy()
    if out.empty:
        return pd.DataFrame(
            columns=["season", "team", "week", "player_key", "roster_active"]
        )
    out["season"] = out["season"].astype(int)
    out["week"] = out["week"].astype(int)
    out["position"] = opportunity_position(out["position"]).astype(str)
    out["player_key"] = crossseason.player_key(out)
    out = out[
        out.get("player_id", pd.Series(pd.NA, index=out.index)).notna()
        & out["position"].isin(MODEL_POSITIONS)
    ].copy()
    if out.empty:
        return pd.DataFrame(
            columns=["season", "team", "week", "player_key", "roster_active"]
        )
    out["team"] = _normalise_team(out["team"].astype(str), out["season"])
    status = out.get("status", pd.Series("", index=out.index)).astype(str).str.upper()
    out["roster_active"] = status.eq("ACT").astype(int)
    keys = ["season", "team", "week", "player_key"]
    return (
        out.groupby(keys, dropna=False)["roster_active"]
        .max()
        .reset_index()
        .sort_values(keys)
        .reset_index(drop=True)
    )


def build_injury_episodes(
    injuries: pd.DataFrame | None,
    weekly_rosters: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build reported injury episodes and their observed availability duration."""
    reports = normalise_injury_reports(injuries)
    if reports.empty:
        return pd.DataFrame(
            columns=[
                "season",
                "team",
                "player_key",
                "position",
                "episode_start_week",
                "episode_end_week",
                "injury_severity",
                "injury_practice_severity",
                "injury_body_group",
                "recovery_weeks",
                "recovery_censored",
            ]
        )
    reports = reports.sort_values(
        ["season", "team", "player_key", "week"]
    ).reset_index(drop=True)
    group_keys = ["season", "team", "player_key"]
    gap = reports.groupby(group_keys, dropna=False)["week"].diff().gt(1)
    reports["_episode"] = gap.fillna(True).groupby(
        [reports[key] for key in group_keys], dropna=False
    ).cumsum().astype(int)
    episodes = (
        reports.groupby(group_keys + ["_episode"], dropna=False)
        .agg(
            position=("position", "first"),
            episode_start_week=("week", "min"),
            episode_end_week=("week", "max"),
            injury_severity=("injury_severity", "max"),
            injury_practice_severity=("injury_practice_severity", "max"),
            injury_body_group=("injury_body_group", "first"),
        )
        .reset_index()
    )
    availability = _normalise_roster_availability(weekly_rosters)
    if availability.empty:
        episodes["recovery_weeks"] = (
            episodes["episode_end_week"] - episodes["episode_start_week"] + 1
        ).astype(float)
        episodes["recovery_censored"] = 1
        return episodes.drop(columns="_episode")

    recovery_weeks: list[float] = []
    censored: list[int] = []
    for episode in episodes.itertuples(index=False):
        state = availability[
            (availability["season"] == episode.season)
            & (availability["team"] == episode.team)
            & (availability["player_key"] == episode.player_key)
            & (availability["week"] >= episode.episode_start_week)
        ].sort_values("week")
        if state.empty:
            recovery_weeks.append(float(episode.episode_end_week - episode.episode_start_week + 1))
            censored.append(1)
            continue
        return_rows = state[
            state["week"].gt(episode.episode_end_week)
            & state["roster_active"].eq(1)
        ]
        if return_rows.empty:
            window = state
            censored.append(1)
        else:
            returned_week = int(return_rows.iloc[0]["week"])
            window = state[state["week"].lt(returned_week)]
            censored.append(0)
        recovery_weeks.append(float((1 - window["roster_active"]).clip(lower=0).sum()))
    episodes["recovery_weeks"] = recovery_weeks
    episodes["recovery_censored"] = censored
    return episodes.drop(columns="_episode")


def _current_snapshot(
    rows: pd.DataFrame,
    reports: pd.DataFrame,
    *,
    injury_snapshot: pd.DataFrame | None,
    cutoff_week: int,
    source_snapshot_seasons: set[int],
) -> tuple[pd.DataFrame, set[int]]:
    seasons = sorted(pd.to_numeric(rows["season"], errors="coerce").dropna().astype(int).unique())
    if injury_snapshot is None:
        current = reports[
            reports["season"].isin(seasons) & reports["week"].eq(int(cutoff_week))
        ].copy()
        available = set(seasons).intersection(source_snapshot_seasons)
        return current, available

    supplied = injury_snapshot.copy()
    if "season" not in supplied:
        if len(seasons) != 1:
            raise ValueError(
                "injury_snapshot without a season can only be used for one projection season"
            )
        supplied["season"] = seasons[0]
    if "week" not in supplied:
        supplied["week"] = int(cutoff_week)
    available = set(
        pd.to_numeric(supplied["season"], errors="coerce").dropna().astype(int).unique()
    ).intersection(seasons)
    current = normalise_injury_reports(supplied)
    current = current[current["season"].isin(seasons)].copy()
    return current, available


def _expected_recovery(
    current: pd.DataFrame,
    episodes: pd.DataFrame,
) -> pd.DataFrame:
    """Temporally fit empirical-Bayes expected unavailable weeks for snapshots."""
    if current.empty:
        return pd.DataFrame(
            columns=["season", "player_key", "current_injury_expected_recovery_weeks"]
        )
    outputs = []
    for season, snapshot in current.groupby("season", sort=True, dropna=False):
        history = episodes[
            (episodes["season"] < int(season))
            & episodes["recovery_censored"].eq(0)
            & episodes["recovery_weeks"].notna()
        ].copy()
        global_mean = float(history["recovery_weeks"].mean()) if not history.empty else 0.0
        if history.empty:
            snapshot = snapshot.copy()
            snapshot["current_injury_expected_recovery_weeks"] = 0.0
            outputs.append(snapshot)
            continue
        grouped = (
            history.groupby(["injury_body_group", "injury_severity"], dropna=False)
            .agg(total=("recovery_weeks", "sum"), count=("recovery_weeks", "size"))
        )
        player = (
            history.groupby("player_key", dropna=False)
            .agg(total=("recovery_weeks", "sum"), count=("recovery_weeks", "size"))
        )
        expected = []
        for row in snapshot.itertuples(index=False):
            key = (row.injury_body_group, row.injury_severity)
            if key in grouped.index:
                group = grouped.loc[key]
                group_mean = float(
                    (group["total"] + _RECOVERY_GROUP_STRENGTH * global_mean)
                    / (group["count"] + _RECOVERY_GROUP_STRENGTH)
                )
            else:
                group_mean = global_mean
            if row.player_key in player.index:
                personal = player.loc[row.player_key]
                value = float(
                    (personal["total"] + _PLAYER_RECOVERY_STRENGTH * group_mean)
                    / (personal["count"] + _PLAYER_RECOVERY_STRENGTH)
                )
            else:
                value = group_mean
            expected.append(float(np.clip(value, 0.0, 18.0)))
        snapshot = snapshot.copy()
        snapshot["current_injury_expected_recovery_weeks"] = expected
        outputs.append(snapshot)
    out = pd.concat(outputs, ignore_index=True)
    return out[
        [
            "season",
            "player_key",
            "injury_reported",
            "injury_severity",
            "injury_practice_severity",
            "current_injury_expected_recovery_weeks",
        ]
    ]


def add_season_injury_features(
    rows: pd.DataFrame,
    *,
    injuries: pd.DataFrame | None = None,
    weekly_rosters: pd.DataFrame | None = None,
    injury_snapshot: pd.DataFrame | None = None,
    cutoff_week: int = 1,
    history_years: int = 3,
) -> pd.DataFrame:
    """Attach prior injury burden and current expected-recovery features.

    All returned columns are known by the projection cutoff.  Realized recovery
    durations are shifted into later-season feature rows; the current-injury
    expectation is fit using only completed earlier-season episodes.
    """
    required = {"season", "player_key"}
    missing = required - set(rows.columns)
    if missing:
        raise ValueError(f"season injury rows are missing columns: {sorted(missing)}")
    out = rows.copy().reset_index(drop=True)
    reports = normalise_injury_reports(injuries)
    episodes = build_injury_episodes(injuries, weekly_rosters)
    input_report_years = _input_report_seasons(injuries)
    numeric = [
        "prior_injury_report_weeks_3yr",
        "prior_injury_out_weeks_3yr",
        "prior_injury_episode_count_3yr",
        "prior_injury_mean_recovery_weeks_3yr",
        "prior_injury_weeks_since_last",
        "current_injury_snapshot_available",
        "current_injury_reported",
        "current_injury_severity",
        "current_injury_practice_severity",
        "current_injury_expected_recovery_weeks",
    ]
    for name in numeric:
        out[name] = 0.0
    out["injury_history_available"] = 0
    seasons = sorted(pd.to_numeric(out["season"], errors="coerce").dropna().astype(int).unique())
    for season in seasons:
        if any(year in input_report_years for year in range(season - history_years, season)):
            out.loc[out["season"].eq(season), "injury_history_available"] = 1

    current, snapshot_years = _current_snapshot(
        out,
        reports,
        injury_snapshot=injury_snapshot,
        cutoff_week=cutoff_week,
        source_snapshot_seasons=input_report_years,
    )
    for season in snapshot_years:
        out.loc[out["season"].eq(season), "current_injury_snapshot_available"] = 1.0
    expected = _expected_recovery(current, episodes)
    if not expected.empty:
        expected = expected.drop_duplicates(["season", "player_key"], keep="last").set_index(
            ["season", "player_key"]
        )
        keys = pd.MultiIndex.from_frame(out[["season", "player_key"]])
        for source, target in (
            ("injury_reported", "current_injury_reported"),
            ("injury_severity", "current_injury_severity"),
            ("injury_practice_severity", "current_injury_practice_severity"),
            ("current_injury_expected_recovery_weeks", "current_injury_expected_recovery_weeks"),
        ):
            out[target] = expected[source].reindex(keys).fillna(0.0).to_numpy(dtype=float)

    if reports.empty:
        return out

    for season in seasons:
        row_mask = out["season"].eq(season)
        history = reports[
            reports["season"].ge(season - history_years)
            & reports["season"].lt(season)
        ]
        history_episodes = episodes[
            episodes["season"].ge(season - history_years)
            & episodes["season"].lt(season)
        ]
        if history.empty:
            continue
        counts = history.groupby("player_key", dropna=False).agg(
            report_weeks=("week", "size"),
            out_weeks=("injury_severity", lambda values: int((values >= 3).sum())),
            last_season=("season", "max"),
            last_week=("week", "max"),
        )
        episode_counts = history_episodes.groupby("player_key", dropna=False).agg(
            episode_count=("episode_start_week", "size"),
        )
        completed_recovery = history_episodes[
            history_episodes["recovery_censored"].eq(0)
        ].groupby("player_key", dropna=False).agg(
            mean_recovery=("recovery_weeks", "mean"),
        )
        values = counts.join(episode_counts, how="left").join(
            completed_recovery, how="left"
        )
        values["weeks_since_last"] = (
            (season - values["last_season"]) * 18 + (1 - values["last_week"])
        ).clip(lower=0)
        keys = out.loc[row_mask, "player_key"]
        for source, target in (
            ("report_weeks", "prior_injury_report_weeks_3yr"),
            ("out_weeks", "prior_injury_out_weeks_3yr"),
            ("episode_count", "prior_injury_episode_count_3yr"),
            ("mean_recovery", "prior_injury_mean_recovery_weeks_3yr"),
            ("weeks_since_last", "prior_injury_weeks_since_last"),
        ):
            out.loc[row_mask, target] = keys.map(values[source]).fillna(0.0).to_numpy()

    return out
