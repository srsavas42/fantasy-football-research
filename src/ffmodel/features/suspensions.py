"""Suspensions and the commissioner's exempt list, read out of the roster feed.

A suspension is the one availability event that is *known before it happens*.
The injury layer forecasts a hazard; a four-game ban announced in July is not a
hazard, it is arithmetic. Modelling it as risk throws away the certainty and
biases the projection toward the mean of players who were never suspended.

Nothing new has to be downloaded to get this. The nflverse weekly roster feed
already carries the NFL's own transaction codes in ``status_description_abbr``,
which separate the *reason* a player sits inside the coarse ``status`` bucket
the pipeline currently reads. ``preseason_roster_snapshot`` keeps only that
coarse bucket, so a suspended player has until now been indistinguishable from
one on injured reserve.

The encoding changes at 2020, which is a trap worth naming: before 2020 a
suspension is the top-level status ``SUS``; from 2020 it is ``RES`` carrying
``R40``. A reader that knows only one of the two silently loses half the
history. Worse, ``SUS`` is not in :data:`~ffmodel.features.season_average.
ROSTER_STATUSES`, so pre-2020 suspended players are dropped from the snapshot
altogether rather than merely mislabelled.

Codes were verified against suspensions whose lengths are a matter of public
record, and the flagged weeks reproduce the announced bans exactly:

    Deshaun Watson    2022  weeks 1-12   11 games, conduct
    DeAndre Hopkins   2022  weeks 1-6     6 games, PED
    Alvin Kamara      2023  weeks 1-3     3 games, conduct
    Jameson Williams  2023  weeks 1-4     4 games, gambling
    Jameson Williams  2024  weeks 8-9     2 games, PED
    Rashee Rice       2025  weeks 1-6     6 games, conduct

Two kinds of ban behave differently and must not be pooled.

**Definite** bans (``R40``/``SUS``) have a length fixed when they are announced.
When that announcement precedes week 1 the games lost are known with certainty
at projection time, and :func:`preseason_suspension_games` returns them as a
count to be subtracted rather than a probability to be sampled.

**Indefinite** bans (``R30``) and the **exempt list** (``EXE``/``E02``) have no
announced end. These are genuine duration problems, and the roster feed
measures them badly: a player banned indefinitely is removed from the roster
entirely rather than parked on a reserve list, so the flagged weeks *undercount*
the ban by however long he was off the roster. Calvin Ridley's 2022 season is
the clean demonstration -- 9 flagged weeks, 9 further weeks absent, against a
ban that actually cost all 17. :func:`suspension_spells` therefore reports
``roster_absent_weeks`` beside the flagged count, and marks a spell ``censored``
when it runs to the end of the season, so neither number is mistaken for the
truth on its own.

The exempt-list sample also carries a confound that rules out fitting anything
to it as-is: 11 of the 18 skill-position spells are from 2020, when the list was
used for COVID-19 roster mechanics rather than discipline. Netting those out
leaves single figures, which is why this module publishes the empirical table
and stops there instead of shipping a fitted hazard. See
:func:`exempt_duration_table`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SKILL_POSITIONS = ("QB", "RB", "WR", "TE", "FB", "HB")

# NFL transaction codes, as they appear in ``status_description_abbr``.
#
# ``R30`` and ``E02`` are grouped apart from ``R40`` because their length is not
# known in advance; see the module docstring. ``E14`` is deliberately *not*
# here: it is the International Pathway exemption, which reads as an exempt-list
# code and is not a disciplinary action at all -- Christian Wade, David Bada and
# Junior Aho account for every occurrence.
DEFINITE_CODES = frozenset({"R40"})
INDEFINITE_CODES = frozenset({"R30"})

# The exempt list is identified by its *status*, not its reason code. Before
# 2020 the code is null on these rows, so requiring ``E02`` silently dropped
# every pre-2020 placement -- Reuben Foster's 2018 season among them -- while
# reporting a clean history.
#
# These two codes are exempt placements that are not disciplinary and must not
# be pooled with the ones that are: ``E14`` is the International Pathway
# exemption (Christian Wade, David Bada, Junior Aho), ``E18`` the 2020
# COVID-19 exemption.
EXEMPT_CODES = frozenset({"E02"})
EXEMPT_NONDISCIPLINARY_CODES = frozenset({"E14", "E18"})

# Pre-2020 encoding: the ban is the top-level status, with no reason code.
LEGACY_DEFINITE_STATUS = "SUS"
EXEMPT_STATUS = "EXE"

SUSPENSION_KINDS = ("definite", "indefinite", "exempt")


def _regular_season(rosters: pd.DataFrame, *, max_week: int = 18) -> pd.DataFrame:
    out = rosters
    if "game_type" in out.columns:
        out = out[out["game_type"].isna() | out["game_type"].eq("REG")]
    out = out.copy()
    out["week"] = pd.to_numeric(out["week"], errors="coerce")
    return out[out["week"].between(1, max_week)]


def classify_suspension(rosters: pd.DataFrame) -> pd.Series:
    """Label each roster row ``definite``, ``indefinite``, ``exempt`` or NA.

    Handles both encodings: the pre-2020 ``SUS`` status and the ``R40``/``R30``
    reason codes that replaced it. A frame missing ``status_description_abbr``
    still resolves the legacy status rather than raising, because the older
    nflverse snapshots do not carry the column.
    """
    status = rosters.get("status", pd.Series(pd.NA, index=rosters.index))
    status = status.astype("string").str.upper()
    if "status_description_abbr" in rosters.columns:
        code = rosters["status_description_abbr"].astype("string").str.upper()
    else:
        code = pd.Series(pd.NA, index=rosters.index, dtype="string")

    out = pd.Series(pd.NA, index=rosters.index, dtype="string")
    out = out.mask(
        status.eq(EXEMPT_STATUS) & ~code.isin(EXEMPT_NONDISCIPLINARY_CODES),
        "exempt",
    )
    out = out.mask(code.isin(INDEFINITE_CODES), "indefinite")
    out = out.mask(code.isin(DEFINITE_CODES), "definite")
    # Legacy rows carry no reason code, so this may only fire where the codes
    # above did not.
    out = out.mask(out.isna() & status.eq(LEGACY_DEFINITE_STATUS), "definite")
    return out


def suspension_spells(
    rosters: pd.DataFrame,
    *,
    positions: tuple[str, ...] | None = SKILL_POSITIONS,
    max_week: int = 18,
) -> pd.DataFrame:
    """Per player-season suspension spells with their censoring exposed.

    ``flagged_weeks`` counts weeks the player sat on a suspension code.
    ``roster_absent_weeks`` counts regular-season weeks he appears nowhere in
    the feed, which for an indefinite ban is part of the same absence and for a
    definite one is normal roster churn. ``censored`` marks a spell still
    running at ``max_week``. A caller that adds the two columns for a definite
    ban is double-counting; a caller that ignores ``roster_absent_weeks`` for an
    indefinite one is undercounting. They are reported separately because no
    single number is right for both.
    """
    rows = _regular_season(rosters, max_week=max_week)
    if positions is not None and "position" in rows.columns:
        rows = rows[rows["position"].astype("string").str.upper().isin(positions)]
    rows = rows.copy()
    rows["susp_kind"] = classify_suspension(rows)

    name = "full_name" if "full_name" in rows.columns else "player_name"
    keys = ["season", name]
    weeks_on_roster = rows.groupby(keys, dropna=False)["week"].nunique()

    flagged = rows[rows["susp_kind"].notna()]
    if flagged.empty:
        return pd.DataFrame(
            columns=[
                "season", "player_name", "team", "susp_kind", "flagged_weeks",
                "first_week", "last_week", "roster_absent_weeks",
                "preseason_known", "censored",
            ]
        )

    spells = (
        flagged.groupby(keys + ["team", "susp_kind"], dropna=False)
        .agg(
            flagged_weeks=("week", "nunique"),
            first_week=("week", "min"),
            last_week=("week", "max"),
        )
        .reset_index()
        .rename(columns={name: "player_name"})
    )
    absent = (max_week - spells.set_index(keys[0:1] + ["player_name"]).index.map(
        lambda k: weeks_on_roster.get(k, 0)
    )).astype(int)
    spells["roster_absent_weeks"] = np.maximum(absent, 0)
    spells["preseason_known"] = spells["first_week"].eq(1)
    spells["censored"] = spells["last_week"].ge(max_week)
    return spells.sort_values(["season", "player_name"]).reset_index(drop=True)


def preseason_suspension_games(
    rosters: pd.DataFrame,
    *,
    cutoff_week: int = 1,
    positions: tuple[str, ...] | None = SKILL_POSITIONS,
    max_week: int = 18,
) -> pd.DataFrame:
    """Games a definite ban costs, for bans already in force at ``cutoff_week``.

    Restricted to definite bans on purpose. An indefinite ban or an exempt-list
    placement in force at week 1 has no known length, so there is no honest
    count to return for it and it is left to the availability model's ordinary
    risk machinery.

    This is leakage-safe only in the sense that the *placement* is observable at
    the cutoff. The length is read from the whole season, which is exactly right
    at serve time -- an announced ban's length is public the day it is handed
    down -- and is a backfill for historical rows, where the announced length is
    not separately recorded in the feed. A walk-forward arm that wants to score
    this feature must treat the length as known at the cutoff, which is what a
    drafter in August actually knew.
    """
    spells = suspension_spells(rosters, positions=positions, max_week=max_week)
    if spells.empty:
        return pd.DataFrame(columns=["season", "player_name", "team", "suspended_games"])
    known = spells[
        spells["susp_kind"].eq("definite")
        & spells["first_week"].le(cutoff_week)
    ]
    return (
        known.groupby(["season", "player_name", "team"], as_index=False)["flagged_weeks"]
        .max()
        .rename(columns={"flagged_weeks": "suspended_games"})
        .reset_index(drop=True)
    )


def exempt_duration_table(
    rosters: pd.DataFrame,
    *,
    positions: tuple[str, ...] | None = SKILL_POSITIONS,
    max_week: int = 18,
    drop_covid_season: bool = True,
) -> pd.DataFrame:
    """Observed exempt-list and indefinite-ban spell lengths.

    The empirical record, not a fitted model. ``drop_covid_season`` removes
    2020, when the exempt list was used for COVID-19 roster mechanics rather
    than discipline and which otherwise supplies the majority of the spells.
    What remains is a single-figure sample with a censored tail, which is too
    little to fit a hazard to and enough to bound one: see the module docstring.
    """
    spells = suspension_spells(rosters, positions=positions, max_week=max_week)
    if spells.empty:
        return spells
    out = spells[spells["susp_kind"].isin(("exempt", "indefinite"))]
    if drop_covid_season:
        out = out[out["season"].ne(2020)]
    return out.reset_index(drop=True)
