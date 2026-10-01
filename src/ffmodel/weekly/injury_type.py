"""What kind of injury is on the report, grouped by mechanism.

The injury feed names a **body part** and nothing else. Across 87,686 text entries
2016-2025 there is not one tear, strain, sprain, pull, soreness, contusion,
fracture or surgery -- so the "nature" of an injury, which is what would say how
long it keeps a player out, is simply not in the data this repo can train on.
Sleeper carries richer notes but only as a present-day snapshot, with no history
to learn an effect from.

What the body part does support is a coarse guess at the mechanism: a hamstring
is a muscle, a knee is a joint, a rib is a bone or a bruise. That is a proxy for
nature, not the thing itself, and it is deliberately coarse -- eighty-odd
free-text strings ("right shoulder", "rib"/"ribs", "quadricep") collapse to ten
groups, because most body parts have too few rows to say anything about on their
own.

Measured walk-forward on 2023-2025 (``docs/ros-reconciliation-2026-10.md``), the
mechanism groups are no better than body part, and both add about one point of MAE
on players who missed a game with an injury designation -- a small gain that sits
inside what three holdout seasons can resolve. They are in the shipped model
because the variant that carries them measured best, not because the type has been
shown to matter; it is also the natural place to add real severity information if
a source ever supplies it.

**Source.** The game-status field is filled on 48% of report rows, the practice
field on 99.5%, so the type is read from the game-status text and falls back to the
practice text. That doubles the rows that carry a type (8% to 17% of all player
weeks) and is the one place the type visibly moved a bias. The practice report is
filed earlier in the week than the game status, which is what a Wednesday run has.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from ffmodel.data import ingest

MECHANISMS = (
    "muscle",
    "joint_ligament",
    "head",
    "bone_contact",
    "tendon_severe",
    "spine",
    "illness",
    "nonmed",
    "multiple",
    "other",
)

_STATUS_SEVERITY = {"Out": 3, "Doubtful": 2, "Questionable": 1}

# Order matters: the first group whose keyword appears wins, so the specific
# comes before the general ("hip flexor" is a muscle before "hip" is a joint,
# "not injury related - ankle" is not an ankle).
_RULES = (
    ("nonmed", ("not injury", "personal", "rest", "coach", "non-injury", "non football",
                "suspen", "covid", "team decision", "discipline", "travel", "inactive", "did not")),
    ("illness", ("illness", "flu", "sick", "appendi", "liver", "lung", "stomach", "medical", "hernia")),
    ("head", ("concussion", "head", "face", "nose", "mouth", "teeth", "tooth", "eye", "jaw")),
    ("tendon_severe", ("achilles", "patell", "acl")),
    ("muscle", ("hamstring", "calf", "quad", "thigh", "groin", "adductor", "hip flexor", "glute",
                "core muscle", "oblique", "trapez", "bicep", "tricep", "lat", "pec", "abdom", "muscle")),
    ("spine", ("back", "neck", "spine", "lumbar", "cervical")),
    ("bone_contact", ("rib", "chest", "sternum", "hand", "finger", "thumb", "forearm",
                      "collarbone", "clavicle")),
    ("joint_ligament", ("knee", "ankle", "shoulder", "elbow", "wrist", "hip", "foot", "toe",
                        "heel", "lisfranc", "mcl")),
)


def mechanism(text) -> str:
    """The mechanism group for one free-text injury string; ``"none"`` if absent."""
    if text is None or (isinstance(text, float) and np.isnan(text)) or pd.isna(text):
        return "none"
    t = str(text).lower().replace("left ", "").replace("right ", "").strip()
    parts = [x for x in t.replace("/", ",").replace(";", ",").split(",") if x.strip()]
    # Several injuries at once is its own statement, and the wrong thing to
    # reduce to whichever keyword happens to come first.
    if len(parts) >= 2 and "not injury" not in t:
        return "multiple"
    for name, keys in _RULES:
        if any(k in t for k in keys):
            return name
    return "other"


def load_injury_mechanisms(seasons: Iterable[int]) -> pd.DataFrame:
    """One row per (season, week, player) with the mechanism on that week's report."""
    empty = pd.DataFrame(columns=["season", "week", "player_key", "mech_cur"])
    try:
        raw = ingest.load_injuries(sorted({int(s) for s in seasons}))
    except Exception:
        return empty
    if raw.empty:
        return empty
    if "game_type" in raw.columns:
        raw = raw[raw["game_type"] == "REG"]
    raw = raw.assign(
        season=pd.to_numeric(raw["season"], errors="coerce"),
        week=pd.to_numeric(raw["week"], errors="coerce"),
        severity=raw["report_status"].map(_STATUS_SEVERITY).fillna(0),
        player_key=raw["gsis_id"].astype(str),
    ).dropna(subset=["season", "week"])
    # A player can appear more than once in a week as the report is updated; the
    # most severe entry is the one that says what kept him out.
    raw = raw.sort_values("severity").drop_duplicates(["season", "week", "player_key"], keep="last")
    text = raw["report_primary_injury"].fillna(raw.get("practice_primary_injury"))
    out = raw[["season", "week", "player_key"]].copy()
    out["mech_cur"] = text.map(mechanism)
    out[["season", "week"]] = out[["season", "week"]].astype(int)
    return out


def mechanism_columns() -> tuple[str, ...]:
    """The dummy columns :func:`attach_injury_type` writes, current week then previous."""
    return tuple(f"mechc_{m}" for m in MECHANISMS) + tuple(f"mechp_{m}" for m in MECHANISMS)


def attach_injury_type(frame: pd.DataFrame, *, seasons: Iterable[int] | None = None) -> pd.DataFrame:
    """Add the mechanism on this week's report and on the previous game's.

    The previous game's is the one that matters for a player who just missed
    time: it is what kept him out. "Previous" is the previous *row* for the
    player within the season, so a bye never counts as a game.
    """
    seasons = sorted(frame["season"].unique().tolist()) if seasons is None else list(seasons)
    out = frame.copy()
    table = load_injury_mechanisms(seasons)
    out = out.drop(columns=[c for c in ("mech_cur", "mech_prev") if c in out.columns])
    out = out.merge(table, on=["season", "week", "player_key"], how="left")
    out["mech_cur"] = out["mech_cur"].fillna("none")
    order = out.sort_values(["player_key", "season", "week"], kind="mergesort").index
    lag = out.loc[order].groupby(["player_key", "season"], sort=False)["mech_cur"].shift(1)
    out["mech_prev"] = lag.reindex(out.index).fillna("none")
    for name in MECHANISMS:
        out[f"mechc_{name}"] = (out["mech_cur"] == name).astype(float)
        out[f"mechp_{name}"] = (out["mech_prev"] == name).astype(float)
    return out
