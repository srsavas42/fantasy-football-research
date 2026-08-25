"""Preseason market consensus as a model input.

Average draft position is the only signal in this pipeline that is not a
transform of something that happened on a field. It aggregates what drafters
believed before a season started -- beat reports, camp bodies, contracts,
holdouts, a new coordinator's stated plan, a rookie's first-team reps -- none of
which reaches a box score until the season it describes is already over. That
makes it a genuinely new source rather than another view of the history the
model already has, and because it is published before the season it forecasts,
it is legitimate at serve time.

It also makes the model a partial follower of the market, and that is a real
cost to state plainly: a projection that reads ADP is no longer independent of
consensus, and "the model beats ADP" stops meaning what it meant. The question
worth answering is not whether the model can beat consensus from scratch but
whether it adds anything on top of it, and that question needs the feature in
order to be asked.

Two encoding decisions carry most of the risk.

**Depth is capped at a common rank.** The published lists run from 352 rows
(2022) to 1046 (2019). Left uncapped, "undrafted" would mean rank>352 in one
season and rank>1046 in another, and the feature would encode which file was
longer. Capped at :data:`ADP_DEPTH`, every season contributes a comparable
population -- 236 to 267 skill-position players across 2015-2026.

**Undrafted is a value, not a gap.** The feature matrices fill missing entries
with the column median, which for a player the market declined to rank would
assert an average draft position: precisely the opposite of what his absence
says. These columns therefore carry no missing values. A player the market did
not rank is placed one past the cap and flagged, so the model can learn the
level and the within-drafted gradient separately.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

MODEL_POSITIONS = ("QB", "RB", "WR", "TE")

# Every FantasyPros file from 2015 on lists at least 352 players, so this depth
# is available in all of them. See the module docstring.
ADP_DEPTH = 300

ADP_FEATURES = ("adp_log_rank", "adp_position_log_rank", "adp_drafted")

# Positions carrying their own rank slope and their own drafted effect. One is
# left out as the reference, the way position dummies are coded elsewhere in
# this package; with all four the design would be rank deficient and the SVD
# would drop a direction anyway.
INTERACTION_POSITIONS = MODEL_POSITIONS[:-1]

ADP_INTERACTION_FEATURES = tuple(
    f"adp_log_rank_x_{position.lower()}" for position in INTERACTION_POSITIONS
) + tuple(f"adp_drafted_x_{position.lower()}" for position in INTERACTION_POSITIONS)

# Resolved from this file rather than the working directory: the frames get
# built from scripts, notebooks and tests that do not share a cwd, and a
# relative path would make the feature present or absent depending on where
# python was started.
DEFAULT_ADP_DIR = Path(__file__).resolve().parents[3] / "ADP"


def _name_key(name: pd.Series) -> pd.Series:
    """Lowercase letters only, generational suffixes dropped.

    Deliberately not fuzzy. A fuzzy join would pair the wrong player silently
    and the mistake would surface as a coefficient rather than an exception.
    """
    text = name.astype(str).str.lower()
    text = text.str.replace(r"\s+(jr|sr|ii|iii|iv|v)\.?$", "", regex=True)
    return text.str.replace(r"[^a-z]", "", regex=True)


def read_adp_file(path: Path) -> pd.DataFrame:
    """One season's list, reduced to name key, position, and two ranks.

    The player column is ``"Name TEAM (BYE)"`` in recent files and a bare name
    in older ones -- the team suffix is present on 10% of 2015 rows and 84% of
    2026 rows -- so the team is not usable as a join key and the parse falls
    back to the whole string.
    """
    raw = pd.read_csv(path)
    player = raw["Player (Bye)"].astype(str).str.strip()
    parsed = player.str.extract(r"^(?P<name>.*?)\s+[A-Z]{2,3}\s*\(\w+\)$")["name"]
    out = pd.DataFrame(
        {
            "key": _name_key(parsed.fillna(player)),
            "adp_position": raw["POS"].astype(str).str.extract(r"^([A-Z]+)")[0],
            "adp_rank": pd.to_numeric(raw["Rank"], errors="coerce"),
            # "RB7" carries the positional rank the overall rank does not: a
            # quarterback taken 40th overall is the fourth at his position,
            # while a running back taken 40th is the fifteenth.
            "adp_position_rank": pd.to_numeric(
                raw["POS"].astype(str).str.extract(r"(\d+)$")[0], errors="coerce"
            ),
        }
    )
    out = out[
        out.adp_rank.le(ADP_DEPTH)
        & out.adp_position.isin(MODEL_POSITIONS)
        & out.key.ne("")
    ]
    # Two ranked players who normalize to the same key cannot be told apart, so
    # neither gets a rank. Assigning one of them the other's draft position is
    # the failure this join exists to avoid.
    return out[~out.key.duplicated(keep=False)].reset_index(drop=True)


def load_adp(seasons, directory: Path | str = DEFAULT_ADP_DIR) -> pd.DataFrame:
    """Every requested season's list, stacked. Missing files are an error.

    A silently absent file would mark an entire season undrafted, which reads as
    a season in which the market liked nobody rather than as missing data.
    """
    directory = Path(directory)
    frames = []
    for season in sorted({int(s) for s in seasons}):
        path = directory / f"FantasyPros_{season}_Overall_ADP_Rankings.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"no ADP list for {season} at {path}. Every season in the frame "
                "needs one -- an absent file would mark the whole season "
                "undrafted rather than unknown"
            )
        block = read_adp_file(path)
        block["season"] = season
        frames.append(block)
    if not frames:
        # No seasons asked for. Reachable from the projection path, whose
        # history helper is legitimately empty, and distinct from a season whose
        # file is missing -- that case raises above and must keep raising.
        return pd.DataFrame(columns=["player_name", "position", "team", "rank", "season"])
    return pd.concat(frames, ignore_index=True)


def add_market_adp_features(
    rows: pd.DataFrame,
    *,
    directory: Path | str = DEFAULT_ADP_DIR,
    require_position_match: bool = True,
) -> pd.DataFrame:
    """Attach preseason consensus rank to player-season rows.

    ``require_position_match`` drops a join whose ADP position disagrees with
    the roster position. It costs the occasional hybrid who is an RB to the
    market and a WR to nflverse, and it buys protection against a name that
    happens to be shared across positions. Losing a rank is a smaller error than
    inventing one.

    A player who appears twice in a season because he was traded gets the same
    rank on both stints, which is correct: he was drafted once.

    A missing directory leaves the columns off entirely, so a checkout without
    the ADP data still builds frames; the models that consume the feature raise
    on the absent column rather than fitting without it. A directory that exists
    but is missing one season is a different thing -- corrupt input, not absent
    input -- and :func:`load_adp` raises.
    """
    out = rows.copy()
    if not Path(directory).is_dir():
        return out
    seasons = pd.to_numeric(out.get("season"), errors="coerce").dropna().unique()
    adp = load_adp(seasons, directory)

    if out.empty:
        # Nothing to join. Reachable from the projection path's history helper,
        # whose frame is empty when the earliest season rebuilds to no rows; the
        # merge below would otherwise fail looking for "key" on an empty ADP
        # frame, naming a column rather than the cause.
        for name in ADP_FEATURES:
            out[name] = np.zeros(0, dtype=float)
        return out

    key = _name_key(out["player_name"])
    frame = pd.DataFrame({"season": pd.to_numeric(out["season"], errors="coerce"), "key": key})
    frame["position"] = out["position"].astype(str).str.upper()
    merged = frame.merge(adp, on=["season", "key"], how="left")
    if len(merged) != len(out):
        raise AssertionError(
            "the ADP join changed the row count, which means a duplicate "
            "survived read_adp_file's collision guard"
        )
    if require_position_match:
        disagrees = merged.adp_position.notna() & merged.adp_position.ne(merged.position)
        merged.loc[disagrees, ["adp_rank", "adp_position_rank"]] = np.nan

    drafted = merged.adp_rank.notna()
    # One past the cap, so an unranked player sits just outside the deepest
    # ranked one rather than at an arbitrary distance from him.
    rank = merged.adp_rank.fillna(ADP_DEPTH + 1)
    # The positional list is far shorter than the overall one, so its sentinel
    # is the deepest rank actually observed at that position plus one, computed
    # per position rather than shared.
    position_rank = merged.adp_position_rank.copy()
    overall = merged.adp_position_rank.max()
    # A position the market ranked nobody at falls back to the deepest rank seen
    # anywhere. Falling back to 1 instead would place an unranked player at the
    # top of his position, which is the sentinel pointing the wrong way.
    fallback = (overall + 1) if pd.notna(overall) else 1.0
    for _, block in merged.groupby("position"):
        observed = block.adp_position_rank.max()
        floor = (observed + 1) if pd.notna(observed) else fallback
        position_rank.loc[block.index] = block.adp_position_rank.fillna(floor)
    position_rank = position_rank.fillna(fallback)

    out["adp_rank"] = rank.to_numpy(dtype=float)
    out["adp_log_rank"] = np.log(rank.to_numpy(dtype=float))
    out["adp_position_log_rank"] = np.log(position_rank.to_numpy(dtype=float))
    out["adp_drafted"] = drafted.to_numpy(dtype=float)

    # Interactions, because the submodels form a linear predictor with an
    # additive position effect and cannot otherwise express that points fall
    # with rank at a different rate for a quarterback than for a running back.
    #
    # TESTED AND REJECTED. Built on a probe that fitted rank, position and
    # drafted and nothing else, where these terms are worth 4.11% on logit snap
    # share. With the model's own SNAP_FEATURES present they are worth +0.04%:
    # the usage history already carries whatever the market's positional
    # structure knows about exposure. In the pipeline they cost 1.12% pooled
    # drafted-pool MAE across three holdouts of three, at double the fit time,
    # because their encoding is collinear with the main effects and the shared
    # Normal(0, 0.35) prior collapses the whole ADP block from 1.13x to 0.25x of
    # its unregularized magnitude -- taking the working main effects down with
    # it. Re-encoding as absolute per-position slopes handles the penalty better
    # and is still worse than no interaction at every penalty strength.
    #
    # Kept, unused, so the negative result stays reproducible. Do not enable
    # without a reason these measurements do not already cover. See
    # docs/adp-ablation-2026-08.md.
    #
    # There is deliberately no drafted-by-rank term. Every unranked player sits
    # at the same sentinel, so rank is an exact linear combination of the
    # intercept, the drafted flag and their product -- the flag already gives
    # drafted players their own slope, and the interaction adds a column
    # without adding rank.
    position_of = out["position"].astype(str).str.upper()
    for position in INTERACTION_POSITIONS:
        indicator = position_of.eq(position).to_numpy(dtype=float)
        out[f"adp_log_rank_x_{position.lower()}"] = indicator * out["adp_log_rank"]
        out[f"adp_drafted_x_{position.lower()}"] = indicator * out["adp_drafted"]
    return out
