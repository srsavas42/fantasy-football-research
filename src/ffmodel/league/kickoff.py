"""When each club plays, so a lineup can lock one player at a time.

A fantasy week is not one deadline. Every player locks when *his* game kicks
off, and a manager keeps deciding around the ones who have not started yet: the
Sunday afternoon flex is still open while the Sunday morning games are being
played, and the Monday night slot is open all weekend. The environment used to
collapse that into a single decision taken before any game, which is the
strictest reading of the rules and not the one any league uses.

A week has about seven distinct kickoff times, and a fifteen-man roster spans
about five of them. That leaves roughly four decision points after the first, of
which around three offer a genuine choice -- two or more unlocked players who
could fill the same slot. Every week of the three holdout seasons has at least
one.

**What the extra decision points are worth depends entirely on what changes
between them**, and only one thing does: the score. Nobody learns anything new
about a player's Sunday afternoon by watching Sunday morning. What they learn is
whether they are ahead. That matters because the league pays a win bonus rather
than paying points, so a manager who is thirty behind going into the last game
wants the volatile player and a manager who is thirty ahead wants the steady one
-- and neither preference exists at all in a policy that maximises points.

So the sequencing is a rule of the environment, available to every team.
Policies that ignore it produce exactly the lineup they produce today, because
re-running the same scoring function on the same information returns the same
answer. Only a policy that reads the in-week state behaves differently, and
:attr:`ffmodel.league.policies.Policy.reactive` is how one says so -- the
environment re-scores only those, which keeps a week costing what it costs now
for everybody else.

Kickoff times come from nflverse's schedule and are cached, because they are a
fact about a season that never changes once it is over.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

CACHE = Path(".cache")


def _cache_path(seasons) -> Path:
    return CACHE / f"kickoffs_{min(seasons)}_{max(seasons)}.parquet"


def load_kickoffs(seasons, *, cache: Path | None = None) -> pd.DataFrame:
    """``season, week, club, kickoff`` for every regular-season game.

    One row per club per game, so a player's kickoff is his club's kickoff.
    """
    seasons = sorted({int(s) for s in seasons})
    path = cache if cache is not None else _cache_path(seasons)
    if path.exists():
        return pd.read_parquet(path)

    import nflreadpy as nfl

    raw = nfl.load_schedules(seasons).to_pandas()
    raw = raw[raw["game_type"] == "REG"].copy()
    kickoff = pd.to_datetime(
        raw["gameday"].astype(str) + " " + raw["gametime"].astype(str), errors="coerce"
    )
    if kickoff.isna().all():
        raise ValueError("nflverse schedule carried no usable kickoff times")
    raw["kickoff"] = kickoff

    frames = []
    for side in ("home_team", "away_team"):
        frames.append(
            raw[["season", "week", side, "kickoff"]].rename(columns={side: "club"})
        )
    out = pd.concat(frames, ignore_index=True)
    out = out.dropna(subset=["kickoff"]).astype({"season": int, "week": int})
    out = out.drop_duplicates(["season", "week", "club"])
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)
    return out


@dataclass
class KickoffSlots:
    """Which decision point each player belongs to, for one season.

    Slots are indexed within a week: 0 is the first game of the week, and a
    player in slot ``k`` can still be moved at every decision point up to and
    including ``k``. A player whose club has no game -- a bye, or the week
    Buffalo and Cincinnati never played -- has no slot and cannot be started
    anyway.
    """

    season: int
    _slot: dict[tuple[int, str], int] = field(default_factory=dict, repr=False)
    _counts: dict[int, int] = field(default_factory=dict, repr=False)

    def slots_in(self, week: int) -> int:
        """How many decision points ``week`` has."""
        return self._counts.get(int(week), 1)

    def slot_of_club(self, club: str, week: int) -> int | None:
        return self._slot.get((int(week), str(club)))

    def slot_of(self, key: str, week: int, club_of) -> int | None:
        club = club_of(key, week)
        return None if club is None else self.slot_of_club(club, int(week))


def build_kickoff_slots(seasons, season: int, *, cache: Path | None = None):
    """Index every club's kickoff into a within-week slot number."""
    frame = load_kickoffs(seasons, cache=cache)
    block = frame[frame["season"] == int(season)]
    if block.empty:
        raise ValueError(f"no schedule rows for season {season}")

    slot: dict[tuple[int, str], int] = {}
    counts: dict[int, int] = {}
    for week, games in block.groupby("week", sort=True):
        order = {
            time: index
            for index, time in enumerate(sorted(games["kickoff"].unique()))
        }
        counts[int(week)] = len(order)
        for club, time in zip(games["club"], games["kickoff"]):
            slot[(int(week), str(club))] = order[time]
    return KickoffSlots(season=int(season), _slot=slot, _counts=counts)
