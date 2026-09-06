"""Who can be picked up right now, and who is locked up.

The environment used to keep one list of free agents and let anybody take
anybody, capped at one add a week. Both halves of that were wrong. Real leagues
cap nothing -- a manager can churn the back of a roster all week if he is willing
to keep cutting -- and a player who is cut does not land in the pool available to
the fastest click. He sits on waivers for two days first.

The rules implemented here, and they are the ones a league states:

**A dropped player goes on waivers for 48 hours.** Nobody can add him until the
period ends. This is what stops a manager cutting a starter on Wednesday and
picking him straight back up when the hole he was covering turns out not to
matter -- which the roster mechanic was doing before this existed, in the same
transaction.

**Everyone else is instantly addable, with no cap.** A player who has cleared
waivers, or who was never on them, can be taken by whoever wants him, as often as
a manager is willing to cut somebody to make room. The roster size is the only
budget.

**A player dropped within 24 hours of being added skips waivers.** Without this
rule a manager can lock a player away from the league for two days by adding and
immediately cutting him, and the automatic roster housekeeping would do it by
accident every time it claimed a replacement and then needed the spot back.

Time is tracked in hours from the start of the season, because two days is not a
number of weeks and the mechanic is invisible at weekly granularity: a week has
two transaction points, Wednesday and Friday, and the whole point of a 48-hour
period is that it separates them. A player cut on Wednesday is available again on
Friday; one cut on Friday is not available until the following Wednesday, because
his period ends after the games have been played.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Hours in the transaction week. Wednesday is when waivers are understood to
# run, and Friday is two days later, which is exactly the period's length -- so
# the two phases are what make it observable rather than an accounting entry.
HOURS_PER_WEEK = 168
WEDNESDAY, FRIDAY = 0, 48
PHASES = (WEDNESDAY, FRIDAY)

WAIVER_HOURS = 48
INSTANT_DROP_HOURS = 24


def hour_of(week_index: int, phase: int) -> int:
    """Hours from the season's start for a transaction phase."""
    return week_index * HOURS_PER_WEEK + phase


@dataclass
class WaiverWire:
    """The pool, split into what is available now and what is locked.

    Holds every player who is not on a roster. A player nobody has ever rostered
    has no history here and is free, which is the right default: undrafted
    players are available from week one.
    """

    _clears: dict[str, int] = field(default_factory=dict, repr=False)
    _added: dict[str, int] = field(default_factory=dict, repr=False)

    def is_free(self, key: str, hour: int) -> bool:
        """Can this player be added right now?"""
        clears = self._clears.get(key)
        return clears is None or hour >= clears

    def on_waivers(self, hour: int) -> set[str]:
        return {key for key, clears in self._clears.items() if hour < clears}

    def clears_at(self, key: str) -> int | None:
        return self._clears.get(key)

    def added(self, key: str, hour: int) -> None:
        """Record a player joining a roster."""
        self._added[key] = int(hour)
        self._clears.pop(key, None)

    def dropped(self, key: str, hour: int) -> bool:
        """Record a player leaving a roster. Returns whether he went on waivers.

        The 24-hour exemption is checked against when *this* roster added him,
        which is the only reading that does what the rule is for: it lets a
        manager undo a mistake without letting him quarantine a player.
        """
        hour = int(hour)
        added = self._added.pop(key, None)
        if added is not None and hour - added < INSTANT_DROP_HOURS:
            self._clears.pop(key, None)
            return False
        self._clears[key] = hour + WAIVER_HOURS
        return True

    def free_agents(self, pool, hour: int) -> list[str]:
        """Those of ``pool`` who can be added at ``hour``, order preserved."""
        return [key for key in pool if self.is_free(key, hour)]

    def drafted(self, keys, hour: int = -HOURS_PER_WEEK) -> None:
        """Seed the wire from the draft.

        The default hour is a week before the season, and it is not cosmetic. A
        draft recorded at hour zero makes every cut in week one look like a
        24-hour undo, so a manager could churn his whole bench through free
        agency in the opening week and nobody would ever hit waivers. The draft
        is days earlier than the first transaction, and saying so is what makes
        the first week behave like every other one.
        """
        for key in keys:
            self._added[key] = int(hour)
