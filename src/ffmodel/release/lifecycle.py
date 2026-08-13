"""Fail-closed release lifecycle rules owned by the storage boundary."""

from __future__ import annotations

from .contract import ReleaseState
from .errors import ReleaseContractError


class LifecycleError(ReleaseContractError):
    """Raised when a persisted release lifecycle operation is unsafe."""


_ALLOWED: dict[ReleaseState, frozenset[ReleaseState]] = {
    ReleaseState.CAPTURED: frozenset({ReleaseState.BOUND}),
    ReleaseState.BOUND: frozenset({ReleaseState.FITTED}),
    ReleaseState.FITTED: frozenset({ReleaseState.VALIDATED}),
    ReleaseState.VALIDATED: frozenset({ReleaseState.AWAITING_APPROVAL}),
    ReleaseState.AWAITING_APPROVAL: frozenset({ReleaseState.PUBLISHABLE, ReleaseState.REJECTED}),
    ReleaseState.PUBLISHABLE: frozenset({ReleaseState.PUBLISHED}),
    ReleaseState.REJECTED: frozenset(),
    ReleaseState.PUBLISHED: frozenset(),
}


def validate_transition(current: ReleaseState, target: ReleaseState) -> None:
    """Allow only the explicit v1 forward state graph."""
    if not isinstance(current, ReleaseState) or not isinstance(target, ReleaseState):
        raise LifecycleError("release states must be typed ReleaseState values")
    if target not in _ALLOWED[current]:
        raise LifecycleError(f"invalid release transition: {current.value} -> {target.value}")
