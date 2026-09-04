"""Typed, versioned inputs for annual prediction releases.

This module declares data and validation only; lifecycle transitions, fitting,
storage, and command execution remain owned by later tickets.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum

from .errors import ReleaseContractError

CONTRACT_VERSION = "release-contract.v1"
MIN_SAMPLER_DRAWS = 2_000
MIN_SAMPLER_TUNE = 2_000
MIN_SAMPLER_CHAINS = 4


class Position(str, Enum):
    QB = "QB"
    RB = "RB"
    WR = "WR"
    TE = "TE"


class ScoringFormat(str, Enum):
    STANDARD = "standard"
    HALF_PPR = "half_ppr"
    PPR = "ppr"


class ReleaseState(str, Enum):
    """Declared states only; no transition method grants orchestration ownership."""

    CAPTURED = "CAPTURED"
    BOUND = "BOUND"
    FITTED = "FITTED"
    VALIDATED = "VALIDATED"
    AWAITING_APPROVAL = "AWAITING_APPROVAL"
    PUBLISHABLE = "PUBLISHABLE"
    REJECTED = "REJECTED"
    PUBLISHED = "PUBLISHED"


def target_season_cutoff(target_season: int) -> datetime:
    """Return exactly target-year August 31 00:00:00 UTC."""
    if isinstance(target_season, bool) or not isinstance(target_season, int):
        raise ReleaseContractError("target_season must be an explicit integer")
    if not 2000 <= target_season <= 9999:
        raise ReleaseContractError("target_season must be between 2000 and 9999")
    return datetime(target_season, 8, 31, tzinfo=timezone.utc)


def require_utc(value: datetime, *, field_name: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() != timezone.utc.utcoffset(value):
        raise ReleaseContractError(f"{field_name} must be timezone-aware UTC")
    return value.astimezone(timezone.utc)


@dataclass(frozen=True)
class ReleaseIdentity:
    target_season: int
    attempt: str

    def __post_init__(self) -> None:
        target_season_cutoff(self.target_season)
        if not isinstance(self.attempt, str) or not self.attempt or len(self.attempt) > 128 or any(c.isspace() for c in self.attempt):
            raise ReleaseContractError("attempt must be a non-empty, whitespace-free string of at most 128 characters")

    @property
    def cutoff(self) -> datetime:
        return target_season_cutoff(self.target_season)


@dataclass(frozen=True)
class SamplerMinimum:
    """Minimum production budget: 2000 draws/tune and four chains."""

    draws: int = MIN_SAMPLER_DRAWS
    tune: int = MIN_SAMPLER_TUNE
    chains: int = MIN_SAMPLER_CHAINS

    def __post_init__(self) -> None:
        for name, minimum in (("draws", MIN_SAMPLER_DRAWS), ("tune", MIN_SAMPLER_TUNE), ("chains", MIN_SAMPLER_CHAINS)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ReleaseContractError(f"{name} must be an integer of at least {minimum}")


@dataclass(frozen=True)
class ReleaseContract:
    """v1 controls: target season is mandatory, refit is mandatory, reuse is off."""

    identity: ReleaseIdentity
    scoring_formats: tuple[ScoringFormat, ...]
    sampler_minimum: SamplerMinimum = SamplerMinimum()
    refit_required: bool = True
    artifact_reuse_allowed: bool = False
    contract_version: str = CONTRACT_VERSION

    def __post_init__(self) -> None:
        if self.contract_version != CONTRACT_VERSION:
            raise ReleaseContractError(f"contract_version must be {CONTRACT_VERSION!r}")
        if not isinstance(self.identity, ReleaseIdentity) or not isinstance(self.sampler_minimum, SamplerMinimum):
            raise ReleaseContractError("identity and sampler_minimum must use release contract types")
        if self.refit_required is not True or self.artifact_reuse_allowed is not False:
            raise ReleaseContractError("v1 requires refitting and forbids artifact reuse")
        try:
            formats = tuple(ScoringFormat(item) for item in self.scoring_formats)
        except (TypeError, ValueError) as exc:
            raise ReleaseContractError("scoring_formats contains an unknown format") from exc
        if not formats or len(set(formats)) != len(formats):
            raise ReleaseContractError("scoring_formats must be non-empty and unique")
        object.__setattr__(self, "scoring_formats", tuple(sorted(formats, key=lambda item: item.value)))
