"""Fail-closed errors for the versioned annual-release contract."""


class ReleaseContractError(ValueError):
    """Raised when a release-contract invariant is violated."""


class SchemaValidationError(ReleaseContractError):
    """Raised when a serialized release schema is malformed or ambiguous."""
