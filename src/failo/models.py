"""Execution metadata returned by Failo."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = ["Attempt", "ResilientResult", "RetryOutcome"]


@dataclass
class Attempt:
    """A single execution of one operation.

    Attributes:
        provider: Caller-supplied provider label, if any.
        model: Caller-supplied model label, if any.
        attempt_number: 1-based attempt index *within this operation*.
        success: Whether the call returned without raising.
        error: The normalized error when the call failed, else ``None``.
        latency: Wall-clock seconds spent on the call.
    """

    provider: str | None
    model: str | None
    attempt_number: int
    success: bool
    error: Exception | None = None
    latency: float | None = None


@dataclass
class ResilientResult:
    """The outcome of a successful :func:`failo.resilient_call`.

    Attributes:
        response: Whatever the successful callable returned, untouched.
        provider: Label of the provider that succeeded.
        model: Label of the model that succeeded.
        attempts: Total number of executions across every provider.
        fallback_used: ``True`` when the primary did not produce the result.
        total_latency: Wall-clock seconds for the whole resilient call.
        attempt_history: Every attempt, in order, across every provider.
    """

    response: Any
    provider: str | None = None
    model: str | None = None
    attempts: int = 0
    fallback_used: bool = False
    total_latency: float = 0.0
    attempt_history: list[Attempt] = field(default_factory=list)


@dataclass
class RetryOutcome:
    """The outcome of a successful :func:`failo.retry.execute_with_retry`."""

    response: Any
    attempts: list[Attempt] = field(default_factory=list)
