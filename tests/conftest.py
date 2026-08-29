"""Shared fakes for the Failo test suite.

No network, no provider SDKs, no real sleeping: delays are captured by a
recording sleeper so the suite stays fast and deterministic.
"""

from __future__ import annotations

from typing import Any

import pytest


class FakeHTTPError(Exception):
    """Generic HTTP-style provider exception with a status code."""

    def __init__(self, status_code: int, message: str = "boom", **extra: Any) -> None:
        super().__init__(message)
        self.status_code = status_code
        for key, value in extra.items():
            setattr(self, key, value)


class RecordingSleeper:
    """Async sleep replacement that records delays instead of waiting."""

    def __init__(self) -> None:
        self.delays: list[float] = []

    async def __call__(self, delay: float) -> None:
        self.delays.append(delay)


class ScriptedOperation:
    """Callable that raises/returns a scripted sequence of outcomes."""

    def __init__(self, *outcomes: Any) -> None:
        self.outcomes = list(outcomes)
        self.calls = 0

    async def __call__(self) -> Any:
        index = min(self.calls, len(self.outcomes) - 1)
        self.calls += 1
        outcome = self.outcomes[index]
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def always_fail(error: BaseException) -> ScriptedOperation:
    """An operation that always raises ``error``."""
    return ScriptedOperation(error)


def fail_then(error: BaseException, times: int, value: Any) -> ScriptedOperation:
    """Raise ``error`` ``times`` times, then return ``value`` forever."""
    return ScriptedOperation(*([error] * times), value)


@pytest.fixture
def sleeper() -> RecordingSleeper:
    """A fresh recording sleeper per test."""
    return RecordingSleeper()
