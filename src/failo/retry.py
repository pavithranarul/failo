"""The async retry engine.

This module owns the *only* retry loop in Failo.  Everything else --
fallback, the public API, the sync helper -- delegates here so retry
semantics can never drift between entry points.
"""

from __future__ import annotations

import asyncio
import inspect
import time
from collections.abc import Awaitable, Callable
from typing import Any

from .errors import AIError, extract_retry_after
from .models import Attempt, RetryOutcome
from .policies import RetryPolicy

__all__ = ["Operation", "Sleeper", "call_operation", "execute_with_retry"]

#: A user-supplied callable taking no arguments.  It may be an async function,
#: a sync function, or anything returning an awaitable.  Failo never inspects
#: the return value, so provider responses pass through untouched.
Operation = Callable[[], Any]

#: Pluggable sleep, so tests (and future schedulers) can observe delays.
Sleeper = Callable[[float], Awaitable[None]]


async def call_operation(operation: Operation) -> Any:
    """Invoke ``operation``, awaiting it when it returns an awaitable."""
    result = operation()
    if inspect.isawaitable(result):
        return await result
    return result


async def _sleep(delay: float, sleeper: Sleeper | None) -> None:
    """Wait ``delay`` seconds using ``sleeper`` or ``asyncio.sleep``."""
    if delay <= 0:
        return
    if sleeper is None:
        await asyncio.sleep(delay)
    else:
        await sleeper(delay)


def _delay_for(policy: RetryPolicy, attempt_number: int, error: AIError) -> float:
    """Delay before the next attempt, honouring a server retry-after hint."""
    retry_after = error.retry_after if error.retry_after is not None else extract_retry_after(error)
    return policy.get_delay(attempt_number, retry_after=retry_after)


async def execute_with_retry(
    operation: Operation,
    policy: RetryPolicy | None = None,
    *,
    provider: str | None = None,
    model: str | None = None,
    history: list[Attempt] | None = None,
    sleeper: Sleeper | None = None,
) -> RetryOutcome:
    """Run ``operation`` under ``policy`` until it succeeds or gives up.

    ``policy.max_attempts`` counts *total* executions, so ``max_attempts=3``
    means one call plus at most two retries.

    Args:
        operation: Zero-argument callable wrapping the provider call.
        policy: Retry configuration; a default :class:`RetryPolicy` is used
            when omitted.
        provider: Optional provider label recorded on each attempt.
        model: Optional model label recorded on each attempt.
        history: Optional list that attempts are appended to.  The fallback
            engine passes a shared list so history survives across providers.
        sleeper: Optional replacement for :func:`asyncio.sleep`.

    Returns:
        A :class:`~failo.models.RetryOutcome` holding the response and the
        attempts made by this call.

    Raises:
        failo.errors.AIError: The normalized final error, chained to the
            original provider exception via ``__cause__``.
    """
    policy = policy or RetryPolicy()
    shared_history = history if history is not None else []
    attempts: list[Attempt] = []
    last_error: AIError | None = None

    for attempt_number in range(1, policy.max_attempts + 1):
        started = time.monotonic()
        try:
            response = await call_operation(operation)
        except Exception as exc:  # noqa: BLE001 - re-raised after classification
            latency = time.monotonic() - started
            classified = policy.classify(exc)
            last_error = classified
            attempt = Attempt(
                provider=provider,
                model=model,
                attempt_number=attempt_number,
                success=False,
                error=classified,
                latency=latency,
            )
            attempts.append(attempt)
            shared_history.append(attempt)

            is_last_attempt = attempt_number >= policy.max_attempts
            if is_last_attempt or not policy.should_retry(classified):
                classified.attempt_history = list(shared_history)
                if classified is exc:
                    raise
                raise classified from exc

            await _sleep(_delay_for(policy, attempt_number, classified), sleeper)
        else:
            attempt = Attempt(
                provider=provider,
                model=model,
                attempt_number=attempt_number,
                success=True,
                error=None,
                latency=time.monotonic() - started,
            )
            attempts.append(attempt)
            shared_history.append(attempt)
            return RetryOutcome(response=response, attempts=attempts)

    # Unreachable: the loop either returns or raises, and max_attempts >= 1.
    raise last_error if last_error is not None else AIError("Retry loop exited without result")
