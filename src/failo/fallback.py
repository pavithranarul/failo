"""The fallback engine: try each provider in order, retrying each one."""

from __future__ import annotations

import time
from collections.abc import Sequence

from .errors import AIError
from .models import Attempt, ResilientResult
from .policies import FallbackPolicy, RetryPolicy
from .retry import Operation, Sleeper, execute_with_retry

__all__ = ["execute_with_fallback"]


def _label(labels: Sequence[str | None] | None, index: int) -> str | None:
    """Return ``labels[index]`` when available, else ``None``."""
    if labels is None or index >= len(labels):
        return None
    return labels[index]


async def execute_with_fallback(
    operations: Sequence[Operation],
    policy: RetryPolicy | None = None,
    *,
    providers: Sequence[str | None] | None = None,
    models: Sequence[str | None] | None = None,
    fallback_policy: FallbackPolicy | None = None,
    sleeper: Sleeper | None = None,
) -> ResilientResult:
    """Run ``operations`` in order until one succeeds.

    The first entry is the primary; every later entry is a fallback.  Each
    operation gets the full retry policy before the next one is considered,
    and the attempt history is preserved across all of them.

    Args:
        operations: Primary first, then fallbacks.  Any length is supported.
        policy: Retry policy applied to every operation.
        providers: Optional provider labels, positionally aligned.
        models: Optional model labels, positionally aligned.
        fallback_policy: Which failures may move on to the next provider.
        sleeper: Optional replacement for :func:`asyncio.sleep`.

    Returns:
        A :class:`~failo.models.ResilientResult` describing the successful
        call and everything attempted before it.

    Raises:
        ValueError: If ``operations`` is empty.
        failo.errors.AIError: The final normalized error when every operation
            fails, or the first error that fallback is not allowed for.  The
            full attempt history is attached as ``error.attempt_history``.
    """
    if not operations:
        raise ValueError("execute_with_fallback requires at least one operation")

    policy = policy or RetryPolicy()
    fallback_policy = fallback_policy or FallbackPolicy()
    history: list[Attempt] = []
    started = time.monotonic()
    last_index = len(operations) - 1

    for index, operation in enumerate(operations):
        provider = _label(providers, index)
        model = _label(models, index)
        try:
            outcome = await execute_with_retry(
                operation,
                policy,
                provider=provider,
                model=model,
                history=history,
                sleeper=sleeper,
            )
        except AIError as error:
            error.attempt_history = list(history)
            if index == last_index or not fallback_policy.should_fallback(error):
                raise
            continue

        return ResilientResult(
            response=outcome.response,
            provider=provider,
            model=model,
            attempts=len(history),
            fallback_used=index > 0,
            total_latency=time.monotonic() - started,
            attempt_history=list(history),
        )

    # Unreachable: the loop returns on success and re-raises on the last
    # operation's failure.
    raise AIError("Fallback loop exited without result")
