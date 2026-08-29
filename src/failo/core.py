"""Failo's public API.

Failo sits *above* your provider SDK.  You keep your OpenAI / Gemini /
Anthropic client exactly as it is and hand Failo a zero-argument callable
that performs the call.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass, field

from .fallback import execute_with_fallback
from .models import ResilientResult
from .policies import FallbackPolicy, RetryPolicy
from .retry import Operation, Sleeper

__all__ = ["ResilientClient", "resilient_call", "resilient_call_sync"]


def _event_loop_is_running() -> bool:
    """Return whether this thread already has a running event loop."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return False
    return True


async def resilient_call(
    primary: Operation,
    *,
    fallbacks: Sequence[Operation] | None = None,
    retry_policy: RetryPolicy | None = None,
    fallback_policy: FallbackPolicy | None = None,
    providers: Sequence[str | None] | None = None,
    models: Sequence[str | None] | None = None,
    sleeper: Sleeper | None = None,
) -> ResilientResult:
    """Call ``primary`` with retries, falling back to ``fallbacks`` in order.

    Args:
        primary: Zero-argument callable (async or sync) doing the real call.
        fallbacks: Callables tried, in order, if the primary ultimately fails.
        retry_policy: Retry configuration applied to every operation.
        fallback_policy: Which failures may move on to the next provider.
        providers: Optional labels aligned with ``[primary, *fallbacks]``.
        models: Optional labels aligned with ``[primary, *fallbacks]``.
        sleeper: Optional replacement for :func:`asyncio.sleep`.

    Returns:
        A :class:`~failo.models.ResilientResult` with the response and full
        execution metadata.

    Raises:
        failo.errors.AIError: When no operation succeeds.
    """
    operations: list[Operation] = [primary, *(fallbacks or [])]
    return await execute_with_fallback(
        operations,
        retry_policy,
        providers=providers,
        models=models,
        fallback_policy=fallback_policy,
        sleeper=sleeper,
    )


def resilient_call_sync(
    primary: Operation,
    *,
    fallbacks: Sequence[Operation] | None = None,
    retry_policy: RetryPolicy | None = None,
    fallback_policy: FallbackPolicy | None = None,
    providers: Sequence[str | None] | None = None,
    models: Sequence[str | None] | None = None,
) -> ResilientResult:
    """Blocking wrapper around :func:`resilient_call`.

    This is a thin :func:`asyncio.run` shim -- there is no second retry
    implementation.  It must be called from a thread with no running event
    loop; inside async code, await :func:`resilient_call` instead.

    Raises:
        RuntimeError: If an event loop is already running in this thread.
        failo.errors.AIError: When no operation succeeds.
    """
    if _event_loop_is_running():
        raise RuntimeError(
            "resilient_call_sync() cannot run inside a running event loop; "
            "await resilient_call() instead."
        )

    return asyncio.run(
        resilient_call(
            primary,
            fallbacks=fallbacks,
            retry_policy=retry_policy,
            fallback_policy=fallback_policy,
            providers=providers,
            models=models,
        )
    )


@dataclass
class ResilientClient:
    """Reusable holder for policies shared by many calls.

    Example:
        >>> failo = ResilientClient(retry_policy=RetryPolicy(max_attempts=3))
        >>> result = await failo.call(primary=openai_call, fallbacks=[gemini_call])
    """

    retry_policy: RetryPolicy = field(default_factory=RetryPolicy)
    fallback_policy: FallbackPolicy = field(default_factory=FallbackPolicy)
    providers: Sequence[str | None] | None = None
    models: Sequence[str | None] | None = None

    async def call(
        self,
        primary: Operation,
        *,
        fallbacks: Sequence[Operation] | None = None,
        retry_policy: RetryPolicy | None = None,
        fallback_policy: FallbackPolicy | None = None,
        providers: Sequence[str | None] | None = None,
        models: Sequence[str | None] | None = None,
        sleeper: Sleeper | None = None,
    ) -> ResilientResult:
        """Run :func:`resilient_call` with this client's defaults.

        Per-call arguments override the client-level configuration.
        """
        return await resilient_call(
            primary,
            fallbacks=fallbacks,
            retry_policy=retry_policy or self.retry_policy,
            fallback_policy=fallback_policy or self.fallback_policy,
            providers=providers if providers is not None else self.providers,
            models=models if models is not None else self.models,
            sleeper=sleeper,
        )

    def call_sync(
        self,
        primary: Operation,
        *,
        fallbacks: Sequence[Operation] | None = None,
        retry_policy: RetryPolicy | None = None,
        fallback_policy: FallbackPolicy | None = None,
        providers: Sequence[str | None] | None = None,
        models: Sequence[str | None] | None = None,
    ) -> ResilientResult:
        """Blocking variant of :meth:`call`."""
        return resilient_call_sync(
            primary,
            fallbacks=fallbacks,
            retry_policy=retry_policy or self.retry_policy,
            fallback_policy=fallback_policy or self.fallback_policy,
            providers=providers if providers is not None else self.providers,
            models=models if models is not None else self.models,
        )
