"""Basic Failo usage: retry a single provider call.

No API keys and no provider SDK required -- the "provider" here is a fake
callable that fails twice before succeeding.

Run with:  python examples/basic.py
"""

from __future__ import annotations

import asyncio

from failo import RetryPolicy, resilient_call
from failo.errors import ServerError


class FlakyProvider:
    """Stand-in for your real AI client."""

    def __init__(self, failures: int) -> None:
        self.remaining = failures

    async def complete(self, prompt: str) -> str:
        if self.remaining > 0:
            self.remaining -= 1
            raise ServerError("503 Service Unavailable")
        return f"echo: {prompt}"


async def main() -> None:
    provider = FlakyProvider(failures=2)

    # In real code this would be:
    #     return await openai_client.responses.create(...)
    async def call_provider() -> str:
        return await provider.complete("Hello, Failo")

    result = await resilient_call(
        primary=call_provider,
        retry_policy=RetryPolicy(max_attempts=4, initial_delay=0.05),
        providers=["demo"],
        models=["demo-model"],
    )

    print("response:      ", result.response)
    print("provider:      ", result.provider)
    print("model:         ", result.model)
    print("attempts:      ", result.attempts)
    print("fallback_used: ", result.fallback_used)
    print(f"total_latency: {result.total_latency:.3f}s")
    for attempt in result.attempt_history:
        status = "ok" if attempt.success else type(attempt.error).__name__
        print(f"  attempt {attempt.attempt_number}: {status}")


if __name__ == "__main__":
    asyncio.run(main())
