"""Failing over between providers with Failo.

The primary provider is permanently rate-limited, so Failo exhausts its
retries and moves on to the fallback.  No API keys required.

Run with:  python examples/fallback.py
"""

from __future__ import annotations

import asyncio

from failo import FallbackPolicy, RetryPolicy, resilient_call
from failo.errors import AIError, RateLimitError


async def primary_call() -> str:
    """Pretend this wraps your OpenAI client."""
    raise RateLimitError("429 Too Many Requests", retry_after=0.01)


async def fallback_call() -> str:
    """Pretend this wraps your Gemini client."""
    return "Hello from the fallback provider"


async def main() -> None:
    try:
        result = await resilient_call(
            primary=primary_call,
            fallbacks=[
                fallback_call,
            ],
            retry_policy=RetryPolicy(max_attempts=2, initial_delay=0.05),
            fallback_policy=FallbackPolicy(),
            providers=["openai", "google"],
            models=["gpt-5", "gemini-3.7-flash"],
        )
    except AIError as error:
        print("every provider failed:", error)
        return

    print("response:      ", result.response)
    print("provider:      ", result.provider)
    print("model:         ", result.model)
    print("attempts:      ", result.attempts)
    print("fallback_used: ", result.fallback_used)
    print("history:")
    for attempt in result.attempt_history:
        status = "ok" if attempt.success else type(attempt.error).__name__
        print(f"  {attempt.provider} attempt {attempt.attempt_number}: {status}")


if __name__ == "__main__":
    asyncio.run(main())
