from __future__ import annotations

import pytest

from failo import FallbackPolicy, RetryPolicy
from failo.errors import (
    AuthenticationError,
    InvalidRequestError,
    PermissionError,
    RateLimitError,
    ServerError,
)
from failo.fallback import execute_with_fallback
from tests.conftest import FakeHTTPError, RecordingSleeper, ScriptedOperation, fail_then

FAST = RetryPolicy(max_attempts=2, initial_delay=1.0, jitter=False)


async def test_primary_success(sleeper: RecordingSleeper) -> None:
    primary = ScriptedOperation("primary")
    secondary = ScriptedOperation("secondary")
    result = await execute_with_fallback([primary, secondary], FAST, sleeper=sleeper)
    assert result.response == "primary"
    assert result.fallback_used is False
    assert secondary.calls == 0


async def test_fallback_after_primary_failure(sleeper: RecordingSleeper) -> None:
    primary = ScriptedOperation(ServerError("500"))
    secondary = ScriptedOperation("secondary")
    result = await execute_with_fallback(
        [primary, secondary], FAST, providers=["openai", "google"], sleeper=sleeper
    )
    assert result.response == "secondary"
    assert result.provider == "google"
    assert result.fallback_used is True
    # The primary exhausted its retries first.
    assert primary.calls == 2
    assert result.attempts == 3


async def test_multiple_fallbacks(sleeper: RecordingSleeper) -> None:
    primary = ScriptedOperation(ServerError("500"))
    first = ScriptedOperation(RateLimitError("429"))
    second = ScriptedOperation("third time lucky")
    result = await execute_with_fallback(
        [primary, first, second],
        FAST,
        providers=["openai", "google", "anthropic"],
        models=["gpt-5", "gemini-3.7-flash", "claude-sonnet"],
        sleeper=sleeper,
    )
    assert result.response == "third time lucky"
    assert result.provider == "anthropic"
    assert result.model == "claude-sonnet"
    assert result.attempts == 5
    assert result.fallback_used is True


async def test_all_fallbacks_fail(sleeper: RecordingSleeper) -> None:
    primary = ScriptedOperation(ServerError("primary down"))
    secondary = ScriptedOperation(FakeHTTPError(429, "secondary throttled"))
    with pytest.raises(RateLimitError) as excinfo:
        await execute_with_fallback([primary, secondary], FAST, sleeper=sleeper)
    # The final, meaningful error surfaces; earlier failures stay inspectable.
    assert "secondary throttled" in str(excinfo.value)
    history = excinfo.value.attempt_history
    assert len(history) == 4
    assert isinstance(history[0].error, ServerError)


async def test_attempt_history_spans_providers(sleeper: RecordingSleeper) -> None:
    primary = ScriptedOperation(ServerError("500"))
    secondary = fail_then(ServerError("500"), times=1, value="ok")
    result = await execute_with_fallback(
        [primary, secondary], FAST, providers=["openai", "google"], sleeper=sleeper
    )
    providers = [attempt.provider for attempt in result.attempt_history]
    assert providers == ["openai", "openai", "google", "google"]
    assert [a.attempt_number for a in result.attempt_history] == [1, 2, 1, 2]
    assert result.attempt_history[-1].success is True


async def test_permission_error_does_not_retry_but_does_fall_back(
    sleeper: RecordingSleeper,
) -> None:
    primary = ScriptedOperation(FakeHTTPError(403))
    secondary = ScriptedOperation("ok")
    result = await execute_with_fallback([primary, secondary], FAST, sleeper=sleeper)
    assert result.response == "ok"
    assert primary.calls == 1  # retry decision: no
    assert result.fallback_used is True  # fallback decision: yes


async def test_authentication_error_does_not_fall_back_by_default(
    sleeper: RecordingSleeper,
) -> None:
    primary = ScriptedOperation(FakeHTTPError(401))
    secondary = ScriptedOperation("ok")
    with pytest.raises(AuthenticationError):
        await execute_with_fallback([primary, secondary], FAST, sleeper=sleeper)
    assert secondary.calls == 0


async def test_fallback_policy_is_configurable(sleeper: RecordingSleeper) -> None:
    primary = ScriptedOperation(FakeHTTPError(400))
    secondary = ScriptedOperation("ok")
    policy = FallbackPolicy(fallback_on_invalid_request=True)
    result = await execute_with_fallback(
        [primary, secondary], FAST, fallback_policy=policy, sleeper=sleeper
    )
    assert result.response == "ok"

    strict = ScriptedOperation(FakeHTTPError(403))
    with pytest.raises(PermissionError):
        await execute_with_fallback(
            [strict, ScriptedOperation("ok")],
            FAST,
            fallback_policy=FallbackPolicy(fallback_on_permission_error=False),
            sleeper=sleeper,
        )


def test_fallback_policy_defaults() -> None:
    policy = FallbackPolicy()
    assert policy.should_fallback(RateLimitError())
    assert policy.should_fallback(ServerError())
    assert policy.should_fallback(PermissionError())
    assert not policy.should_fallback(AuthenticationError())
    assert not policy.should_fallback(InvalidRequestError())
    assert not policy.should_fallback(ValueError("unknown"))


async def test_empty_operations_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least one operation"):
        await execute_with_fallback([], FAST)


async def test_missing_labels_are_none(sleeper: RecordingSleeper) -> None:
    primary = ScriptedOperation(ServerError("500"))
    secondary = ScriptedOperation("ok")
    result = await execute_with_fallback(
        [primary, secondary], FAST, providers=["openai"], sleeper=sleeper
    )
    assert result.provider is None
    assert result.model is None
