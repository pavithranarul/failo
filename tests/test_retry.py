from __future__ import annotations

import asyncio

import pytest

from failo import RetryPolicy
from failo.errors import (
    AIError,
    AuthenticationError,
    ConnectionError,
    InvalidRequestError,
    PermissionError,
    RateLimitError,
    ServerError,
    TimeoutError,
)
from failo.retry import execute_with_retry
from tests.conftest import FakeHTTPError, RecordingSleeper, ScriptedOperation, fail_then

FAST = RetryPolicy(max_attempts=3, initial_delay=1.0, jitter=False)


async def test_success_on_first_attempt(sleeper: RecordingSleeper) -> None:
    operation = ScriptedOperation("ok")
    outcome = await execute_with_retry(operation, FAST, sleeper=sleeper)
    assert outcome.response == "ok"
    assert operation.calls == 1
    assert sleeper.delays == []


@pytest.mark.parametrize("status", [429, 500, 502, 503, 504])
async def test_retry_on_rate_limit_and_server_errors(
    status: int, sleeper: RecordingSleeper
) -> None:
    operation = fail_then(FakeHTTPError(status), times=1, value="recovered")
    outcome = await execute_with_retry(operation, FAST, sleeper=sleeper)
    assert outcome.response == "recovered"
    assert operation.calls == 2


async def test_retry_on_rate_limit(sleeper: RecordingSleeper) -> None:
    operation = fail_then(RateLimitError("429"), times=2, value="ok")
    assert (await execute_with_retry(operation, FAST, sleeper=sleeper)).response == "ok"
    assert operation.calls == 3


async def test_retry_on_server_error(sleeper: RecordingSleeper) -> None:
    operation = fail_then(ServerError("500"), times=1, value="ok")
    assert (await execute_with_retry(operation, FAST, sleeper=sleeper)).response == "ok"


async def test_retry_on_timeout(sleeper: RecordingSleeper) -> None:
    operation = fail_then(asyncio.TimeoutError(), times=1, value="ok")
    outcome = await execute_with_retry(operation, FAST, sleeper=sleeper)
    assert outcome.response == "ok"
    assert operation.calls == 2


async def test_retry_on_connection_error(sleeper: RecordingSleeper) -> None:
    operation = fail_then(OSError("connection reset"), times=1, value="ok")
    outcome = await execute_with_retry(operation, FAST, sleeper=sleeper)
    assert outcome.response == "ok"
    assert operation.calls == 2


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (FakeHTTPError(400), InvalidRequestError),
        (FakeHTTPError(401), AuthenticationError),
        (FakeHTTPError(403), PermissionError),
    ],
)
async def test_no_retry_on_client_errors(
    error: Exception, expected: type[AIError], sleeper: RecordingSleeper
) -> None:
    operation = ScriptedOperation(error, "never reached")
    with pytest.raises(expected):
        await execute_with_retry(operation, FAST, sleeper=sleeper)
    assert operation.calls == 1
    assert sleeper.delays == []


async def test_no_retry_on_invalid_request(sleeper: RecordingSleeper) -> None:
    operation = ScriptedOperation(InvalidRequestError("bad body"), "never")
    with pytest.raises(InvalidRequestError):
        await execute_with_retry(operation, FAST, sleeper=sleeper)
    assert operation.calls == 1


async def test_no_retry_on_authentication_error(sleeper: RecordingSleeper) -> None:
    operation = ScriptedOperation(AuthenticationError("bad key"), "never")
    with pytest.raises(AuthenticationError):
        await execute_with_retry(operation, FAST, sleeper=sleeper)
    assert operation.calls == 1


async def test_no_retry_on_permission_error(sleeper: RecordingSleeper) -> None:
    operation = ScriptedOperation(PermissionError("forbidden"), "never")
    with pytest.raises(PermissionError):
        await execute_with_retry(operation, FAST, sleeper=sleeper)
    assert operation.calls == 1


async def test_no_retry_on_unknown_error(sleeper: RecordingSleeper) -> None:
    operation = ScriptedOperation(ValueError("mystery"), "never")
    with pytest.raises(AIError):
        await execute_with_retry(operation, FAST, sleeper=sleeper)
    assert operation.calls == 1


async def test_max_attempts(sleeper: RecordingSleeper) -> None:
    operation = ScriptedOperation(ServerError("500"))
    policy = RetryPolicy(max_attempts=4, initial_delay=1.0, jitter=False)
    with pytest.raises(ServerError):
        await execute_with_retry(operation, policy, sleeper=sleeper)
    # max_attempts counts total executions, not extra retries.
    assert operation.calls == 4
    assert len(sleeper.delays) == 3


async def test_original_error_is_preserved_when_retries_are_exhausted(
    sleeper: RecordingSleeper,
) -> None:
    raw = FakeHTTPError(503, "upstream down")
    operation = ScriptedOperation(raw)
    with pytest.raises(ServerError) as excinfo:
        await execute_with_retry(operation, FAST, sleeper=sleeper)
    assert excinfo.value.original is raw
    assert excinfo.value.__cause__ is raw
    assert len(excinfo.value.attempt_history) == 3


async def test_already_normalized_error_is_not_chained_to_itself(
    sleeper: RecordingSleeper,
) -> None:
    """A user-raised AIError re-raises as-is, never with __cause__ == itself."""
    raised = ServerError("upstream down")
    with pytest.raises(ServerError) as excinfo:
        await execute_with_retry(ScriptedOperation(raised), FAST, sleeper=sleeper)
    assert excinfo.value is raised
    assert excinfo.value.__cause__ is None
    assert len(excinfo.value.attempt_history) == 3


async def test_exponential_backoff(sleeper: RecordingSleeper) -> None:
    operation = ScriptedOperation(ServerError("500"))
    policy = RetryPolicy(max_attempts=4, initial_delay=1.0, jitter=False)
    with pytest.raises(ServerError):
        await execute_with_retry(operation, policy, sleeper=sleeper)
    assert sleeper.delays == [1.0, 2.0, 4.0]


async def test_backoff_is_capped_by_max_delay(sleeper: RecordingSleeper) -> None:
    operation = ScriptedOperation(ServerError("500"))
    policy = RetryPolicy(max_attempts=5, initial_delay=1.0, max_delay=3.0, jitter=False)
    with pytest.raises(ServerError):
        await execute_with_retry(operation, policy, sleeper=sleeper)
    assert sleeper.delays == [1.0, 2.0, 3.0, 3.0]


async def test_backoff_can_be_disabled(sleeper: RecordingSleeper) -> None:
    operation = ScriptedOperation(ServerError("500"))
    policy = RetryPolicy(
        max_attempts=4, initial_delay=2.0, exponential_backoff=False, jitter=False
    )
    with pytest.raises(ServerError):
        await execute_with_retry(operation, policy, sleeper=sleeper)
    assert sleeper.delays == [2.0, 2.0, 2.0]


async def test_jitter_can_be_disabled(sleeper: RecordingSleeper) -> None:
    policy = RetryPolicy(max_attempts=3, initial_delay=1.0, jitter=False)
    assert [policy.get_delay(n) for n in (1, 2, 3)] == [1.0, 2.0, 4.0]


def test_full_jitter_stays_within_bounds() -> None:
    policy = RetryPolicy(initial_delay=1.0, jitter=True)
    delays = [policy.get_delay(3) for _ in range(200)]
    assert all(0.0 <= delay <= 4.0 for delay in delays)
    assert len(set(delays)) > 1  # actually randomized


async def test_retry_after(sleeper: RecordingSleeper) -> None:
    error = FakeHTTPError(429, retry_after=12)
    operation = fail_then(error, times=1, value="ok")
    policy = RetryPolicy(max_attempts=2, initial_delay=1.0, jitter=True)
    outcome = await execute_with_retry(operation, policy, sleeper=sleeper)
    assert outcome.response == "ok"
    # retry_after wins over exponential backoff and is not jittered.
    assert sleeper.delays == [12.0]


async def test_retry_after_below_cap_is_used_verbatim(sleeper: RecordingSleeper) -> None:
    operation = fail_then(FakeHTTPError(429, retry_after=5), times=1, value="ok")
    policy = RetryPolicy(max_attempts=2, initial_delay=1.0, jitter=True, max_retry_after=60.0)
    await execute_with_retry(operation, policy, sleeper=sleeper)
    assert sleeper.delays == [5.0]


@pytest.mark.parametrize("retry_after", [120, 3600])
async def test_retry_after_above_cap_is_clamped(
    retry_after: int, sleeper: RecordingSleeper
) -> None:
    operation = fail_then(FakeHTTPError(429, retry_after=retry_after), times=1, value="ok")
    policy = RetryPolicy(max_attempts=2, initial_delay=1.0, jitter=False, max_retry_after=60.0)
    await execute_with_retry(operation, policy, sleeper=sleeper)
    assert sleeper.delays == [60.0]


async def test_retry_after_is_uncapped_when_max_retry_after_is_none(
    sleeper: RecordingSleeper,
) -> None:
    operation = fail_then(FakeHTTPError(429, retry_after=3600), times=1, value="ok")
    policy = RetryPolicy(max_attempts=2, initial_delay=1.0, jitter=False, max_retry_after=None)
    await execute_with_retry(operation, policy, sleeper=sleeper)
    assert sleeper.delays == [3600.0]


def test_retry_after_is_never_jittered() -> None:
    """A clamped or verbatim retry-after must be exact, not randomized."""
    policy = RetryPolicy(initial_delay=1.0, jitter=True, max_retry_after=60.0)
    assert {policy.get_delay(1, retry_after=5) for _ in range(100)} == {5.0}
    assert {policy.get_delay(3, retry_after=900) for _ in range(100)} == {60.0}

    uncapped = RetryPolicy(initial_delay=1.0, jitter=True, max_retry_after=None)
    assert {uncapped.get_delay(1, retry_after=3600) for _ in range(100)} == {3600.0}


def test_max_retry_after_does_not_affect_backoff_delays() -> None:
    """max_retry_after bounds only the server hint; max_delay bounds backoff."""
    policy = RetryPolicy(initial_delay=1.0, max_delay=30.0, jitter=False, max_retry_after=2.0)
    assert policy.get_delay(4) == 8.0
    assert policy.get_delay(1, retry_after=10) == 2.0


def test_max_retry_after_default_is_sixty_seconds() -> None:
    assert RetryPolicy().max_retry_after == 60.0


async def test_retry_after_can_be_ignored(sleeper: RecordingSleeper) -> None:
    operation = fail_then(FakeHTTPError(429, retry_after=12), times=1, value="ok")
    policy = RetryPolicy(max_attempts=2, initial_delay=1.0, jitter=False, respect_retry_after=False)
    await execute_with_retry(operation, policy, sleeper=sleeper)
    assert sleeper.delays == [1.0]


def test_invalid_max_retry_after() -> None:
    with pytest.raises(ValueError, match="max_retry_after"):
        RetryPolicy(max_retry_after=-1)


async def test_sync_callables_are_supported(sleeper: RecordingSleeper) -> None:
    calls = {"n": 0}

    def sync_operation() -> str:
        calls["n"] += 1
        if calls["n"] == 1:
            raise ServerError("500")
        return "ok"

    outcome = await execute_with_retry(sync_operation, FAST, sleeper=sleeper)
    assert outcome.response == "ok"


async def test_custom_retryable_errors(sleeper: RecordingSleeper) -> None:
    policy = RetryPolicy(
        max_attempts=3,
        initial_delay=1.0,
        jitter=False,
        retryable_errors=(PermissionError,),
    )
    operation = fail_then(FakeHTTPError(403), times=1, value="ok")
    assert (await execute_with_retry(operation, policy, sleeper=sleeper)).response == "ok"

    never = ScriptedOperation(ServerError("500"))
    with pytest.raises(ServerError):
        await execute_with_retry(never, policy, sleeper=sleeper)
    assert never.calls == 1


async def test_attempt_history_records_each_try(sleeper: RecordingSleeper) -> None:
    operation = fail_then(ServerError("500"), times=1, value="ok")
    outcome = await execute_with_retry(
        operation, FAST, provider="openai", model="gpt-5", sleeper=sleeper
    )
    assert [a.attempt_number for a in outcome.attempts] == [1, 2]
    assert [a.success for a in outcome.attempts] == [False, True]
    assert all(a.provider == "openai" and a.model == "gpt-5" for a in outcome.attempts)
    assert all(a.latency is not None and a.latency >= 0 for a in outcome.attempts)


def test_invalid_policy_configuration() -> None:
    with pytest.raises(ValueError, match="max_attempts"):
        RetryPolicy(max_attempts=0)
    with pytest.raises(ValueError, match="initial_delay"):
        RetryPolicy(initial_delay=-1)


def test_default_retryable_classification() -> None:
    policy = RetryPolicy()
    assert policy.should_retry(RateLimitError())
    assert policy.should_retry(ServerError())
    assert policy.should_retry(TimeoutError())
    assert policy.should_retry(ConnectionError())
    assert not policy.should_retry(AuthenticationError())
    assert not policy.should_retry(PermissionError())
    assert not policy.should_retry(InvalidRequestError())
    assert not policy.should_retry(ValueError("unknown"))
