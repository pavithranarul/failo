from __future__ import annotations

import pytest

import failo
from failo import ResilientClient, RetryPolicy, resilient_call, resilient_call_sync
from failo.errors import AIError, ServerError
from tests.conftest import RecordingSleeper, ScriptedOperation, fail_then

FAST = RetryPolicy(max_attempts=2, initial_delay=1.0, jitter=False)


async def test_resilient_call_without_fallbacks(sleeper: RecordingSleeper) -> None:
    result = await resilient_call(ScriptedOperation("ok"), retry_policy=FAST, sleeper=sleeper)
    assert result.response == "ok"
    assert result.attempts == 1
    assert result.fallback_used is False


async def test_provider_metadata(sleeper: RecordingSleeper) -> None:
    result = await resilient_call(
        ScriptedOperation("ok"),
        fallbacks=[ScriptedOperation("nope")],
        retry_policy=FAST,
        providers=["openai", "google"],
        models=["gpt-5", "gemini-3.7-flash"],
        sleeper=sleeper,
    )
    assert result.provider == "openai"


async def test_model_metadata(sleeper: RecordingSleeper) -> None:
    result = await resilient_call(
        ScriptedOperation(ServerError("500")),
        fallbacks=[ScriptedOperation("ok")],
        retry_policy=FAST,
        providers=["openai", "google"],
        models=["gpt-5", "gemini-3.7-flash"],
        sleeper=sleeper,
    )
    assert result.model == "gemini-3.7-flash"
    assert result.provider == "google"


async def test_fallback_used(sleeper: RecordingSleeper) -> None:
    direct = await resilient_call(ScriptedOperation("ok"), retry_policy=FAST, sleeper=sleeper)
    assert direct.fallback_used is False

    failed_over = await resilient_call(
        ScriptedOperation(ServerError("500")),
        fallbacks=[ScriptedOperation("ok")],
        retry_policy=FAST,
        sleeper=sleeper,
    )
    assert failed_over.fallback_used is True


async def test_total_latency(sleeper: RecordingSleeper) -> None:
    async def slow() -> str:
        return "ok"

    result = await resilient_call(slow, retry_policy=FAST, sleeper=sleeper)
    assert result.total_latency >= 0.0
    assert result.total_latency >= sum(a.latency or 0.0 for a in result.attempt_history)


async def test_attempt_history_is_exposed(sleeper: RecordingSleeper) -> None:
    result = await resilient_call(
        fail_then(ServerError("500"), times=1, value="ok"), retry_policy=FAST, sleeper=sleeper
    )
    assert len(result.attempt_history) == 2
    assert result.attempts == 2
    assert result.attempt_history[0].success is False
    assert isinstance(result.attempt_history[0].error, ServerError)


async def test_resilient_client_reuses_policy(sleeper: RecordingSleeper) -> None:
    client = ResilientClient(
        retry_policy=RetryPolicy(max_attempts=3, initial_delay=1, jitter=False),
        providers=["openai", "google"],
        models=["gpt-5", "gemini-3.7-flash"],
    )
    operation = ScriptedOperation(ServerError("500"))
    result = await client.call(
        operation, fallbacks=[ScriptedOperation("ok")], sleeper=sleeper
    )
    assert operation.calls == 3
    assert result.provider == "google"
    assert result.response == "ok"


async def test_resilient_client_per_call_override(sleeper: RecordingSleeper) -> None:
    client = ResilientClient(retry_policy=RetryPolicy(max_attempts=5, jitter=False))
    operation = ScriptedOperation(ServerError("500"))
    with pytest.raises(ServerError):
        await client.call(
            operation, retry_policy=RetryPolicy(max_attempts=1), sleeper=sleeper
        )
    assert operation.calls == 1


async def test_failure_raises_normalized_error(sleeper: RecordingSleeper) -> None:
    with pytest.raises(AIError) as excinfo:
        await resilient_call(
            ScriptedOperation(ServerError("primary")),
            fallbacks=[ScriptedOperation(ServerError("secondary"))],
            retry_policy=FAST,
            sleeper=sleeper,
        )
    assert len(excinfo.value.attempt_history) == 4


def test_resilient_call_sync() -> None:
    result = resilient_call_sync(
        ScriptedOperation("ok"), retry_policy=RetryPolicy(max_attempts=1)
    )
    assert result.response == "ok"


async def test_resilient_call_sync_rejects_running_loop() -> None:
    with pytest.raises(RuntimeError, match="running event loop"):
        resilient_call_sync(ScriptedOperation("ok"))


def test_public_api_surface() -> None:
    assert failo.__version__ == "0.1.0"
    for name in ("resilient_call", "ResilientClient", "RetryPolicy", "FallbackPolicy"):
        assert hasattr(failo, name)


def test_no_provider_sdk_imports() -> None:
    """The package source must never import a provider SDK or HTTP client."""
    import pathlib
    import re

    package = pathlib.Path(failo.__file__).parent
    forbidden = re.compile(
        r"^\s*(?:from|import)\s+"
        r"(openai|anthropic|google|langchain|litellm|httpx|requests|aiohttp)\b",
        re.MULTILINE,
    )
    for module in package.glob("*.py"):
        assert not forbidden.search(module.read_text()), module.name


def test_public_api_is_frozen_for_v0_1() -> None:
    """The v0.1 export surface must not shrink or be renamed."""
    expected = {
        "resilient_call",
        "resilient_call_sync",
        "ResilientClient",
        "RetryPolicy",
        "FallbackPolicy",
        "ErrorClassifier",
        "classify_error",
        "execute_with_retry",
        "execute_with_fallback",
        "Attempt",
        "ResilientResult",
        "RetryOutcome",
        "AIError",
        "RateLimitError",
        "AuthenticationError",
        "PermissionError",
        "InvalidRequestError",
        "ServerError",
        "TimeoutError",
        "ConnectionError",
        "__version__",
    }
    assert expected <= set(failo.__all__)
    for name in expected:
        assert hasattr(failo, name), name


def test_error_hierarchy_is_intact() -> None:
    from failo.errors import (
        AuthenticationError,
        ConnectionError,
        InvalidRequestError,
        PermissionError,
        RateLimitError,
        ServerError,
        TimeoutError,
    )

    subclasses = (
        RateLimitError,
        AuthenticationError,
        PermissionError,
        InvalidRequestError,
        ServerError,
        TimeoutError,
        ConnectionError,
    )
    for error_type in subclasses:
        assert issubclass(error_type, AIError)
        assert not issubclass(error_type, tuple(set(subclasses) - {error_type}))
