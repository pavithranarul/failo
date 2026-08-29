# Failo

[![CI](https://github.com/pavithranarul/failo/actions/workflows/tests.yml/badge.svg)](https://github.com/pavithranarul/failo/actions/workflows/tests.yml)
[![PyPI](https://img.shields.io/pypi/v/failo.svg)](https://pypi.org/project/failo/)
[![Python](https://img.shields.io/pypi/pyversions/failo.svg)](https://pypi.org/project/failo/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
Reliable AI calls with retry, fallback, and failover support.

Failo is a lightweight, provider-independent resilience layer for AI applications. It sits on top of your existing AI SDK and handles transient failures without forcing you to replace your provider client.

```text
Application
     |
     v
   Failo
     |
 +---+---+---+
 |   |   |   |
OpenAI Gemini Claude Other
```

```text
                    Application
                         |
                         v
                  +--------------+
                  |    Failo     |
                  |              |
                  | Retry        |
                  | Classification
                  | Fallback     |
                  | Policies     |
                  +------+-------+
                         |
              Existing provider callable
                         |
          +--------------+--------------+
          |              |              |
       OpenAI         Gemini         Anthropic
```

- **Zero runtime dependencies.** Standard library only.
- **No SDK lock-in.** Failo never imports `openai`, `anthropic`, `google-genai`, LangChain, or LiteLLM. You keep your client; you hand Failo a callable.
- **Async-first**, with a thin blocking helper for sync codebases.
- **Typed**, with `py.typed` shipped.

> **Status: v0.1.0, early release.** The API is small and tested, but it has not been battle-tested in production. Treat it as alpha.

## Install

```bash
pip install failo
```

Requires Python 3.10+.

## Quick start

```python
from failo import resilient_call

async def call_openai():
    return await openai_client.responses.create(model="gpt-5", input="Hello")

result = await resilient_call(primary=call_openai)

print(result.response)
```

`primary` is any zero-argument callable — async or sync. Failo calls it, times it, classifies whatever it raises, and decides whether to try again.

## Fallback across providers

```python
from failo import RetryPolicy, resilient_call

async def call_openai():
    return await openai_client.responses.create(model="gpt-5", input=prompt)

async def call_gemini():
    return await gemini_client.aio.models.generate_content(model="gemini-3.7-flash", contents=prompt)

async def call_anthropic():
    return await anthropic_client.messages.create(model="claude-sonnet", messages=messages)

result = await resilient_call(
    primary=call_openai,
    fallbacks=[call_gemini, call_anthropic],
    retry_policy=RetryPolicy(max_attempts=3),
    providers=["openai", "google", "anthropic"],
    models=["gpt-5", "gemini-3.7-flash", "claude-sonnet"],
)

print(result.provider)       # which provider actually answered
print(result.fallback_used)  # True if the primary did not
```

Execution order:

```text
Primary
  |
  +-- retry
  +-- retry
  |
  X
  |
Fallback 1
  |
  +-- retry
  |
  X
  |
Fallback 2
  |
  +-- success
```

Each provider receives its own full retry budget before Failo moves to the next fallback.

## Reusing a policy

```python
from failo import ResilientClient, RetryPolicy

failo = ResilientClient(retry_policy=RetryPolicy(max_attempts=3, initial_delay=1))

result = await failo.call(primary=openai_call, fallbacks=[gemini_call])
```

## Sync helper

```python
from failo import resilient_call_sync

result = resilient_call_sync(primary=call_provider)
```

`resilient_call_sync` is an `asyncio.run` shim over the same engine — there is no second retry implementation. It raises `RuntimeError` if an event loop is already running in the thread; inside async code, await `resilient_call` instead.

## What Failo handles

| Situation | Behavior |
| --- | --- |
| **Rate limits** (429) | Retry with backoff; honors `retry_after` when the provider exposes it, capped by `max_retry_after` |
| **Server errors** (500/502/503/504) | Retry with exponential backoff and jitter |
| **Timeouts** (408, `asyncio.TimeoutError`) | Retry |
| **Connection failures** (`OSError`, `ConnectionError`) | Retry |
| **Bad request** (400) | No retry, no fallback |
| **Auth failure** (401) | No retry, no fallback |
| **Permission denied** (403) | No retry, but **does** fail over to the next provider |
| **Unknown errors** | No retry, no fallback — surfaced as `AIError` |
| **Execution metadata** | Provider, model, attempt count, latency, full attempt history |

### Retry

```python
RetryPolicy(
    max_attempts=3,          # total executions, including the first
    initial_delay=1.0,
    max_delay=30.0,          # bounds computed backoff
    exponential_backoff=True,
    jitter=True,             # full jitter: uniform in [0, delay]
    respect_retry_after=True,
    max_retry_after=60.0,    # bounds a server-supplied Retry-After; None = no cap
)
```

Backoff without jitter: attempt 1 waits 1s, attempt 2 waits 2s, attempt 3 waits 4s, capped at `max_delay`.

When an exception exposes `retry_after` (attribute or `Retry-After` header), Failo prefers it over the computed delay and does not jitter it — the server already said when to come back.

`max_retry_after` keeps that from becoming a denial of service against yourself: a provider answering `Retry-After: 3600` should not park your request for an hour.

| `Retry-After` | `max_retry_after` | Delay |
| --- | --- | --- |
| 5 | 60.0 (default) | 5s |
| 120 | 60.0 | 60s |
| 3600 | 60.0 | 60s |
| 3600 | `None` | 3600s |

The two bounds are separate on purpose: `max_delay` caps computed backoff, `max_retry_after` caps the server's hint. Set `max_retry_after=None` to honor the provider unconditionally, or `respect_retry_after=False` to ignore the hint and always use backoff.

### Retry is not the same decision as fallback

This distinction is deliberate:

```text
403
 |
 +-- retry?    NO   (the same call will fail the same way)
 |
 +-- fallback? YES  (a different provider may well allow it)
```

`RetryPolicy` answers the first question, `FallbackPolicy` the second:

```python
from failo import FallbackPolicy

FallbackPolicy(
    fallback_on_rate_limit=True,
    fallback_on_server_error=True,
    fallback_on_timeout=True,
    fallback_on_connection_error=True,
    fallback_on_permission_error=True,
    fallback_on_authentication_error=False,
    fallback_on_invalid_request=False,
    fallback_on_unknown_error=False,
)
```

### Error classification

Failo normalizes provider exceptions into its own hierarchy, without importing any SDK:

```text
AIError
├── RateLimitError
├── AuthenticationError
├── PermissionError
├── InvalidRequestError
├── ServerError
├── TimeoutError
└── ConnectionError
```

Classification is best-effort and generic: an HTTP-style status code is read from common attributes (`status_code`, `status`, `http_status`, `code`, or a nested `response`), stdlib timeout/connection errors are recognized, and a few unambiguous class names (`RateLimitExceeded`, `DeadlineExceeded`) are matched. Anything unrecognized becomes a bare `AIError`, which no default policy retries — unknown failures stay safe.

You can extend it without touching global state:

```python
from failo import ErrorClassifier, RetryPolicy
from failo.errors import AIError, RateLimitError

def vendor_rule(error: BaseException) -> AIError | None:
    if type(error).__name__ == "VendorQuotaExceeded":
        return RateLimitError("vendor quota", original=error)
    return None

policy = RetryPolicy(classifier=ErrorClassifier().with_rules(vendor_rule))
```

### Execution metadata

```python
result.response         # exactly what your callable returned
result.provider         # label of the provider that succeeded
result.model            # label of the model that succeeded
result.attempts         # total executions across all providers
result.fallback_used    # True if the primary did not produce the result
result.total_latency    # seconds, time.monotonic()-based
result.attempt_history  # list[Attempt]: provider, model, attempt_number,
                        # success, error, latency
```

When every provider fails, Failo raises the final normalized error, chained to the original provider exception via `__cause__`, with the whole history attached:

```python
from failo.errors import AIError

try:
    result = await resilient_call(primary=call_openai, fallbacks=[call_gemini])
except AIError as error:
    print(error.__cause__)        # the original SDK exception
    print(error.attempt_history)  # every attempt against every provider
```

## Testing your own retry logic

Every entry point accepts a `sleeper` — an async callable used instead of `asyncio.sleep`, so your tests observe delays instead of waiting for them:

```python
delays = []

async def fake_sleep(seconds: float) -> None:
    delays.append(seconds)

await resilient_call(primary=flaky, retry_policy=RetryPolicy(jitter=False), sleeper=fake_sleep)
assert delays == [1.0, 2.0]
```

## Planned, not implemented

These are **not** in v0.1.0):

- Provider adapters (OpenAI / Gemini / Anthropic) as optional extras
- Circuit breakers
- Provider health tracking
- Client-side rate limiting
- OpenTelemetry tracing and metrics
- Streaming-aware retries
- Hedged / parallel requests

The core is designed so these can land without breaking the public API.

## Development

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

pytest
ruff check .
mypy src
```

## License

MIT — see [LICENSE](LICENSE).
