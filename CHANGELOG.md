# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned

* Optional provider adapters (OpenAI, Gemini, Anthropic)
* Circuit breaker and provider health tracking
* Client-side rate limiting
* OpenTelemetry tracing and metrics

## [0.1.0] - 2026-08-29

### Added

* `resilient_call`, `resilient_call_sync`, and `ResilientClient` public API.
* Async retry engine with exponential backoff, full jitter, and `max_attempts` counted as total executions.
* `Retry-After` support: a delay exposed by the exception (attribute or `Retry-After` header) takes precedence over computed backoff, is never jittered, and is bounded by `RetryPolicy.max_retry_after` (default 60s, `None` to honor the provider unconditionally).
* Fallback engine supporting an unlimited number of fallback operations, with attempt history preserved across providers.
* `RetryPolicy` and `FallbackPolicy` as independent decisions, so a 403 can skip retries while still failing over.
* Normalized error hierarchy (`AIError` and seven subclasses) plus a generic, extensible `ErrorClassifier` that imports no provider SDK.
* Execution metadata through `Attempt` and `ResilientResult`.
* Injectable `sleeper` for fast, deterministic tests.
* Zero runtime dependencies; `py.typed` shipped.