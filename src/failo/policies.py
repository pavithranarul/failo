"""Retry and fallback policies.

The two decisions Failo makes are deliberately independent:

* ``RetryPolicy`` answers "should I call this *same* operation again?"
* ``FallbackPolicy`` answers "should I move on to the *next* provider?"

A 403 is the canonical example -- never worth retrying, often worth failing
over.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from .errors import (
    AIError,
    AuthenticationError,
    ConnectionError,
    ErrorClassifier,
    InvalidRequestError,
    PermissionError,
    RateLimitError,
    ServerError,
    TimeoutError,
    classify_error,
)

__all__ = [
    "DEFAULT_RETRYABLE_ERRORS",
    "FallbackPolicy",
    "RetryPolicy",
]

#: Transient failures that are worth trying again against the same provider.
DEFAULT_RETRYABLE_ERRORS: tuple[type[AIError], ...] = (
    RateLimitError,
    ServerError,
    TimeoutError,
    ConnectionError,
)


@dataclass
class RetryPolicy:
    """How a single operation is retried.

    Attributes:
        max_attempts: Total executions, *including* the first one.
        initial_delay: Delay in seconds before the second attempt.
        max_delay: Upper bound applied to every computed delay.
        exponential_backoff: Double the delay per attempt when ``True``.
        jitter: Apply full jitter (uniform in ``[0, delay]``) when ``True``.
        retryable_errors: Normalized error types considered transient.
        respect_retry_after: Prefer ``error.retry_after`` over backoff.
        max_retry_after: Upper bound for a server-supplied ``retry_after``.
            ``None`` honours the server's delay however long it is.
        classifier: Optional custom error classifier.
    """

    max_attempts: int = 3
    initial_delay: float = 1.0
    max_delay: float = 30.0
    exponential_backoff: bool = True
    jitter: bool = True
    retryable_errors: tuple[type[AIError], ...] = DEFAULT_RETRYABLE_ERRORS
    respect_retry_after: bool = True
    max_retry_after: float | None = 60.0
    classifier: ErrorClassifier | None = None

    def __post_init__(self) -> None:
        """Validate the numeric configuration."""
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        if self.initial_delay < 0:
            raise ValueError("initial_delay must be >= 0")
        if self.max_delay < 0:
            raise ValueError("max_delay must be >= 0")
        if self.max_retry_after is not None and self.max_retry_after < 0:
            raise ValueError("max_retry_after must be >= 0 or None")

    def classify(self, error: Exception) -> AIError:
        """Normalize ``error`` using this policy's classifier."""
        return classify_error(error, self.classifier)

    def should_retry(self, error: Exception) -> bool:
        """Return whether ``error`` is transient enough to try again.

        Unknown errors are never retryable: they classify to the base
        :class:`~failo.errors.AIError`, which is not in ``retryable_errors``.
        """
        return isinstance(self.classify(error), tuple(self.retryable_errors))

    def get_delay(self, attempt_number: int, *, retry_after: float | None = None) -> float:
        """Seconds to wait *after* attempt ``attempt_number`` (1-based).

        Attempt 1 yields ``initial_delay``, attempt 2 twice that, and so on,
        capped at ``max_delay``.  A server-supplied ``retry_after`` wins over
        the computed backoff and is never jittered -- the server already told
        us when to come back -- but it is capped at ``max_retry_after`` so a
        provider cannot park the caller for an hour.  ``max_delay`` does not
        apply to ``retry_after``; the two bounds are deliberately separate.
        """
        if retry_after is not None and self.respect_retry_after:
            delay = max(0.0, retry_after)
            if self.max_retry_after is not None:
                delay = min(delay, self.max_retry_after)
            return delay

        exponent = max(0, attempt_number - 1)
        if self.exponential_backoff:
            delay = self.initial_delay * (2.0**exponent)
        else:
            delay = self.initial_delay
        delay = min(delay, self.max_delay)
        if self.jitter:
            delay = random.uniform(0.0, delay)  # noqa: S311 - not cryptographic
        return delay


@dataclass
class FallbackPolicy:
    """Which failures are allowed to move on to the next provider.

    Defaults cover the cases where another provider plausibly helps.  A
    malformed request or a bad key is the caller's problem, so those do not
    fail over by default; an unclassified error stays put too, because Failo
    cannot tell whether the call had a side effect.
    """

    fallback_on_rate_limit: bool = True
    fallback_on_server_error: bool = True
    fallback_on_timeout: bool = True
    fallback_on_connection_error: bool = True
    fallback_on_permission_error: bool = True
    fallback_on_authentication_error: bool = False
    fallback_on_invalid_request: bool = False
    fallback_on_unknown_error: bool = False
    classifier: ErrorClassifier | None = field(default=None)

    def _flag_for(self, error: AIError) -> bool:
        """Map a normalized error onto its configured flag."""
        if isinstance(error, RateLimitError):
            return self.fallback_on_rate_limit
        if isinstance(error, ServerError):
            return self.fallback_on_server_error
        if isinstance(error, TimeoutError):
            return self.fallback_on_timeout
        if isinstance(error, ConnectionError):
            return self.fallback_on_connection_error
        if isinstance(error, PermissionError):
            return self.fallback_on_permission_error
        if isinstance(error, AuthenticationError):
            return self.fallback_on_authentication_error
        if isinstance(error, InvalidRequestError):
            return self.fallback_on_invalid_request
        return self.fallback_on_unknown_error

    def should_fallback(self, error: Exception) -> bool:
        """Return whether the next provider should be tried after ``error``."""
        return self._flag_for(classify_error(error, self.classifier))
