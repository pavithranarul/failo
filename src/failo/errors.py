"""Normalized error model for Failo.

Failo never imports a provider SDK.  Instead it *classifies* whatever
exception a user-supplied callable raised into a small, stable hierarchy
that retry and fallback policies can reason about.

Classification is driven by :class:`ErrorClassifier`, an immutable object
holding an ordered tuple of rules.  Provider-specific adapters can be added
later by building a new classifier with extra rules -- no module-level
registry is mutated, so there is no global mutable state.
"""

from __future__ import annotations

import asyncio
import builtins
import dataclasses
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - import cycle avoidance only
    from .models import Attempt

__all__ = [
    "AIError",
    "AuthenticationError",
    "ConnectionError",
    "DEFAULT_CLASSIFIER",
    "ErrorClassifier",
    "InvalidRequestError",
    "PermissionError",
    "RateLimitError",
    "STATUS_CODE_MAP",
    "ServerError",
    "TimeoutError",
    "classify_error",
    "extract_retry_after",
    "extract_status_code",
]


class AIError(Exception):
    """Base exception for Failo.

    Args:
        message: Human readable description.
        status_code: HTTP-style status code when one could be extracted.
        retry_after: Server-supplied delay in seconds, when advertised.
        original: The provider exception this error was classified from.
    """

    def __init__(  # noqa: D107 - documented in the class docstring
        self,
        message: str = "",
        *,
        status_code: int | None = None,
        retry_after: float | None = None,
        original: BaseException | None = None,
    ) -> None:
        super().__init__(message or self.__class__.__doc__ or self.__class__.__name__)
        self.status_code = status_code
        self.retry_after = retry_after
        self.original = original
        # Populated by the retry/fallback engines so a caller catching the
        # final error can still inspect everything that was tried.
        self.attempt_history: list[Attempt] = []


class RateLimitError(AIError):
    """The AI provider rate-limited the request."""


class AuthenticationError(AIError):
    """Authentication failed."""


class PermissionError(AIError):  # noqa: A001 - deliberate normalized name
    """The request was not permitted."""


class InvalidRequestError(AIError):
    """The request was invalid."""


class ServerError(AIError):
    """The AI provider returned a server-side error."""


class TimeoutError(AIError):  # noqa: A001 - deliberate normalized name
    """The AI request timed out."""


class ConnectionError(AIError):  # noqa: A001 - deliberate normalized name
    """The connection to the AI provider failed."""


STATUS_CODE_MAP: Mapping[int, type[AIError]] = {
    400: InvalidRequestError,
    401: AuthenticationError,
    403: PermissionError,
    408: TimeoutError,
    429: RateLimitError,
    500: ServerError,
    502: ServerError,
    503: ServerError,
    504: ServerError,
}

#: Attributes commonly used by HTTP clients to expose a status code.
_STATUS_ATTRS: tuple[str, ...] = ("status_code", "status", "http_status", "code")

#: Attributes commonly used to expose a retry-after hint.
_RETRY_AFTER_ATTRS: tuple[str, ...] = ("retry_after", "retry_after_seconds", "retry_delay")

#: Objects that may nest the real status code (``error.response.status_code``).
_NESTED_ATTRS: tuple[str, ...] = ("response", "resp")


def _coerce_int(value: Any) -> int | None:
    """Return ``value`` as an int when it plausibly is one, else ``None``."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def _coerce_float(value: Any) -> float | None:
    """Return ``value`` as a non-negative float when possible, else ``None``."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        number = float(value)
    elif isinstance(value, str):
        try:
            number = float(value.strip())
        except ValueError:
            return None
    else:
        return None
    return number if number >= 0 else None


def _headers_of(obj: Any) -> Mapping[str, Any] | None:
    """Return a mapping of headers hanging off ``obj``, if any."""
    headers = getattr(obj, "headers", None)
    return headers if isinstance(headers, Mapping) else None


def extract_status_code(error: BaseException) -> int | None:
    """Best-effort extraction of an HTTP status code from ``error``.

    Looks at common attribute names on the exception itself and on a nested
    ``response`` object.  Returns ``None`` when nothing sensible is found.
    """
    candidates: list[Any] = [error]
    for name in _NESTED_ATTRS:
        nested = getattr(error, name, None)
        if nested is not None:
            candidates.append(nested)

    for candidate in candidates:
        for attr in _STATUS_ATTRS:
            status = _coerce_int(getattr(candidate, attr, None))
            if status is not None and 100 <= status <= 599:
                return status
    return None


def extract_retry_after(error: BaseException) -> float | None:
    """Best-effort extraction of a retry-after delay (seconds) from ``error``.

    Checks common attribute names and a ``Retry-After`` header on the
    exception or its nested response.  Only non-negative numeric values are
    accepted; HTTP-date style values are ignored in v0.1.
    """
    candidates: list[Any] = [error]
    for name in _NESTED_ATTRS:
        nested = getattr(error, name, None)
        if nested is not None:
            candidates.append(nested)

    for candidate in candidates:
        for attr in _RETRY_AFTER_ATTRS:
            delay = _coerce_float(getattr(candidate, attr, None))
            if delay is not None:
                return delay
        headers = _headers_of(candidate)
        if headers is not None:
            for key, value in headers.items():
                if isinstance(key, str) and key.lower() == "retry-after":
                    delay = _coerce_float(value)
                    if delay is not None:
                        return delay
    return None


#: A rule maps a raw exception to a normalized error, or declines with ``None``.
ClassifierRule = Callable[[BaseException], AIError | None]


def _rule_passthrough(error: BaseException) -> AIError | None:
    """Return ``error`` unchanged when it is already normalized."""
    return error if isinstance(error, AIError) else None


def _rule_status_code(error: BaseException) -> AIError | None:
    """Classify from an HTTP-style status code when one can be extracted."""
    status = extract_status_code(error)
    if status is None:
        return None
    error_type = STATUS_CODE_MAP.get(status)
    if error_type is None:
        # Unknown 5xx is still server-side; anything else stays unclassified
        # so it does not silently become retryable.
        if 500 <= status <= 599:
            error_type = ServerError
        else:
            return None
    return error_type(
        f"{error_type.__name__} (status {status}): {error}",
        status_code=status,
        retry_after=extract_retry_after(error),
        original=error,
    )


def _rule_builtin_timeout(error: BaseException) -> AIError | None:
    """Classify stdlib timeout errors (``asyncio.TimeoutError`` included)."""
    if isinstance(error, (builtins.TimeoutError, asyncio.TimeoutError)):
        return TimeoutError(
            f"Request timed out: {error}",
            retry_after=extract_retry_after(error),
            original=error,
        )
    return None


def _rule_builtin_connection(error: BaseException) -> AIError | None:
    """Classify stdlib connection/socket failures."""
    if isinstance(error, (builtins.ConnectionError, OSError)):
        return ConnectionError(
            f"Connection to the provider failed: {error}",
            retry_after=extract_retry_after(error),
            original=error,
        )
    return None


def _rule_duck_typed_name(error: BaseException) -> AIError | None:
    """Classify by exception class name for SDKs without status codes.

    Deliberately conservative: only unambiguous names are matched, and the
    result is always one of the normalized types.
    """
    name = type(error).__name__.lower()
    message = str(error) or name
    retry_after = extract_retry_after(error)
    if "ratelimit" in name or "toomanyrequests" in name:
        return RateLimitError(message, retry_after=retry_after, original=error)
    if "timeout" in name or "deadlineexceeded" in name:
        return TimeoutError(message, retry_after=retry_after, original=error)
    if "connection" in name:
        return ConnectionError(str(error) or name, original=error)
    return None


DEFAULT_RULES: tuple[ClassifierRule, ...] = (
    _rule_passthrough,
    _rule_status_code,
    _rule_builtin_timeout,
    _rule_builtin_connection,
    _rule_duck_typed_name,
)


@dataclasses.dataclass(frozen=True)
class ErrorClassifier:
    """Immutable, ordered collection of classification rules.

    The first rule that returns an :class:`AIError` wins.  When every rule
    declines, the exception is wrapped in a plain :class:`AIError`, which is
    *not* retryable by default -- unknown failures must stay safe.
    """

    rules: tuple[ClassifierRule, ...] = DEFAULT_RULES

    def with_rules(self, *rules: ClassifierRule, prepend: bool = True) -> ErrorClassifier:
        """Return a new classifier with ``rules`` added.

        Provider adapters use this to plug SDK-specific detection in front of
        the generic rules without mutating shared state.
        """
        extra = tuple(rules)
        return ErrorClassifier(extra + self.rules if prepend else self.rules + extra)

    def classify(self, error: BaseException) -> AIError:
        """Normalize ``error`` into an :class:`AIError`."""
        for rule in self.rules:
            classified = rule(error)
            if classified is not None:
                if classified is not error and classified.__cause__ is None:
                    classified.__cause__ = error
                return classified
        unknown = AIError(f"Unclassified provider error: {error!r}", original=error)
        unknown.__cause__ = error
        return unknown


DEFAULT_CLASSIFIER = ErrorClassifier()


def classify_error(error: Exception, classifier: ErrorClassifier | None = None) -> AIError:
    """Normalize ``error`` into Failo's error hierarchy.

    Args:
        error: The exception raised by the user's callable.
        classifier: Optional custom classifier; defaults to the built-in one.

    Returns:
        An :class:`AIError` subclass instance.  Unrecognized exceptions become
        a bare :class:`AIError`, which no default policy retries.
    """
    return (classifier or DEFAULT_CLASSIFIER).classify(error)
