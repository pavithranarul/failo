from __future__ import annotations

import asyncio
import socket

import pytest

from failo import errors
from failo.errors import (
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
from tests.conftest import FakeHTTPError


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (400, InvalidRequestError),
        (401, AuthenticationError),
        (403, PermissionError),
        (408, TimeoutError),
        (429, RateLimitError),
        (500, ServerError),
        (502, ServerError),
        (503, ServerError),
        (504, ServerError),
    ],
)
def test_status_code_mapping(status: int, expected: type[AIError]) -> None:
    classified = classify_error(FakeHTTPError(status))
    assert type(classified) is expected
    assert classified.status_code == status


def test_unknown_status_code_is_safe() -> None:
    classified = classify_error(FakeHTTPError(418))
    assert type(classified) is AIError
    assert not isinstance(classified, (RateLimitError, ServerError, TimeoutError, ConnectionError))


def test_unknown_5xx_is_server_error() -> None:
    assert isinstance(classify_error(FakeHTTPError(599)), ServerError)


def test_unknown_exception_stays_base_ai_error() -> None:
    classified = classify_error(ValueError("what even is this"))
    assert type(classified) is AIError
    assert isinstance(classified.original, ValueError)


def test_already_normalized_errors_pass_through() -> None:
    original = RateLimitError("slow down")
    assert classify_error(original) is original


def test_original_exception_is_preserved() -> None:
    raw = FakeHTTPError(500)
    classified = classify_error(raw)
    assert classified.original is raw
    assert classified.__cause__ is raw


def test_nested_response_status_code() -> None:
    class Response:
        status_code = 429

    class SDKError(Exception):
        response = Response()

    assert isinstance(classify_error(SDKError()), RateLimitError)


def test_builtin_timeout_and_connection_errors() -> None:
    assert isinstance(classify_error(asyncio.TimeoutError()), TimeoutError)
    assert isinstance(classify_error(socket.gaierror("dns down")), ConnectionError)
    assert isinstance(classify_error(OSError("socket closed")), ConnectionError)


def test_duck_typed_class_names() -> None:
    class RateLimitExceeded(Exception):
        pass

    class DeadlineExceeded(Exception):
        pass

    assert isinstance(classify_error(RateLimitExceeded()), RateLimitError)
    assert isinstance(classify_error(DeadlineExceeded()), TimeoutError)


def test_retry_after_attribute_is_extracted() -> None:
    assert classify_error(FakeHTTPError(429, retry_after=7)).retry_after == 7.0


def test_retry_after_header_is_extracted() -> None:
    assert classify_error(FakeHTTPError(429, headers={"Retry-After": "2.5"})).retry_after == 2.5


def test_negative_retry_after_is_ignored() -> None:
    assert errors.extract_retry_after(FakeHTTPError(429, retry_after=-5)) is None


def test_classifier_is_extensible_without_global_state() -> None:
    class VendorQuotaError(Exception):
        pass

    def rule(error: BaseException) -> AIError | None:
        if isinstance(error, VendorQuotaError):
            return RateLimitError("vendor quota", original=error)
        return None

    custom = ErrorClassifier().with_rules(rule)
    assert isinstance(custom.classify(VendorQuotaError()), RateLimitError)
    # The default classifier is untouched.
    assert type(classify_error(VendorQuotaError())) is AIError
