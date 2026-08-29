"""Failo -- reliable AI calls with retry, fallback, and failover support.

Failo is a provider-independent resilience layer.  It never imports an AI
SDK; you keep your existing client and wrap the call in a callable.

    from failo import resilient_call

    result = await resilient_call(primary=call_openai, fallbacks=[call_gemini])
    print(result.response, result.provider, result.fallback_used)
"""

from .core import ResilientClient, resilient_call, resilient_call_sync
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
from .fallback import execute_with_fallback
from .models import Attempt, ResilientResult, RetryOutcome
from .policies import FallbackPolicy, RetryPolicy
from .retry import execute_with_retry

__version__ = "0.1.0"

__all__ = [
    "AIError",
    "Attempt",
    "AuthenticationError",
    "ConnectionError",
    "ErrorClassifier",
    "FallbackPolicy",
    "InvalidRequestError",
    "PermissionError",
    "RateLimitError",
    "ResilientClient",
    "ResilientResult",
    "RetryOutcome",
    "RetryPolicy",
    "ServerError",
    "TimeoutError",
    "__version__",
    "classify_error",
    "execute_with_fallback",
    "execute_with_retry",
    "resilient_call",
    "resilient_call_sync",
]
