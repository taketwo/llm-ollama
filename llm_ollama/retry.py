"""Retry helpers for transient Ollama HTTP errors."""

import asyncio
import sys
import time

import ollama

#: Maximum number of attempts for transient Ollama HTTP errors.
OLLAMA_MAX_RETRIES = 5

#: Backoff factor (seconds) for rate-limit retries.
OLLAMA_RETRY_BACKOFF = 1.0


def ollama_error_message(exc: Exception, *, include_status: bool = False) -> str:
    """Extract a human-readable message from an Ollama or generic exception."""
    message = ""
    if hasattr(exc, "error"):
        message = str(getattr(exc, "error"))  # noqa: B009
    elif str(exc):
        message = str(exc)
    if include_status and hasattr(exc, "status_code"):
        status_code = getattr(exc, "status_code")  # noqa: B009
        if status_code is not None and status_code != -1:
            return f"{message} (status code: {status_code})"
    return message


class OllamaGiveUp(Exception):
    """Raised when retries are exhausted or the error is not retryable."""

    def __init__(self, error: str, status_code: int | None = None) -> None:
        super().__init__(error)
        self.error = error
        self.status_code = status_code


def ollama_warn(message: str, status_code: int | None = None) -> None:
    """Emit a warning about an Ollama failure to STDERR without crashing."""
    prefix = f"Ollama warning ({status_code})" if status_code else "Ollama warning"
    print(f"{prefix}: {message}", file=sys.stderr)


def is_retryable_status(status_code: int) -> bool:
    """Return True for transient HTTP status codes that are worth retrying."""
    return status_code in {429, 500, 501, 502, 503, 504}


def ollama_retry(call):
    """Yield a single callable that retries transient Ollama ResponseErrors.

    The returned callable performs up to :data:`OLLAMA_MAX_RETRIES` attempts internally.
    Non-retryable errors (e.g. 410 Gone) and exhausted retryable errors raise
    :class:`OllamaGiveUp`, which the caller can catch to emit a warning and continue.

    """
    last_error: OllamaGiveUp | None = None

    def attempt_fn():
        nonlocal last_error
        for attempt_number in range(OLLAMA_MAX_RETRIES):
            try:
                return call()
            except ollama.ResponseError as exc:
                if exc.status_code == 410:
                    # 410 Gone is permanent for this model; do not retry.
                    raise OllamaGiveUp(
                        ollama_error_message(exc),
                        status_code=exc.status_code,
                    ) from exc
                if is_retryable_status(exc.status_code):
                    last_error = OllamaGiveUp(
                        ollama_error_message(exc),
                        status_code=exc.status_code,
                    )
                    is_last = attempt_number == OLLAMA_MAX_RETRIES - 1
                    if exc.status_code == 429 and not is_last:
                        time.sleep(OLLAMA_RETRY_BACKOFF * (attempt_number + 1))
                    if is_last:
                        raise last_error from exc
                    continue
                # Any other ResponseError is not retryable.
                raise OllamaGiveUp(
                    ollama_error_message(exc),
                    status_code=exc.status_code,
                ) from exc
            except Exception as exc:
                # Non-ResponseError exceptions (network, etc.) are not handled here;
                # let them propagate to preserve existing behaviour.
                raise OllamaGiveUp(ollama_error_message(exc)) from exc
        return None  # pragma: no cover

    yield attempt_fn


def async_ollama_retry(call):
    """Async equivalent of :func:`ollama_retry`."""
    last_error: OllamaGiveUp | None = None

    async def attempt_fn():
        nonlocal last_error
        for attempt_number in range(OLLAMA_MAX_RETRIES):
            try:
                return await call()
            except ollama.ResponseError as exc:
                if exc.status_code == 410:
                    raise OllamaGiveUp(
                        ollama_error_message(exc),
                        status_code=exc.status_code,
                    ) from exc
                if is_retryable_status(exc.status_code):
                    last_error = OllamaGiveUp(
                        ollama_error_message(exc),
                        status_code=exc.status_code,
                    )
                    is_last = attempt_number == OLLAMA_MAX_RETRIES - 1
                    if exc.status_code == 429 and not is_last:
                        await asyncio.sleep(OLLAMA_RETRY_BACKOFF * (attempt_number + 1))
                    if is_last:
                        raise last_error from exc
                    continue
                raise OllamaGiveUp(
                    ollama_error_message(exc),
                    status_code=exc.status_code,
                ) from exc
            except Exception as exc:
                raise OllamaGiveUp(ollama_error_message(exc)) from exc
        return None  # pragma: no cover

    yield attempt_fn
