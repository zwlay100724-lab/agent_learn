"""Retry utility for model queries."""

import logging
import os

from tenacity import Retrying, before_sleep_log, retry_if_not_exception_type, stop_after_attempt, wait_exponential

from trace_swe.infra.event_log import current_event_log


def _emit_retry(retry_state) -> None:
    writer = current_event_log()
    if writer is None:
        return
    error = retry_state.outcome.exception() if retry_state.outcome else None
    writer.retry_count += 1
    writer.emit(
        "llm_retry",
        attempt=retry_state.attempt_number,
        error_type=type(error).__name__ if error else None,
        error_message=str(error) if error else None,
    )


def _log_and_emit_retry(logger: logging.Logger):
    log = before_sleep_log(logger, logging.WARNING)

    def callback(retry_state) -> None:
        log(retry_state)
        _emit_retry(retry_state)

    return callback


def retry(*, logger: logging.Logger, abort_exceptions: list[type[Exception]]) -> Retrying:
    """Thin wrapper around tenacity.Retrying to make use of global config etc.

    Args:
        logger: Logger to use for reporting retries
        abort_exceptions: Exceptions to abort on.

    Returns:
        A tenacity.Retrying object.
    """
    return Retrying(
        reraise=True,
        stop=stop_after_attempt(int(os.getenv("TRACE_SWE_MODEL_RETRY_STOP_AFTER_ATTEMPT", "10"))),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        before_sleep=_log_and_emit_retry(logger),
        retry=retry_if_not_exception_type(tuple(abort_exceptions)),
    )
