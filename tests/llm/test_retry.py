import logging

from trace_swe.infra.event_log import EventLogWriter, bind_event_log, read_events
from trace_swe.llm.utils.retry import retry


def test_retry_emits_attempt_and_error(tmp_path):
    writer = EventLogWriter(tmp_path / "x.jsonl")
    calls = 0

    def flaky():
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("boom")
        return "ok"

    with bind_event_log(writer):
        for attempt in retry(logger=logging.getLogger("test"), abort_exceptions=[]):
            with attempt:
                result = flaky()

    assert result == "ok"
    retries = [e for e in read_events(writer.path) if e["event"] == "llm_retry"]
    assert [(e["attempt"], e["error_type"], e["error_message"]) for e in retries] == [(1, "RuntimeError", "boom")]
