import json

import pytest

from trace_swe.infra.event_log import EventLogWriter, bind_event_log, current_event_log, digest, read_events


def test_emit_writes_one_json_object_per_line(tmp_path):
    path = tmp_path / "run.events.jsonl"
    writer = EventLogWriter(path, run_id="abc")
    writer.emit("run_start", task="t")
    writer.emit("step_start", n_calls=0)
    writer.close()

    events = read_events(path)
    assert [e["event"] for e in events] == ["run_start", "step_start"]
    assert [e["seq"] for e in events] == [0, 1]
    assert {e["run_id"] for e in events} == {"abc"}
    assert all(e["v"] == 1 for e in events)
    assert events[0]["task"] == "t"
    assert len(path.read_text().splitlines()) == 2


def test_envelope_cannot_be_overridden_by_payload(tmp_path):
    event = EventLogWriter(tmp_path / "x.jsonl").emit("run_start", seq=99, v=9)
    assert (event["seq"], event["v"], event["event"]) == (0, 1, "run_start")


def test_new_run_resets_seq_and_run_id(tmp_path):
    writer = EventLogWriter(tmp_path / "x.jsonl", run_id="first")
    writer.emit("a")
    writer.new_run("second")
    writer.emit("b")
    assert [(e["run_id"], e["seq"], e["event"]) for e in read_events(writer.path)] == [
        ("first", 0, "a"),
        ("second", 0, "b"),
    ]


def test_disabled_writer_writes_nothing(tmp_path):
    writer = EventLogWriter(None)
    assert writer.emit("a") is None
    assert not list(tmp_path.iterdir())


def test_read_events_tolerates_truncated_final_line(tmp_path):
    path = tmp_path / "x.jsonl"
    writer = EventLogWriter(path)
    writer.emit("a")
    writer.close()
    with path.open("a") as f:
        f.write('{"event": "b", "seq"')
    assert [e["event"] for e in read_events(path)] == ["a"]


def test_read_events_rejects_mid_file_corruption(tmp_path):
    path = tmp_path / "x.jsonl"
    path.write_text('{"event": "a"}\nnot json\n{"event": "c"}\n')
    with pytest.raises(json.JSONDecodeError):
        read_events(path)


def test_writer_disables_itself_on_oserror(tmp_path):
    writer = EventLogWriter(tmp_path)  # a directory cannot be opened as a file
    assert writer.emit("a") is None
    assert not writer.enabled


def test_bind_event_log_is_scoped(tmp_path):
    writer = EventLogWriter(tmp_path / "x.jsonl")
    assert current_event_log() is None
    with bind_event_log(writer):
        assert current_event_log() is writer
    assert current_event_log() is None


def test_digest_is_stable_and_order_independent():
    assert digest({"a": 1, "b": 2}) == digest({"b": 2, "a": 1})
    assert digest({"a": 1}) != digest({"a": 2})
