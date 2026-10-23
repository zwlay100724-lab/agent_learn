from trace_swe.infra.output_store import OutputStore


def test_save_writes_text_and_creates_directories(tmp_path):
    path = OutputStore(tmp_path, max_bytes=100).save("hello", "run/000001.txt")
    assert path == tmp_path / "run" / "000001.txt"
    assert path.read_text() == "hello"


def test_save_truncates_at_max_bytes(tmp_path):
    path = OutputStore(tmp_path, max_bytes=10).save("x" * 100, "out.txt")
    content = path.read_text()
    assert content.startswith("x" * 10)
    assert "truncated at 10 bytes" in content


def test_save_is_disabled_without_root():
    assert OutputStore(None, max_bytes=10).save("hello", "out.txt") is None


def test_save_returns_none_on_oserror(tmp_path):
    (tmp_path / "adir").mkdir()
    assert OutputStore(tmp_path, max_bytes=100).save("hello", "adir") is None
