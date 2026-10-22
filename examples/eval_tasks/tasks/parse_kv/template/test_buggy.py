from buggy import parse_kv


def test_trims_whitespace():
    assert parse_kv(" a = 1 \n b = 2 ") == {"a": "1", "b": "2"}


def test_keeps_inner_equals():
    assert parse_kv("url = a=b") == {"url": "a=b"}


def test_ignores_blank_lines():
    assert parse_kv("\n\n a = 1 \n") == {"a": "1"}
