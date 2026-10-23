from buggy import dedupe


def test_preserves_order():
    assert dedupe([3, 1, 3, 2, 1]) == [3, 1, 2]


def test_handles_strings():
    assert dedupe(["b", "a", "b"]) == ["b", "a"]


def test_empty():
    assert dedupe([]) == []
