from buggy import sum_range


def test_inclusive():
    assert sum_range(1, 5) == 15


def test_single():
    assert sum_range(3, 3) == 3


def test_empty_range():
    assert sum_range(1, 0) == 0
