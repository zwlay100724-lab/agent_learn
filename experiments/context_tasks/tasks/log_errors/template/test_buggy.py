from pathlib import Path

from buggy import count_errors


def test_counts_level_case_insensitively():
    text = "2026 error disk\n2026 ERROR net\n2026 Error dns\n"
    assert count_errors(text) == 3


def test_ignores_error_inside_a_message():
    text = "2026 INFO an error happened\n2026 WARN ERROR threshold\n2026 ERROR real\n"
    assert count_errors(text) == 1


def test_blank_lines():
    assert count_errors("\n2026 ERROR real\n\n") == 1


def test_large_log():
    assert count_errors(Path("app.log").read_text()) == 1001
