def count_errors(log_text):
    """Count lines whose level is ERROR, ignoring case."""
    return sum(1 for line in log_text.splitlines() if "ERROR" in line)
