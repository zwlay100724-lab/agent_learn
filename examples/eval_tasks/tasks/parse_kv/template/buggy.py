def parse_kv(text):
    """Parse 'key=value' lines, trimming whitespace around keys and values."""
    result = {}
    for line in text.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            result[key] = value
    return result
