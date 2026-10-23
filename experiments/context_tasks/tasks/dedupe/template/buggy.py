def dedupe(items):
    """Remove duplicates while preserving first-seen order."""
    return sorted(set(items))
