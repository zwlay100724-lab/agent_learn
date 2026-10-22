"""Normalize token usage from completion and response API results."""

from typing import Any

USAGE_FIELDS = (
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "cache_read_input_tokens",
    "cache_creation_input_tokens",
)


def _get(obj: Any, key: str) -> Any:
    return obj.get(key) if isinstance(obj, dict) else getattr(obj, key, None)


def _int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def extract_usage(response: Any) -> dict[str, int | None] | None:
    """Return normalized token counts, or None when the provider reports no usage."""
    usage = _get(response, "usage")
    if usage is None:
        return None
    prompt = _int(_get(usage, "prompt_tokens"))
    if prompt is None:
        prompt = _int(_get(usage, "input_tokens"))
    completion = _int(_get(usage, "completion_tokens"))
    if completion is None:
        completion = _int(_get(usage, "output_tokens"))
    total = _int(_get(usage, "total_tokens"))
    if total is None and prompt is not None and completion is not None:
        total = prompt + completion
    details = _get(usage, "prompt_tokens_details") or _get(usage, "input_tokens_details")
    cache_read = _int(_get(usage, "cache_read_input_tokens"))
    if cache_read is None:
        cache_read = _int(_get(details, "cached_tokens"))
    cache_creation = _int(_get(usage, "cache_creation_input_tokens"))
    if all(value is None for value in (prompt, completion, total, cache_read, cache_creation)):
        return None
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": total,
        "cache_read_input_tokens": cache_read,
        "cache_creation_input_tokens": cache_creation,
    }


def sum_usage(items: list[dict]) -> dict:
    """Sum normalized usage dictionaries and report how many carried usage."""
    present = [item for item in items if any(item.get(key) is not None for key in USAGE_FIELDS)]
    if not present:
        return {**dict.fromkeys(USAGE_FIELDS), "n_calls_with_usage": 0}
    return {
        **{key: sum(item[key] for item in present if item.get(key) is not None) for key in USAGE_FIELDS},
        "n_calls_with_usage": len(present),
    }
