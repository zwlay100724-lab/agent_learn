from types import SimpleNamespace

import pytest

from trace_swe.llm.utils.usage import extract_usage


def test_extracts_completion_usage_from_dict():
    response = {"usage": {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14}}
    assert extract_usage(response) == {
        "prompt_tokens": 10,
        "completion_tokens": 4,
        "total_tokens": 14,
        "cache_read_input_tokens": None,
        "cache_creation_input_tokens": None,
    }


def test_extracts_usage_from_response_object():
    usage = SimpleNamespace(prompt_tokens=10, completion_tokens=4, total_tokens=14)
    assert extract_usage(SimpleNamespace(usage=usage))["total_tokens"] == 14


def test_maps_response_api_input_and_output_tokens():
    result = extract_usage({"usage": {"input_tokens": 8, "output_tokens": 2}})
    assert (result["prompt_tokens"], result["completion_tokens"], result["total_tokens"]) == (8, 2, 10)


def test_extracts_cached_tokens_from_details():
    response = {
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 1,
            "total_tokens": 11,
            "prompt_tokens_details": {"cached_tokens": 6},
        }
    }
    assert extract_usage(response)["cache_read_input_tokens"] == 6


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        ({}, None),
        ({"usage": None}, None),
        ({"usage": {"prompt_tokens": None}}, None),
        ({"usage": {"prompt_tokens": 3}}, 3),
        ({"usage": {"prompt_tokens": True}}, None),
    ],
)
def test_missing_usage_is_none_not_zero(response, expected):
    result = extract_usage(response)
    assert (result["prompt_tokens"] if result else None) == expected
