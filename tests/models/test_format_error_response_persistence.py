"""FormatError must carry the unparsed LLM response for trajectory inspection."""

import json
from unittest.mock import MagicMock, patch

import pytest

from minisweagent.exceptions import FormatError
from minisweagent.models.litellm_model import LitellmModel


def _bad_tool_call_mock() -> MagicMock:
    tc = MagicMock()
    tc.id = "call_xyz"
    tc.function.name = "unknown_tool"
    tc.function.arguments = "{}"
    return tc


def _query_with_bad_tool_call(model: LitellmModel, response: MagicMock) -> FormatError:
    with (
        patch.object(LitellmModel, "_query", return_value=response),
        patch.object(LitellmModel, "_calculate_cost", return_value={"cost": 0.02}),
    ):
        with pytest.raises(FormatError) as excinfo:
            model.query([{"role": "user", "content": "hi"}])
    return excinfo.value


def test_litellm_model_format_error_persists_response() -> None:
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.tool_calls = [_bad_tool_call_mock()]
    response.model_dump.return_value = {"id": "resp_1", "nested": {"k": "v"}}

    error = _query_with_bad_tool_call(LitellmModel(model_name="test/model"), response)
    extra = error.messages[0]["extra"]

    assert extra["response"] == {"id": "resp_1", "nested": {"k": "v"}}
    assert extra["cost"] == 0.02
    response.model_dump.assert_any_call(mode="json")


def test_persisted_response_round_trips_through_json() -> None:
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.tool_calls = [_bad_tool_call_mock()]
    response.model_dump.return_value = {"id": "abc", "nested": {"k": "v"}}

    error = _query_with_bad_tool_call(LitellmModel(model_name="test/model"), response)

    reloaded = json.loads(json.dumps(error.messages[0]))
    assert reloaded["extra"]["response"] == {"id": "abc", "nested": {"k": "v"}}


def test_format_error_still_propagates_after_persisting_response() -> None:
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.tool_calls = [_bad_tool_call_mock()]
    response.model_dump.return_value = {"id": "abc"}

    assert _query_with_bad_tool_call(LitellmModel(model_name="test/model"), response)
