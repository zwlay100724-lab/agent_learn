import pytest

from trace_swe.agent import get_agent, get_agent_class
from trace_swe.agent.default import DefaultAgent
from trace_swe.agent.interactive import InteractiveAgent
from trace_swe.execution.local import LocalEnvironment
from trace_swe.llm.test_models import DeterministicModel


class TestGetAgentClass:
    @pytest.mark.parametrize(
        ("spec", "expected"),
        [
            ("trace_swe.agent.default.DefaultAgent", DefaultAgent),
            ("trace_swe.agent.interactive.InteractiveAgent", InteractiveAgent),
        ],
    )
    def test_full_path(self, spec, expected):
        assert get_agent_class(spec) is expected

    @pytest.mark.parametrize(
        ("spec", "expected"),
        [
            ("default", DefaultAgent),
            ("interactive", InteractiveAgent),
        ],
    )
    def test_shorthand(self, spec, expected):
        assert get_agent_class(spec) is expected

    def test_invalid_spec(self):
        with pytest.raises(ValueError, match="Unknown agent type"):
            get_agent_class("invalid_agent")

    def test_invalid_module(self):
        with pytest.raises(ValueError, match="Unknown agent type"):
            get_agent_class("nonexistent.module.Class")


class TestGetAgent:
    @pytest.fixture
    def model(self):
        return DeterministicModel(outputs=[])

    @pytest.fixture
    def env(self):
        return LocalEnvironment()

    @pytest.fixture
    def base_config(self):
        return {"system_template": "test", "instance_template": "test"}

    def test_default_type(self, model, env, base_config):
        agent = get_agent(model, env, base_config, default_type="default")
        assert isinstance(agent, DefaultAgent)

    def test_agent_class_in_config(self, model, env, base_config):
        agent = get_agent(model, env, {**base_config, "agent_class": "interactive"}, default_type="default")
        assert isinstance(agent, InteractiveAgent)

    def test_config_passed_to_agent(self, model, env, base_config):
        agent = get_agent(model, env, {**base_config, "step_limit": 42}, default_type="default")
        assert agent.config.step_limit == 42
