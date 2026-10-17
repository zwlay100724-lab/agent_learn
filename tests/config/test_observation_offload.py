from pathlib import Path

import pytest
import yaml
from jinja2 import StrictUndefined, Template

CONFIG_DIR = Path(__file__).parent.parent.parent / "src" / "minisweagent" / "config"
CASES = [
    ("default.yaml", "<full_output_path>"),
    ("mini.yaml", '"full_output_path"'),
    ("mini_textbased.yaml", "<full_output_path>"),
]


def load_template(config_file: str) -> Template:
    with (CONFIG_DIR / config_file).open() as f:
        return Template(yaml.safe_load(f)["model"]["observation_template"], undefined=StrictUndefined)


@pytest.mark.parametrize(("config_file", "marker"), CASES)
def test_long_output_without_path_renders(config_file, marker):
    result = load_template(config_file).render(output={"returncode": 0, "output": "x" * 20000, "exception_info": ""})
    assert "x" * 5000 in result
    assert marker not in result


@pytest.mark.parametrize(("config_file", "marker"), CASES)
def test_long_output_with_path_renders(config_file, marker):
    result = load_template(config_file).render(
        output={"returncode": 0, "output": "x" * 20000, "exception_info": "", "output_path": "/tmp/full.txt"}
    )
    assert marker in result
    assert "/tmp/full.txt" in result


@pytest.mark.parametrize(("config_file", "marker"), CASES)
def test_short_output_never_shows_path(config_file, marker):
    result = load_template(config_file).render(
        output={"returncode": 0, "output": "hi", "exception_info": "", "output_path": "/tmp/full.txt"}
    )
    assert "hi" in result
    assert marker not in result
    assert "/tmp/full.txt" not in result