from unittest.mock import patch

from click.testing import CliRunner
from llm.cli import cli
from llm.plugins import pm


def test_plugin_is_installed():
    names = [mod.__name__ for mod in pm.get_plugins()]
    assert "llm_ollama" in names


@patch(
    "llm_ollama.ollama.list",
    return_value={
        "models": [{"name": "llama2:latest"}, {"name": "stable-code:latest"}],
    },
)
def test_registered_models(mock_ollama_list):
    runner = CliRunner()
    result = runner.invoke(cli, ["models", "list"])
    assert result.exit_code == 0, result.output
    for fragment in ("Ollama: llama2:latest", "Ollama: stable-code:latest"):
        assert fragment in result.output
