from unittest.mock import patch

from llm import get_models_with_aliases
from llm.plugins import pm

from llm_ollama import Ollama


def test_plugin_is_installed():
    names = [mod.__name__ for mod in pm.get_plugins()]
    assert "llm_ollama" in names


@patch("llm_ollama.ollama.list")
def test_registered_models(mock_ollama_list):
    return_value = {
        "models": [
            {
                "name": "stable-code:3b",
                "digest": "aa5ab8afb86208e1c097028d63074f0142ce6079420ea6f68f219933361fd869",
            },
            {
                "name": "llama2:7b",
                "digest": "78e26419b4469263f75331927a00a0284ef6544c1975b826b15abdaef17bb962",
            },
            {
                "name": "llama2:latest",
                "digest": "78e26419b4469263f75331927a00a0284ef6544c1975b826b15abdaef17bb962",
            },
            {
                "name": "phi:latest",
                "digest": "e2fd6321a5fe6bb3ac8a4e6f1cf04477fd2dea2924cf53237a995387e152ee9c",
            },
        ],
    }
    expected = (
        ("llama2:7b", ["llama2:latest", "llama2"]),
        ("phi:latest", ["phi"]),
        ("stable-code:3b", []),
    )
    mock_ollama_list.return_value = return_value
    registered_ollama_models = sorted(
        [m for m in get_models_with_aliases() if isinstance(m.model, Ollama)],
        key=lambda m: m.model.model_id,
    )
    assert len(registered_ollama_models) == len(expected)
    for model, (name, aliases) in zip(registered_ollama_models, expected):
        assert model.model.model_id == name
        assert model.aliases == aliases
