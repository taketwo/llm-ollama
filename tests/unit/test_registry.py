from unittest.mock import Mock

from httpx import ConnectError
from llm import get_embedding_models_with_aliases, get_models_with_aliases
from llm.plugins import load_plugins, pm

from llm_ollama.models.chat import OllamaChatModel
from llm_ollama.models.embed import OllamaEmbeddingModel


def test_plugin_is_installed():
    load_plugins()
    names = [mod.__name__ for mod in pm.get_plugins()]
    assert "llm_ollama" in names


def test_registered_chat_models(mock_ollama_client):
    expected = (
        ("deepseek-r1:70b-llama-distill-q4_K_M", ["deepseek-r1:70b"]),
        ("llama2:7b-q4_K_M", ["llama2:7b", "llama2:latest", "llama2"]),
        ("phi3:latest", ["phi3"]),
        ("stable-code:3b", []),
    )
    registered_ollama_models = sorted(
        [m for m in get_models_with_aliases() if isinstance(m.model, OllamaChatModel)],
        key=lambda m: m.model.model_id,
    )
    assert len(registered_ollama_models) == len(expected)
    for model, (name, aliases) in zip(registered_ollama_models, expected):
        assert model.model.model_id == name
        assert model.aliases == aliases


def test_registered_embedding_models(mock_ollama_client):
    expected = (
        ("deepseek-r1:70b-llama-distill-q4_K_M", ["deepseek-r1:70b"]),
        ("llama2:7b-q4_K_M", ["llama2:7b", "llama2:latest", "llama2"]),
        ("mxbai-embed-large:latest", ["mxbai-embed-large"]),
        ("phi3:latest", ["phi3"]),
        ("stable-code:3b", []),
    )
    registered_ollama_models = sorted(
        [
            m
            for m in get_embedding_models_with_aliases()
            if isinstance(m.model, OllamaEmbeddingModel)
        ],
        key=lambda m: m.model.model_id,
    )
    assert len(registered_ollama_models) == len(expected)
    for model, (name, aliases) in zip(registered_ollama_models, expected):
        assert model.model.model_id == name
        assert model.aliases == aliases


def test_registered_models_when_ollama_is_down(mocker):
    client = Mock()
    client.list.side_effect = ConnectError("[Errno 111] Connection refused")
    mocker.patch("ollama.Client", return_value=client)
    assert not any(
        isinstance(m.model, OllamaChatModel) for m in get_models_with_aliases()
    )
