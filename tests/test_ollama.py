from unittest.mock import patch

import pytest

from httpx import ConnectError

from llm import get_embedding_models_with_aliases, get_models_with_aliases
from llm.plugins import pm

from llm_ollama import Ollama, OllamaEmbed


@pytest.fixture
def mock_ollama():
    with (
        patch("llm_ollama.ollama.list") as mock_list,
        patch(
            "llm_ollama.ollama.show",
        ) as mock_show,
    ):
        return_value = {
            "models": [
                {
                    "model": "stable-code:3b",
                    "digest": "aa5ab8afb86208e1c097028d63074f0142ce6079420ea6f68f219933361fd869",
                    "modelinfo": {
                        "general.architecture": "stablelm",
                    },
                },
                {
                    "model": "llama2:7b",
                    "digest": "78e26419b4469263f75331927a00a0284ef6544c1975b826b15abdaef17bb962",
                    "modelinfo": {
                        "general.architecture": "llama",
                    },
                },
                {
                    "model": "llama2:latest",
                    "digest": "78e26419b4469263f75331927a00a0284ef6544c1975b826b15abdaef17bb962",
                    "modelinfo": {
                        "general.architecture": "llama",
                    },
                },
                {
                    "model": "phi3:latest",
                    "digest": "e2fd6321a5fe6bb3ac8a4e6f1cf04477fd2dea2924cf53237a995387e152ee9c",
                    "modelinfo": {
                        "general.architecture": "phi3",
                    },
                },
                {
                    "model": "mxbai-embed-large:latest",
                    "digest": "468836162de7f81e041c43663fedbbba921dcea9b9fefea135685a39b2d83dd8",
                    "modelinfo": {
                        "general.architecture": "bert",
                        "bert.pooling_type": 2,
                    },
                },
            ],
        }
        mock_list.return_value = return_value
        mock_show.side_effect = lambda name: next(
            m for m in return_value["models"] if m["model"] == name
        )
        yield mock_list, mock_show


def test_registered_chat_models(mock_ollama):
    expected = (
        ("llama2:7b", ["llama2:latest", "llama2"]),
        ("phi3:latest", ["phi3"]),
        ("stable-code:3b", []),
    )
    registered_ollama_models = sorted(
        [m for m in get_models_with_aliases() if isinstance(m.model, Ollama)],
        key=lambda m: m.model.model_id,
    )
    assert len(registered_ollama_models) == len(expected)
    for model, (name, aliases) in zip(registered_ollama_models, expected):
        assert model.model.model_id == name
        assert model.aliases == aliases


def test_registered_embedding_models(mock_ollama):
    expected = (
        ("llama2:7b", ["llama2:latest", "llama2"]),
        ("mxbai-embed-large:latest", ["mxbai-embed-large"]),
        ("phi3:latest", ["phi3"]),
        ("stable-code:3b", []),
    )
    registered_ollama_models = sorted(
        [
            m
            for m in get_embedding_models_with_aliases()
            if isinstance(m.model, OllamaEmbed)
        ],
        key=lambda m: m.model.model_id,
    )
    assert len(registered_ollama_models) == len(expected)
    for model, (name, aliases) in zip(registered_ollama_models, expected):
        assert model.model.model_id == name
        assert model.aliases == aliases


@pytest.mark.parametrize(
    ("envvar_value", "expected_truncate_value"),
    [
        (None, True),
        ("True", True),
        ("true", True),
        ("yes", True),
        ("y", True),
        ("on", True),
        ("False", False),
        ("false", False),
        ("no", False),
        ("n", False),
        ("off", False),
    ],
)
@patch("llm_ollama.ollama.embed")
def test_model_embed(
    mock_ollama_embed,
    envvar_value,
    expected_truncate_value,
    monkeypatch,
):
    expected = [0.1] * 1024
    mock_ollama_embed.return_value = {"embeddings": [expected]}
    if envvar_value is not None:
        monkeypatch.setenv("OLLAMA_EMBED_TRUNCATE", envvar_value)
    else:
        monkeypatch.delenv("OLLAMA_EMBED_TRUNCATE", raising=False)
    result = OllamaEmbed("mxbai-embed-large:latest").embed("string to embed")
    assert result == expected
    _, called_kwargs = mock_ollama_embed.call_args
    assert called_kwargs.get("truncate") is expected_truncate_value


@patch("llm_ollama.ollama.list")
def test_registered_models_when_ollama_is_down(mock_ollama_list):
    mock_ollama_list.side_effect = ConnectError("[Errno 111] Connection refused")
    assert not any(isinstance(m.model, Ollama) for m in get_models_with_aliases())
