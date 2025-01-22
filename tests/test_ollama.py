from unittest.mock import patch, AsyncMock

import pytest
import os

from httpx import ConnectError

from llm import (
    get_embedding_models_with_aliases,
    get_models_with_aliases,
    get_async_model,
)
from llm.plugins import load_plugins, pm

from llm_ollama import Ollama, OllamaEmbed


from ollama import AsyncClient


IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"


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
                    "model": "llama2:7b-q4_K_M",
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
                {
                    "model": "deepseek-r1:70b",
                    "digest": "0c1615a8ca32ef41e433aa420558b4685f9fc7f3fd74119860a8e2e389cd7942",
                    "modelinfo": {
                        "general.architecture": "llama",
                    },
                },
                {
                    "model": "deepseek-r1:70b-llama-distill-q4_K_M",
                    "digest": "0c1615a8ca32ef41e433aa420558b4685f9fc7f3fd74119860a8e2e389cd7942",
                    "modelinfo": {
                        "general.architecture": "llama",
                    },
                },
            ],
        }
        mock_list.return_value = return_value
        mock_show.side_effect = lambda name: next(
            m for m in return_value["models"] if m["model"] == name
        )
        yield mock_list, mock_show


def test_plugin_is_installed():
    load_plugins()
    names = [mod.__name__ for mod in pm.get_plugins()]
    assert "llm_ollama" in names


def test_registered_chat_models(mock_ollama):
    expected = (
        ("deepseek-r1:70b-llama-distill-q4_K_M", ["deepseek-r1:70b"]),
        ("llama2:7b-q4_K_M", ["llama2:7b", "llama2:latest", "llama2"]),
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


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Test doesn't work in Github Actions.")
@pytest.mark.asyncio
async def test_actual_run():
    """Tests actual run. Needs llama3.2"""
    model = get_async_model("llama3.2:latest")
    response = model.prompt("a short poem about tea")
    response_text = await response.text()
    assert len(response_text) > 0


@pytest.mark.asyncio
async def test_async_ollama_call(mock_ollama):
    # Mock the asynchronous chat method to return an async iterable
    async def mock_chat(*args, **kwargs):
        messages = [
            {"message": {"content": "Test response 1"}},
            {"message": {"content": "Test response 2"}},
        ]
        for msg in messages:
            yield msg

    # Patch the ollama.AsyncClient.chat method
    with patch("ollama.AsyncClient.chat", new_callable=AsyncMock) as mock_chat_method:
        mock_chat_method.return_value = mock_chat()

        # Instantiate the model and send a prompt
        model = get_async_model("llama2:7b")
        response = model.prompt("Dummy Prompt")
        response_text = await response.text()

        assert response_text == "Test response 1Test response 2"
        mock_chat_method.assert_called_once()
