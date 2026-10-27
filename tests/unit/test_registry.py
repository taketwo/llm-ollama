import re
from unittest.mock import Mock

import ollama
import pytest
from helpers import make_show_fail
from httpx import ConnectError, ConnectTimeout, RemoteProtocolError
from llm import get_embedding_models_with_aliases, get_models_with_aliases
from llm.plugins import load_plugins, pm

from llm_ollama.models.chat import OllamaChatModel
from llm_ollama.models.embed import OllamaEmbeddingModel
from llm_ollama.registry import CapabilityReadError, get_ollama_model_capabilities


def test_plugin_is_installed():
    load_plugins()
    names = [mod.__name__ for mod in pm.get_plugins()]
    assert "llm_ollama" in names


def test_registered_chat_models(mock_ollama_client):
    expected = (
        ("ollama/deepseek-r1:70b-llama-distill-q4_K_M", ["ollama/deepseek-r1:70b"]),
        ("ollama/gemma4:e2b", []),
        (
            "ollama/llama2:7b-q4_K_M",
            ["ollama/llama2:7b", "ollama/llama2:latest", "ollama/llama2"],
        ),
        ("ollama/phi3:latest", ["ollama/phi3"]),
        ("ollama/stable-code:3b", []),
    )
    registered_ollama_models = sorted(
        [m for m in get_models_with_aliases() if isinstance(m.model, OllamaChatModel)],
        key=lambda m: m.model.model_id,
    )
    assert len(registered_ollama_models) == len(expected)
    for model, (model_id, aliases) in zip(
        registered_ollama_models, expected, strict=True
    ):
        assert model.model.model_id == model_id
        assert model.aliases == aliases


def test_registered_embedding_models(mock_ollama_client):
    expected = (("ollama/mxbai-embed-large:latest", ["ollama/mxbai-embed-large"]),)
    registered_ollama_models = sorted(
        [
            m
            for m in get_embedding_models_with_aliases()
            if isinstance(m.model, OllamaEmbeddingModel)
        ],
        key=lambda m: m.model.model_id,
    )
    assert len(registered_ollama_models) == len(expected)
    for model, (model_id, aliases) in zip(
        registered_ollama_models, expected, strict=True
    ):
        assert model.model.model_id == model_id
        assert model.aliases == aliases


def test_registered_models_when_ollama_is_down(mocker):
    client = Mock()
    client.list.side_effect = ConnectError("[Errno 111] Connection refused")
    mocker.patch("ollama.Client", return_value=client)
    assert not any(
        isinstance(m.model, OllamaChatModel) for m in get_models_with_aliases()
    )


def test_interrupt_during_discovery_propagates(mock_ollama_client):
    mock_ollama_client.list.side_effect = KeyboardInterrupt
    with pytest.raises(KeyboardInterrupt):
        get_models_with_aliases()


def _registered_chat_models():
    return [
        m for m in get_models_with_aliases() if isinstance(m.model, OllamaChatModel)
    ]


def _registered_chat_model_ids():
    return {m.model.model_id for m in _registered_chat_models()}


def _registered_embedding_model_ids():
    return {
        m.model.model_id
        for m in get_embedding_models_with_aliases()
        if isinstance(m.model, OllamaEmbeddingModel)
    }


# Per hook: a model it registers, and a function returning the IDs it registers
_HOOKS = [
    pytest.param("phi3:latest", _registered_chat_model_ids, id="chat"),
    pytest.param(
        "mxbai-embed-large:latest", _registered_embedding_model_ids, id="embedding"
    ),
]

_CAPABILITY_ERRORS = [
    pytest.param(ollama.ResponseError("model was retired", 410), id="response-error"),
    pytest.param(ConnectionError("gone"), id="connection-error"),
    pytest.param(ConnectTimeout("connect timed out"), id="connect-timeout"),
    pytest.param(
        RemoteProtocolError("server disconnected"), id="remote-protocol-error"
    ),
]


@pytest.mark.parametrize(
    ("name", "registered_model_ids", "expected"),
    [
        pytest.param(
            "phi3:latest",
            _registered_chat_model_ids,
            {
                "ollama/deepseek-r1:70b-llama-distill-q4_K_M",
                "ollama/gemma4:e2b",
                "ollama/llama2:7b-q4_K_M",
                "ollama/stable-code:3b",
            },
            id="chat",
        ),
        pytest.param(
            "mxbai-embed-large:latest",
            _registered_embedding_model_ids,
            set(),
            id="embedding",
        ),
    ],
)
@pytest.mark.parametrize("error", _CAPABILITY_ERRORS)
def test_model_whose_capabilities_cannot_be_read_is_skipped(
    mock_ollama_client,
    name,
    registered_model_ids,
    expected,
    error,
):
    make_show_fail(mock_ollama_client, name, error)
    assert registered_model_ids() == expected


@pytest.mark.parametrize("error", _CAPABILITY_ERRORS)
def test_capability_read_error_keeps_message_and_cause(mock_ollama_client, error):
    make_show_fail(mock_ollama_client, "phi3:latest", error)
    with pytest.raises(
        CapabilityReadError, match=f"^{re.escape(str(error))}$"
    ) as excinfo:
        get_ollama_model_capabilities("e2fd6321a5fe", "phi3:latest")
    assert excinfo.value.__cause__ is error


@pytest.mark.parametrize(("name", "registered_model_ids"), _HOOKS)
@pytest.mark.parametrize("error", _CAPABILITY_ERRORS)
def test_model_with_cached_capabilities_is_registered_when_show_fails(
    mock_ollama_client,
    name,
    registered_model_ids,
    error,
):
    assert f"ollama/{name}" in registered_model_ids()
    make_show_fail(mock_ollama_client, name, error)
    assert f"ollama/{name}" in registered_model_ids()


@pytest.mark.parametrize(("name", "registered_model_ids"), _HOOKS)
@pytest.mark.parametrize("error", _CAPABILITY_ERRORS)
def test_failed_capabilities_read_is_not_cached(
    mock_ollama_client,
    name,
    registered_model_ids,
    error,
):
    show = make_show_fail(mock_ollama_client, name, error)
    assert f"ollama/{name}" not in registered_model_ids()
    mock_ollama_client.show.side_effect = show
    assert f"ollama/{name}" in registered_model_ids()


@pytest.mark.parametrize(
    "error", [ValueError("invalid capabilities"), KeyboardInterrupt()]
)
def test_unexpected_capability_errors_propagate(mock_ollama_client, error):
    make_show_fail(mock_ollama_client, "phi3:latest", error)
    with pytest.raises(type(error)) as excinfo:
        _registered_chat_model_ids()
    assert excinfo.value is error


def test_registered_models_support_tools_as_reported(mock_ollama_client):
    registered = _registered_chat_models()
    assert registered
    for model in registered:
        expected = model.model.model_id == "ollama/gemma4:e2b"
        assert model.model.supports_tools is expected
        assert model.async_model.supports_tools is expected
