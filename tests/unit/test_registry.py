from unittest.mock import Mock

import ollama
import pytest
from helpers import make_show_fail
from httpx import ConnectError
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
    expected = (
        ("ollama/deepseek-r1:70b-llama-distill-q4_K_M", ["ollama/deepseek-r1:70b"]),
        (
            "ollama/llama2:7b-q4_K_M",
            ["ollama/llama2:7b", "ollama/llama2:latest", "ollama/llama2"],
        ),
        ("ollama/mxbai-embed-large:latest", ["ollama/mxbai-embed-large"]),
        ("ollama/phi3:latest", ["ollama/phi3"]),
        ("ollama/stable-code:3b", []),
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


def _registered_chat_model_ids():
    return {
        m.model.model_id
        for m in get_models_with_aliases()
        if isinstance(m.model, OllamaChatModel)
    }


@pytest.mark.parametrize(
    "error",
    [ollama.ResponseError("phi3 was retired", 410), ConnectionError("gone")],
    ids=["response-error", "connection-error"],
)
def test_model_whose_capabilities_cannot_be_read_is_skipped(mock_ollama_client, error):
    make_show_fail(mock_ollama_client, "phi3:latest", error)
    assert _registered_chat_model_ids() == {
        "ollama/deepseek-r1:70b-llama-distill-q4_K_M",
        "ollama/llama2:7b-q4_K_M",
        "ollama/stable-code:3b",
    }


def test_capability_read_error_keeps_server_message_and_cause(mock_ollama_client):
    error = ollama.ResponseError("phi3 was retired", 410)
    make_show_fail(mock_ollama_client, "phi3:latest", error)
    with pytest.raises(
        CapabilityReadError, match=r"^phi3 was retired \(status code: 410\)$"
    ) as excinfo:
        get_ollama_model_capabilities("e2fd6321a5fe", "phi3:latest")
    assert excinfo.value.__cause__ is error


def test_model_with_cached_capabilities_is_registered_when_show_fails(
    mock_ollama_client,
):
    assert "ollama/phi3:latest" in _registered_chat_model_ids()
    make_show_fail(
        mock_ollama_client, "phi3:latest", ollama.ResponseError("phi3 was retired", 410)
    )
    assert "ollama/phi3:latest" in _registered_chat_model_ids()


def test_failed_capabilities_read_is_not_cached(mock_ollama_client):
    show = make_show_fail(
        mock_ollama_client, "phi3:latest", ollama.ResponseError("phi3 was retired", 410)
    )
    assert "ollama/phi3:latest" not in _registered_chat_model_ids()
    mock_ollama_client.show.side_effect = show
    assert "ollama/phi3:latest" in _registered_chat_model_ids()
