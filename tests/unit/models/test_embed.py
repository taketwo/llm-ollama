from unittest.mock import Mock

import pytest
from helpers import assert_bearer_token

from llm_ollama.models.embed import OllamaEmbeddingModel


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
def test_model_embed(
    mocker,
    envvar_value,
    expected_truncate_value,
    monkeypatch,
):
    expected = [0.1] * 1024

    client = Mock()
    client.embed.return_value = {"embeddings": [expected]}
    mocker.patch("ollama.Client", return_value=client)

    if envvar_value is not None:
        monkeypatch.setenv("OLLAMA_EMBED_TRUNCATE", envvar_value)
    else:
        monkeypatch.delenv("OLLAMA_EMBED_TRUNCATE", raising=False)

    result = OllamaEmbeddingModel(
        model_id="mxbai-embed-large:latest", ollama_name="mxbai-embed-large:latest"
    ).embed("string to embed")
    assert result == expected

    _, called_kwargs = client.embed.call_args
    assert called_kwargs.get("truncate") is expected_truncate_value


def test_embed_key_reaches_client_as_bearer_token(mocker):
    """A per-call key travels from embed() through to the client's auth header."""
    client = Mock()
    client.embed.return_value = {"embeddings": [[0.1]]}
    client_class = mocker.patch("ollama.Client", return_value=client)

    OllamaEmbeddingModel(
        model_id="mxbai-embed-large:latest", ollama_name="mxbai-embed-large:latest"
    ).embed("string to embed", key="caller-key")

    assert_bearer_token(client_class, "caller-key")
