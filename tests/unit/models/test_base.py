from llm_ollama.models.chat import OllamaChatModel
from llm_ollama.models.embed import OllamaEmbeddingModel


def test_get_key_returns_none_when_no_key_is_configured(mocker):
    """Ollama's key is optional, so resolution returns None instead of raising."""
    mocker.patch("llm.get_key", return_value=None)

    assert OllamaChatModel("llama2:7b").get_key() is None
    assert OllamaEmbeddingModel("mxbai-embed-large:latest").get_key() is None
