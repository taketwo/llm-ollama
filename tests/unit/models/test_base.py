from llm_ollama.models.chat import OllamaChatModel
from llm_ollama.models.embed import OllamaEmbeddingModel


def test_get_key_returns_none_when_no_key_is_configured(mocker):
    """Ollama's key is optional, so resolution returns None instead of raising."""
    mocker.patch("llm.get_key", return_value=None)

    chat_model = OllamaChatModel(
        model_id="llama2:7b", ollama_name="llama2:7b", capabilities=["completion"]
    )
    embedding_model = OllamaEmbeddingModel(
        model_id="mxbai-embed-large:latest", ollama_name="mxbai-embed-large:latest"
    )

    assert chat_model.get_key() is None
    assert embedding_model.get_key() is None
