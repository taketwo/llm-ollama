import pytest
import yaml
from click.testing import CliRunner
from llm.cli import cli

from llm_ollama import registry
from llm_ollama.cache import Cache


def test_models_prints_every_name_with_digest_and_capabilities(mock_ollama_client):
    """Names that share a digest each get a row, sorted by name."""
    result = CliRunner().invoke(cli, ["ollama", "models"])

    assert result.exit_code == 0, result.output
    assert [line.rstrip() for line in result.output.splitlines()] == [
        "model                                   digest          capabilities",
        "deepseek-r1:70b                         0c1615a8ca32    completion, thinking",
        "deepseek-r1:70b-llama-distill-q4_K_M    0c1615a8ca32    completion, thinking",
        "llama2:7b                               78e26419b446    completion",
        "llama2:7b-q4_K_M                        78e26419b446    completion",
        "llama2:latest                           78e26419b446    completion",
        "mxbai-embed-large:latest                468836162de7    embedding",
        "phi3:latest                             e2fd6321a5fe    completion",
        "stable-code:3b                          aa5ab8afb862    completion",
    ]


@pytest.mark.parametrize(
    "content",
    [
        pytest.param("42", id="scalar-root"),
        pytest.param(f"version: {Cache.CACHE_VERSION}\ndata: null", id="null-data"),
        pytest.param("data: [unfinished", id="parser-error"),
    ],
)
def test_models_repairs_invalid_capability_cache(mock_ollama_client, content):
    cache_file = registry.cache.cache_dir / "model_capabilities.yaml"
    cache_file.write_text(content)

    result = CliRunner().invoke(cli, ["ollama", "models"])

    assert result.exit_code == 0, result.output
    models = mock_ollama_client.list.return_value["models"]
    assert all(model["model"] in result.output for model in models)
    expected = {model["digest"]: model["capabilities"] for model in models}
    assert mock_ollama_client.show.call_count == len(expected)
    assert yaml.safe_load(cache_file.read_text()) == {
        "version": Cache.CACHE_VERSION,
        "data": expected,
    }

    # Repeating the real Click command reuses the repaired capability entries.
    repeated = CliRunner().invoke(cli, ["ollama", "models"])
    assert repeated.exit_code == 0, repeated.output
    assert repeated.output == result.output
    assert mock_ollama_client.show.call_count == len(expected)
