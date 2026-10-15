from click.testing import CliRunner
from llm.cli import cli


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
