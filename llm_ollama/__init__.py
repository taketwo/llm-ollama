import llm
from llm.utils import dicts_to_table_string

from llm_ollama.registry import (
    get_ollama_model_capabilities,
    get_ollama_models,
    register_embedding_models as register_embedding_models,
    register_models as register_models,
)
from llm_ollama.tools import register_tools as register_tools


@llm.hookimpl
def register_commands(cli):
    @cli.group(name="ollama")
    def ollama_group() -> None:
        """Commands for working with models hosted on Ollama server."""

    @ollama_group.command()
    def models() -> None:
        """List models that are available on Ollama server."""
        to_print = [
            {
                "model": model["model"],
                "digest": model["digest"][:12],
                "capabilities": ", ".join(
                    sorted(
                        get_ollama_model_capabilities(model["digest"], model["model"])
                    ),
                ),
            }
            for model in get_ollama_models()
        ]
        to_print.sort(key=lambda x: x["model"])
        done = dicts_to_table_string(["model", "digest", "capabilities"], to_print)
        print("\n".join(done))
