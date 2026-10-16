"""The ``llm ollama`` command group."""

import click
import llm
from llm.utils import dicts_to_table_string

from llm_ollama.registry import get_ollama_model_capabilities, get_ollama_models


@llm.hookimpl
def register_commands(cli: click.Group) -> None:
    """Add the ``ollama`` command group to the ``llm`` CLI."""

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
