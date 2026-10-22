"""The ``llm ollama`` command group."""

import click
import llm
from llm.utils import dicts_to_table_string

from llm_ollama.registry import (
    CapabilityReadError,
    get_ollama_model_capabilities,
    get_ollama_models,
)


@llm.hookimpl
def register_commands(cli: click.Group) -> None:
    """Add the ``ollama`` command group to the ``llm`` CLI."""

    @cli.group(name="ollama")
    def ollama_group() -> None:
        """Commands for working with models hosted on Ollama server."""

    @ollama_group.command()
    def models() -> None:
        """List models that are available on Ollama server.

        Models whose capabilities cannot be read are listed after the table, with the
        error, on stderr.
        """
        to_print = []
        errors: dict[str, Exception] = {}
        for model in get_ollama_models():
            try:
                capabilities = get_ollama_model_capabilities(
                    model["digest"], model["model"]
                )
            except CapabilityReadError as e:
                errors[model["model"]] = e
                continue
            to_print.append(
                {
                    "model": model["model"],
                    "digest": model["digest"][:12],
                    "capabilities": ", ".join(sorted(capabilities)),
                }
            )
        to_print.sort(key=lambda x: x["model"])
        done = dicts_to_table_string(["model", "digest", "capabilities"], to_print)
        # click.echo flushes, so the table precedes the errors when stdout is piped
        click.echo("\n".join(done))
        if errors:
            click.echo(
                "\nNot registered, because their capabilities could not be read:",
                err=True,
            )
            for name, error in sorted(errors.items()):
                click.echo(f"  {name}: {error}", err=True)
