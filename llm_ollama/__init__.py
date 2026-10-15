import os
import warnings
from collections import defaultdict

import llm
from llm.utils import dicts_to_table_string
from pydantic import TypeAdapter, ValidationError

from llm_ollama.auth import get_client, resolve_key
from llm_ollama.cache import Cache
from llm_ollama.models.chat import AsyncOllama, Ollama
from llm_ollama.tools import register_tools as register_tools

cache = Cache(llm.user_dir() / "llm-ollama" / "cache")


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
                        _get_ollama_model_capabilities(model["digest"], model["model"])
                    ),
                ),
            }
            for model in _get_ollama_models()
        ]
        to_print.sort(key=lambda x: x["model"])
        done = dicts_to_table_string(["model", "digest", "capabilities"], to_print)
        print("\n".join(done))


@llm.hookimpl
def register_models(register, model_aliases=None):
    models = defaultdict(list)
    for model in _get_ollama_models():
        name, digest = model["model"], model["digest"]
        models[digest].append(name)
        if name.endswith(":latest"):
            models[digest].append(name[: -len(":latest")])
    for digest, names in models.items():
        name, aliases = _pick_primary_name(names)
        capabilities = _get_ollama_model_capabilities(digest, name)
        if "completion" not in capabilities:
            continue
        supports_tools = "tools" in capabilities
        register(
            Ollama(name, supports_tools=supports_tools),
            AsyncOllama(name, supports_tools=supports_tools),
            aliases=aliases,
        )


@llm.hookimpl
def register_embedding_models(register):
    models = defaultdict(list)
    for model in _get_ollama_models():
        models[model["digest"]].append(model["model"])
        if model["model"].endswith(":latest"):
            models[model["digest"]].append(model["model"][: -len(":latest")])
    for names in models.values():
        name, aliases = _pick_primary_name(names)
        register(OllamaEmbed(name), aliases=aliases)


class OllamaEmbed(llm.EmbeddingModel):
    supports_text = True
    supports_binary = False
    batch_size = 8

    def __init__(self, model_id):
        self.model_id = model_id
        self.truncate = True

        # Read OLLAMA_EMBED_TRUNCATE environment variable to decide if truncation
        # is enabled. If truncation is disabled and the input is too long, ollama.embed
        # call will fail.
        if (truncate := os.getenv("OLLAMA_EMBED_TRUNCATE")) is not None:
            try:
                self.truncate = TypeAdapter(bool).validate_python(truncate)
            except ValidationError:
                warnings.warn(
                    f"OLLAMA_EMBED_TRUNCATE is set to '{truncate}', which is not a valid boolean value; defaulting to True",
                )

    def __str__(self) -> str:
        return f"Ollama: {self.model_id}"

    def get_key(self, explicit_key: str | None = None) -> str | None:
        return resolve_key(explicit_key)

    def embed_batch(self, items, *, key: str | None = None):
        result = get_client(key=key).embed(
            model=self.model_id,
            input=items,
            truncate=self.truncate,
        )
        yield from result["embeddings"]


def _pick_primary_name(names: list[str]) -> tuple[str, tuple[str, ...]]:
    """Pick the primary model name from a list of names.

    The picking algorithm prefers names with the most specific tag, e.g. "llama2:7b-q4_K_M"
    over "llama2:7b" over "llama2:latest" over "llama2".

    Parameters
    ----------
    names : list[str]
        A non-empty list of model names.

    Returns
    -------
    tuple[str, tuple[str, ...]]
        The primary model name and a tuple with the secondary names.

    """
    if len(names) == 1:
        return names[0], ()
    sorted_names = sorted(
        names,
        key=lambda name: (
            ":" not in name,  # Prefer names with a colon
            name.endswith(":latest"),  # Non-latest tags preferred over latest
            -len(name),  # Prefer longer names (likely more specific/quantized)
            name,  # Finally sort by name itself
        ),
    )
    return sorted_names[0], tuple(sorted_names[1:])


def _get_ollama_models() -> list[dict]:
    """Get a list of models available on Ollama.

    Returns
    -------
    list[dict]
        A list of models available on Ollama. If the Ollama server is down, an empty
        list is returned.

    """
    try:
        return get_client().list()["models"]
    except:
        return []


@cache("model_capabilities", key="digest")
def _get_ollama_model_capabilities(digest: str, model: str) -> list[str]:
    """Get a list of capabilities for a given Ollama model.

    This function may raise an exception if the Ollama server is down or the model does
    not exist.

    Returns
    -------
    list[str]
        A list of capabilities for the given model.

    """
    return get_client().show(model).capabilities or []
