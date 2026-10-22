"""Discovery of Ollama models and their registration with llm."""

from collections import defaultdict
from collections.abc import Callable, Iterator

import llm
import ollama

from llm_ollama.auth import get_client
from llm_ollama.cache import Cache
from llm_ollama.models.chat import AsyncOllamaChatModel, OllamaChatModel
from llm_ollama.models.embed import OllamaEmbeddingModel

cache = Cache(llm.user_dir() / "llm-ollama" / "cache")


class CapabilityReadError(Exception):
    """The Ollama server could not report a model's capabilities."""


@llm.hookimpl
def register_models(register: Callable[..., None]) -> None:
    """Register a chat model for each Ollama model that supports completion.

    Models whose capabilities cannot be read are left out.
    """
    for digest, name, alias_names in _distinct_models():
        try:
            capabilities = get_ollama_model_capabilities(digest, name)
        except CapabilityReadError:
            continue
        if "completion" not in capabilities:
            continue
        supports_tools = "tools" in capabilities
        model_id = _to_model_id(name)
        register(
            OllamaChatModel(
                model_id=model_id, ollama_name=name, supports_tools=supports_tools
            ),
            AsyncOllamaChatModel(
                model_id=model_id, ollama_name=name, supports_tools=supports_tools
            ),
            aliases=[_to_model_id(alias_name) for alias_name in alias_names],
        )


@llm.hookimpl
def register_embedding_models(register: Callable[..., None]) -> None:
    """Register an embedding model for each distinct Ollama model."""
    for _, name, alias_names in _distinct_models():
        register(
            OllamaEmbeddingModel(model_id=_to_model_id(name), ollama_name=name),
            aliases=[_to_model_id(alias_name) for alias_name in alias_names],
        )


def _to_model_id(name: str) -> str:
    """Convert an Ollama model name to the ID or alias llm knows it by."""
    return f"ollama/{name}"


def _distinct_models() -> Iterator[tuple[str, str, tuple[str, ...]]]:
    """Yield the digest, primary name and alias names of each model on the Ollama server.

    Names that share a digest are one model, and a name tagged ``:latest`` also has
    its bare form as an alias name.
    """
    names_by_digest = defaultdict(list)
    for model in get_ollama_models():
        name, digest = model["model"], model["digest"]
        names_by_digest[digest].append(name)
        if name.endswith(":latest"):
            names_by_digest[digest].append(name.removesuffix(":latest"))
    for digest, names in names_by_digest.items():
        name, alias_names = _pick_primary_name(names)
        yield digest, name, alias_names


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
        The primary model name and a tuple with the alias names.

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


def get_ollama_models() -> list[dict]:
    """List the models on the Ollama server; empty if the server cannot be queried."""
    try:
        return get_client().list()["models"]
    except Exception:  # noqa: BLE001
        return []


# `digest` is unused in the body; the cache reads it by name as the cache key.
@cache("model_capabilities", key="digest")
def get_ollama_model_capabilities(
    digest: str,  # noqa: ARG001
    name: str,
) -> list[str]:
    """Return the capabilities the Ollama server reports for the model named ``name``.

    The result is cached on disk by ``digest``, so the server is asked about ``name``
    only on a cache miss. A failure is not cached.

    Raises
    ------
    CapabilityReadError
        If the server answers with an error status, for example for a retired model,
        or cannot be reached.

    """
    try:
        return get_client().show(name).capabilities or []
    except (ollama.ResponseError, ConnectionError) as e:
        raise CapabilityReadError(str(e)) from e
