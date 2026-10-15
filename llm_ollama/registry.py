"""Discovery of Ollama models and their registration with llm."""

from collections import defaultdict

import llm

from llm_ollama.auth import get_client
from llm_ollama.cache import Cache
from llm_ollama.models.chat import AsyncOllama, Ollama
from llm_ollama.models.embed import OllamaEmbed

cache = Cache(llm.user_dir() / "llm-ollama" / "cache")


@llm.hookimpl
def register_models(register, model_aliases=None):
    models = defaultdict(list)
    for model in get_ollama_models():
        name, digest = model["model"], model["digest"]
        models[digest].append(name)
        if name.endswith(":latest"):
            models[digest].append(name[: -len(":latest")])
    for digest, names in models.items():
        name, aliases = _pick_primary_name(names)
        capabilities = get_ollama_model_capabilities(digest, name)
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
    for model in get_ollama_models():
        models[model["digest"]].append(model["model"])
        if model["model"].endswith(":latest"):
            models[model["digest"]].append(model["model"][: -len(":latest")])
    for names in models.values():
        name, aliases = _pick_primary_name(names)
        register(OllamaEmbed(name), aliases=aliases)


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


def get_ollama_models() -> list[dict]:
    """List the models on the Ollama server; empty if the server cannot be queried."""
    try:
        return get_client().list()["models"]
    except:
        return []


@cache("model_capabilities", key="digest")
def get_ollama_model_capabilities(digest: str, model: str) -> list[str]:
    """Return the capabilities the Ollama server reports for a model.

    The result is cached on disk by ``digest``, so the server is asked about ``model``
    only on a cache miss. Raises if the server cannot be reached or does not know the
    model.
    """
    return get_client().show(model).capabilities or []
