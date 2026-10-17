"""Embedding models served by Ollama."""

import os
import warnings
from collections.abc import Iterable, Iterator, Sequence
from typing import cast

import llm
from pydantic import TypeAdapter, ValidationError

from llm_ollama.auth import get_client
from llm_ollama.models.base import BaseMixin


class OllamaEmbeddingModel(BaseMixin, llm.EmbeddingModel):
    """An embedding model served by Ollama."""

    supports_text = True
    supports_binary = False
    batch_size = 8

    def __init__(self, *, model_id: str, ollama_name: str) -> None:
        """Create the model llm knows as ``model_id`` and Ollama as ``ollama_name``.

        Truncation follows ``OLLAMA_EMBED_TRUNCATE``, default on.
        """
        self.model_id = model_id
        self.ollama_name = ollama_name
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
                    stacklevel=2,
                )

    def embed_batch(
        self, items: Iterable[str | bytes], *, key: str | None = None
    ) -> Iterator[list[float]]:
        """Embed each of ``items`` on the Ollama server, yielding one vector per item."""
        result = get_client(key=key).embed(
            model=self.ollama_name,
            # supports_binary is False, so llm passes only strings.
            input=cast("Sequence[str]", items),
            truncate=self.truncate,
        )
        yield from result["embeddings"]
