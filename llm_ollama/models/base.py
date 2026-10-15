"""Behavior shared by every Ollama model class."""

from llm_ollama.auth import resolve_key


class BaseMixin:
    """Display name and key resolution for an Ollama model.

    List it before the ``llm`` base class, whose ``__str__`` and ``get_key`` it
    overrides.
    """

    model_id: str

    def __str__(self) -> str:
        return f"Ollama: {self.model_id}"

    def get_key(self, explicit_key: str | None = None) -> str | None:
        return resolve_key(explicit_key)
