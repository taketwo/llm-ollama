"""Plugin entry point: llm registers this module, so it re-exports every hook."""

from llm_ollama.commands import register_commands as register_commands
from llm_ollama.registry import (
    register_embedding_models as register_embedding_models,
    register_models as register_models,
)
from llm_ollama.tools import register_tools as register_tools
