import inspect
from unittest.mock import Mock

import ollama
import pytest
from helpers import assert_bearer_token
from llm import get_tools

from llm_ollama.tools import _ollama_web_fetch, _ollama_web_search


def test_registered_tools():
    tool_names = get_tools().keys()
    assert "ollama_web_search" in tool_names
    assert "ollama_web_fetch" in tool_names


@pytest.mark.parametrize(
    ("wrapper", "method_name", "call_args"),
    [
        (_ollama_web_search, "web_search", ("query text", 5)),
        (_ollama_web_fetch, "web_fetch", ("https://example.com",)),
    ],
)
def test_web_tool_uses_key_from_llm_store(
    mocker,
    bare_env,
    wrapper,
    method_name,
    call_args,
):
    """A key stored via `llm keys set ollama` reaches the web tool wrappers as a Bearer token."""
    mocker.patch("llm.get_key", return_value="stored-key")
    client = Mock()
    getattr(client, method_name).return_value = "result"
    client_class = mocker.patch("ollama.Client", return_value=client)

    result = wrapper(*call_args)

    assert result == "result"
    getattr(client, method_name).assert_called_once_with(*call_args)
    assert_bearer_token(client_class, "stored-key")


@pytest.mark.parametrize(
    ("wrapper", "client_method"),
    [
        (_ollama_web_search, ollama.Client.web_search),
        (_ollama_web_fetch, ollama.Client.web_fetch),
    ],
)
def test_web_tool_wrapper_signature_matches_ollama_client(wrapper, client_method):
    """The wrapper's parameters match ollama.Client's method, minus self."""
    client_params = list(inspect.signature(client_method).parameters.values())[1:]
    wrapper_params = list(inspect.signature(wrapper).parameters.values())
    assert wrapper_params == client_params
