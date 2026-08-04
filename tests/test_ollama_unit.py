from unittest.mock import AsyncMock, Mock

import llm
import ollama
import pytest
from httpx import ConnectError
from llm import (
    get_async_model,
    get_embedding_models_with_aliases,
    get_model,
    get_models_with_aliases,
    get_tools,
)
from llm.plugins import load_plugins, pm

from llm_ollama import (
    Ollama,
    OllamaEmbed,
    _llm_tool_to_ollama_tool,
)
from llm_ollama.retry import ollama_error_message


@pytest.fixture
def mock_ollama_client(mocker):
    return_value = {
        "models": [
            {
                "model": "stable-code:3b",
                "digest": "aa5ab8afb86208e1c097028d63074f0142ce6079420ea6f68f219933361fd869",
                "model_info": {},
                "capabilities": ["completion"],
            },
            {
                "model": "llama2:7b",
                "digest": "78e26419b4469263f75331927a00a0284ef6544c1975b826b15abdaef17bb962",
                "model_info": {},
                "capabilities": ["completion"],
            },
            {
                "model": "llama2:7b-q4_K_M",
                "digest": "78e26419b4469263f75331927a00a0284ef6544c1975b826b15abdaef17bb962",
                "model_info": {},
                "capabilities": ["completion"],
            },
            {
                "model": "llama2:latest",
                "digest": "78e26419b4469263f75331927a00a0284ef6544c1975b826b15abdaef17bb962",
                "model_info": {},
                "capabilities": ["completion"],
            },
            {
                "model": "phi3:latest",
                "digest": "e2fd6321a5fe6bb3ac8a4e6f1cf04477fd2dea2924cf53237a995387e152ee9c",
                "model_info": {},
                "capabilities": ["completion"],
            },
            {
                "model": "mxbai-embed-large:latest",
                "digest": "468836162de7f81e041c43663fedbbba921dcea9b9fefea135685a39b2d83dd8",
                "model_info": {},
                "capabilities": ["embedding"],
            },
            {
                "model": "deepseek-r1:70b",
                "digest": "0c1615a8ca32ef41e433aa420558b4685f9fc7f3fd74119860a8e2e389cd7942",
                "model_info": {},
                "capabilities": ["completion", "thinking"],
            },
            {
                "model": "deepseek-r1:70b-llama-distill-q4_K_M",
                "digest": "0c1615a8ca32ef41e433aa420558b4685f9fc7f3fd74119860a8e2e389cd7942",
                "model_info": {},
                "capabilities": ["completion", "thinking"],
            },
        ],
    }
    client = mocker.patch("ollama.Client").return_value
    client.list.return_value = return_value
    client.show.side_effect = lambda name: next(
        ollama.ShowResponse(**m) for m in return_value["models"] if m["model"] == name
    )
    return client


def _ollama_chunk(content="", *, tool_calls=None, done=False, usage=False):
    """Build an ollama.ChatResponse chunk for the streaming code paths."""
    typed_calls = [
        ollama.Message.ToolCall(
            function=ollama.Message.ToolCall.Function(name=tc[0], arguments=tc[1]),
        )
        for tc in (tool_calls or [])
    ]
    extras = {"prompt_eval_count": 1, "eval_count": 1} if usage else {}
    return ollama.ChatResponse(
        model="llama2:7b",
        message=ollama.Message(
            role="assistant",
            content=content,
            tool_calls=typed_calls or None,
        ),
        done=done,
        **extras,
    )


def _install_async_chat(mocker, *, chunks=None, response=None, side_effect=None):
    """Mock ollama.AsyncClient.chat for streaming or non-streaming calls."""
    client = AsyncMock()
    if side_effect is not None:
        client.chat.side_effect = side_effect
    elif chunks is not None:

        async def mock_chat(*_args, **_kwargs):
            for chunk in chunks:
                yield chunk

        client.chat.return_value = mock_chat()
    else:
        client.chat.return_value = response
    mocker.patch("ollama.AsyncClient", return_value=client)
    return client


def _install_sync_chat(
    mock_ollama_client, *, chunks=None, response=None, side_effect=None
):
    """Mock ollama.Client.chat for streaming or non-streaming calls."""
    if side_effect is not None:
        mock_ollama_client.chat.side_effect = side_effect
    else:
        mock_ollama_client.chat.return_value = (
            iter(chunks) if chunks is not None else response
        )
    return mock_ollama_client


def _assert_tool_call(tc, name, arguments):
    assert tc.name == name
    assert tc.arguments == arguments


def test_plugin_is_installed():
    load_plugins()
    names = [mod.__name__ for mod in pm.get_plugins()]
    assert "llm_ollama" in names


def test_registered_tools():
    tool_names = get_tools().keys()
    assert "ollama_web_search" in tool_names
    assert "ollama_web_fetch" in tool_names


def test_registered_chat_models(mock_ollama_client):
    expected = (
        ("deepseek-r1:70b-llama-distill-q4_K_M", ["deepseek-r1:70b"]),
        ("llama2:7b-q4_K_M", ["llama2:7b", "llama2:latest", "llama2"]),
        ("phi3:latest", ["phi3"]),
        ("stable-code:3b", []),
    )
    registered_ollama_models = sorted(
        [m for m in get_models_with_aliases() if isinstance(m.model, Ollama)],
        key=lambda m: m.model.model_id,
    )
    assert len(registered_ollama_models) == len(expected)
    for model, (name, aliases) in zip(registered_ollama_models, expected):
        assert model.model.model_id == name
        assert model.aliases == aliases


def test_registered_embedding_models(mock_ollama_client):
    expected = (
        ("deepseek-r1:70b-llama-distill-q4_K_M", ["deepseek-r1:70b"]),
        ("llama2:7b-q4_K_M", ["llama2:7b", "llama2:latest", "llama2"]),
        ("mxbai-embed-large:latest", ["mxbai-embed-large"]),
        ("phi3:latest", ["phi3"]),
        ("stable-code:3b", []),
    )
    registered_ollama_models = sorted(
        [
            m
            for m in get_embedding_models_with_aliases()
            if isinstance(m.model, OllamaEmbed)
        ],
        key=lambda m: m.model.model_id,
    )
    assert len(registered_ollama_models) == len(expected)
    for model, (name, aliases) in zip(registered_ollama_models, expected):
        assert model.model.model_id == name
        assert model.aliases == aliases


@pytest.mark.parametrize(
    ("envvar_value", "expected_truncate_value"),
    [
        (None, True),
        ("True", True),
        ("true", True),
        ("yes", True),
        ("y", True),
        ("on", True),
        ("False", False),
        ("false", False),
        ("no", False),
        ("n", False),
        ("off", False),
    ],
)
def test_model_embed(
    mocker,
    envvar_value,
    expected_truncate_value,
    monkeypatch,
):
    expected = [0.1] * 1024

    client = Mock()
    client.embed.return_value = {"embeddings": [expected]}
    mocker.patch("ollama.Client", return_value=client)

    if envvar_value is not None:
        monkeypatch.setenv("OLLAMA_EMBED_TRUNCATE", envvar_value)
    else:
        monkeypatch.delenv("OLLAMA_EMBED_TRUNCATE", raising=False)

    result = OllamaEmbed("mxbai-embed-large:latest").embed("string to embed")
    assert result == expected

    _, called_kwargs = client.embed.call_args
    assert called_kwargs.get("truncate") is expected_truncate_value


def test_registered_models_when_ollama_is_down(mocker):
    client = Mock()
    client.list.side_effect = ConnectError("[Errno 111] Connection refused")
    mocker.patch("ollama.Client", return_value=client)
    assert not any(isinstance(m.model, Ollama) for m in get_models_with_aliases())


def test_sync_streaming_yields_text(mocker, mock_ollama_client):
    """A streamed sync response yields concatenated text content."""
    _install_sync_chat(
        mock_ollama_client,
        chunks=[
            _ollama_chunk("Test response 1"),
            _ollama_chunk("Test response 2", done=True, usage=True),
        ],
    )

    response = get_model("llama2:7b").prompt("Dummy Prompt")

    assert response.text() == "Test response 1Test response 2"
    mock_ollama_client.chat.assert_called_once()


@pytest.mark.asyncio
async def test_async_streaming_yields_text(mocker, mock_ollama_client):
    """A streamed async response yields concatenated text content."""
    client = _install_async_chat(
        mocker,
        chunks=[
            _ollama_chunk("Test response 1"),
            _ollama_chunk("Test response 2", done=True, usage=True),
        ],
    )

    response = get_async_model("llama2:7b").prompt("Dummy Prompt")

    assert await response.text() == "Test response 1Test response 2"
    client.chat.assert_awaited_once()


def test_sync_streaming_captures_tool_calls(mocker, mock_ollama_client):
    """Streamed chunks carrying tool_calls register on the sync response."""
    _install_sync_chat(
        mock_ollama_client,
        chunks=[
            _ollama_chunk(tool_calls=[("multiply", {"a": 6, "b": 7})]),
            _ollama_chunk("ok", done=True, usage=True),
        ],
    )

    response = get_model("llama2:7b").prompt("Dummy Prompt")
    response.text()
    tool_calls = response.tool_calls()

    assert len(tool_calls) == 1
    _assert_tool_call(tool_calls[0], "multiply", {"a": 6, "b": 7})


@pytest.mark.asyncio
async def test_async_streaming_captures_tool_calls(mocker, mock_ollama_client):
    """Streamed chunks carrying tool_calls register on the async response."""
    client = _install_async_chat(
        mocker,
        chunks=[
            _ollama_chunk(tool_calls=[("multiply", {"a": 6, "b": 7})]),
            _ollama_chunk("ok", done=True, usage=True),
        ],
    )

    response = get_async_model("llama2:7b").prompt("Dummy Prompt")
    await response.text()
    tool_calls = await response.tool_calls()

    assert len(tool_calls) == 1
    _assert_tool_call(tool_calls[0], "multiply", {"a": 6, "b": 7})
    client.chat.assert_awaited_once()


def test_sync_non_streaming_captures_tool_calls(mocker, mock_ollama_client):
    """A non-streamed sync response carrying tool_calls registers on the response."""
    _install_sync_chat(
        mock_ollama_client,
        response=_ollama_chunk(
            tool_calls=[("multiply", {"a": 6, "b": 7})],
            done=True,
            usage=True,
        ),
    )

    response = get_model("llama2:7b").prompt("Dummy Prompt", stream=False)
    response.text()
    tool_calls = response.tool_calls()

    assert len(tool_calls) == 1
    _assert_tool_call(tool_calls[0], "multiply", {"a": 6, "b": 7})


@pytest.mark.asyncio
async def test_async_non_streaming_captures_tool_calls(
    mocker,
    mock_ollama_client,
):
    """A non-streamed async response carrying tool_calls registers on the response."""
    client = _install_async_chat(
        mocker,
        response=_ollama_chunk(
            tool_calls=[("multiply", {"a": 6, "b": 7})],
            done=True,
            usage=True,
        ),
    )

    response = get_async_model("llama2:7b").prompt("Dummy Prompt", stream=False)
    await response.text()
    tool_calls = await response.tool_calls()

    assert len(tool_calls) == 1
    _assert_tool_call(tool_calls[0], "multiply", {"a": 6, "b": 7})
    client.chat.assert_awaited_once()


def _make_kwargs_tool(name="sql_query", description="Run a SQL query"):
    def _impl(**kwargs):
        pass

    return llm.Tool(
        name=name,
        description=description,
        input_schema={
            "type": "object",
            "properties": {
                "database": {"type": "string", "description": "Database name"},
                "sql": {"type": "string", "description": "SQL to execute"},
                "display": {"type": "string", "enum": ["rows", "csv", "both"]},
            },
            "required": ["database", "sql"],
        },
        implementation=_impl,
    )


def test_tool_conversion_typed_signature():
    def multiply(a: int, b: int) -> int:
        """Multiply two numbers."""
        return a * b

    tool = llm.Tool.function(multiply)
    ollama_tool = _llm_tool_to_ollama_tool(tool)

    assert ollama_tool.function.name == "multiply"
    assert ollama_tool.function.description == "Multiply two numbers."
    params = ollama_tool.function.parameters
    assert set(params.properties.keys()) == {"a", "b"}
    assert params.properties["a"].type == "integer"
    assert params.properties["b"].type == "integer"


def test_tool_conversion_kwargs_uses_input_schema():
    ollama_tool = _llm_tool_to_ollama_tool(_make_kwargs_tool())

    assert ollama_tool.function.name == "sql_query"
    assert ollama_tool.function.description == "Run a SQL query"
    params = ollama_tool.function.parameters
    assert set(params.properties.keys()) == {"database", "sql", "display"}
    assert params.properties["database"].type == "string"
    assert params.properties["display"].enum == ["rows", "csv", "both"]
    assert params.required == ["database", "sql"]


def test_tool_conversion_name_and_description_override():
    tool = _make_kwargs_tool(name="custom_name", description="Custom description")
    ollama_tool = _llm_tool_to_ollama_tool(tool)

    assert ollama_tool.function.name == "custom_name"
    assert ollama_tool.function.description == "Custom description"


class TestOllamaErrorHandling:
    """Behaviour around transient HTTP errors from the Ollama server."""

    def test_410_gone_warns_and_yields_empty_response_sync(
        self, mocker, mock_ollama_client, capsys
    ):
        """A 410 during a sync chat should log a warning and not crash the session."""
        _install_sync_chat(
            mock_ollama_client,
            side_effect=ollama.ResponseError("model is gone", status_code=410),
        )

        response = get_model("llama2:7b").prompt("Hello", stream=False)
        text = response.text()

        assert text == ""
        captured = capsys.readouterr()
        assert captured.err
        assert "410" in captured.err or "gone" in captured.err.lower()

    @pytest.mark.asyncio
    async def test_410_gone_warns_and_yields_empty_response_async(
        self, mocker, mock_ollama_client, capsys
    ):
        """A 410 during an async chat should log a warning and not crash the session."""
        client = _install_async_chat(
            mocker,
            side_effect=ollama.ResponseError("model is gone", status_code=410),
        )

        response = get_async_model("llama2:7b").prompt("Hello", stream=False)
        text = await response.text()

        assert text == ""
        captured = capsys.readouterr()
        assert captured.err
        assert "410" in captured.err or "gone" in captured.err.lower()
        client.chat.assert_awaited_once()

    def test_429_retries_then_warns_sync(self, mocker, mock_ollama_client, capsys):
        """A 429 should be retried a few times before giving up with a warning."""
        _install_sync_chat(
            mock_ollama_client,
            side_effect=[
                ollama.ResponseError("rate limited", status_code=429),
                ollama.ResponseError("rate limited", status_code=429),
                _ollama_chunk("ok", done=True, usage=True),
            ],
        )

        response = get_model("llama2:7b").prompt("Hello", stream=False)
        assert response.text() == "ok"
        assert mock_ollama_client.chat.call_count == 3

    @pytest.mark.asyncio
    async def test_429_retries_then_warns_async(
        self, mocker, mock_ollama_client, capsys
    ):
        """A 429 should be retried a few times before giving up with a warning."""
        client = _install_async_chat(
            mocker,
            side_effect=[
                ollama.ResponseError("rate limited", status_code=429),
                _ollama_chunk("ok", done=True, usage=True),
            ],
        )

        response = get_async_model("llama2:7b").prompt("Hello", stream=False)
        assert await response.text() == "ok"
        assert client.chat.await_count == 2

    def test_5xx_retries_then_warns_sync(self, mocker, mock_ollama_client, capsys):
        """500 class errors should be retried with best-of-5 semantics."""
        _install_sync_chat(
            mock_ollama_client,
            side_effect=[
                ollama.ResponseError("overloaded", status_code=503),
                ollama.ResponseError("bad gateway", status_code=502),
                ollama.ResponseError("server error", status_code=500),
                _ollama_chunk("ok", done=True, usage=True),
            ],
        )

        response = get_model("llama2:7b").prompt("Hello", stream=False)
        assert response.text() == "ok"
        assert mock_ollama_client.chat.call_count == 4

    @pytest.mark.asyncio
    async def test_5xx_retries_then_warns_async(
        self, mocker, mock_ollama_client, capsys
    ):
        """500 class errors should be retried with best-of-5 semantics."""
        client = _install_async_chat(
            mocker,
            side_effect=[
                ollama.ResponseError("gateway timeout", status_code=504),
                ollama.ResponseError("not implemented", status_code=501),
                _ollama_chunk("ok", done=True, usage=True),
            ],
        )

        response = get_async_model("llama2:7b").prompt("Hello", stream=False)
        assert await response.text() == "ok"
        assert client.chat.await_count == 3

    def test_410_not_retried_sync(self, mocker, mock_ollama_client):
        """410 is permanent; we should not retry it."""
        _install_sync_chat(
            mock_ollama_client,
            side_effect=ollama.ResponseError("model is gone", status_code=410),
        )

        response = get_model("llama2:7b").prompt("Hello", stream=False)
        assert response.text() == ""
        assert mock_ollama_client.chat.call_count == 1

    def test_error_message_extraction(self):
        assert (
            ollama_error_message(ollama.ResponseError("oops", status_code=500))
            == "oops"
        )
        assert "500" in ollama_error_message(
            ollama.ResponseError("oops", status_code=500), include_status=True
        )
        assert ollama_error_message(ValueError("plain")) == "plain"
        assert ollama_error_message(Exception()) == ""
