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

from llm_ollama import AsyncOllama, Ollama, OllamaEmbed, _llm_tool_to_ollama_tool


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


def _ollama_chunk(
    content="",
    *,
    thinking=None,
    tool_calls=None,
    done=False,
    usage=False,
):
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
            thinking=thinking,
            tool_calls=typed_calls or None,
        ),
        done=done,
        **extras,
    )


def _install_async_chat(mocker, *, chunks=None, response=None):
    """Mock ollama.AsyncClient.chat for streaming or non-streaming calls."""
    client = AsyncMock()
    if chunks is not None:

        async def mock_chat(*_args, **_kwargs):
            for chunk in chunks:
                yield chunk

        client.chat.return_value = mock_chat()
    else:
        client.chat.return_value = response
    mocker.patch("ollama.AsyncClient", return_value=client)
    return client


def _install_sync_chat(mock_ollama_client, *, chunks=None, response=None):
    """Mock ollama.Client.chat for streaming or non-streaming calls."""
    mock_ollama_client.chat.return_value = (
        iter(chunks) if chunks is not None else response
    )
    return mock_ollama_client


def _assert_tool_call(tc, name, arguments):
    assert tc.name == name
    assert tc.arguments == arguments


def _assert_bearer_token(client_class, token):
    _, kwargs = client_class.call_args
    assert kwargs["headers"]["Authorization"] == f"Bearer {token}"


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


def test_get_key_returns_none_when_no_key_is_configured(mocker):
    """Ollama's key is optional, so resolution returns None instead of raising."""
    mocker.patch("llm.get_key", return_value=None)

    assert Ollama("llama2:7b").get_key() is None
    assert OllamaEmbed("mxbai-embed-large:latest").get_key() is None


def test_chat_key_reaches_client_as_bearer_token(mocker, bare_env):
    """A per-call key travels from prompt() through to the client's auth header."""
    client = Mock()
    client.chat.return_value = iter([_ollama_chunk("ok", done=True, usage=True)])
    client_class = mocker.patch("ollama.Client", return_value=client)

    Ollama("llama2:7b").prompt("Dummy Prompt", key="caller-key").text()

    _assert_bearer_token(client_class, "caller-key")


@pytest.mark.asyncio
async def test_async_chat_key_reaches_client_as_bearer_token(mocker, bare_env):
    """The async client is built from the resolved key just as the sync one is."""

    async def mock_chat(*_args, **_kwargs):
        yield _ollama_chunk("ok", done=True, usage=True)

    client = AsyncMock()
    client.chat.return_value = mock_chat()
    client_class = mocker.patch("ollama.AsyncClient", return_value=client)

    await AsyncOllama("llama2:7b").prompt("Dummy Prompt", key="caller-key").text()

    _assert_bearer_token(client_class, "caller-key")


def test_embed_key_reaches_client_as_bearer_token(mocker, bare_env):
    """A per-call key travels from embed() through to the client's auth header."""
    client = Mock()
    client.embed.return_value = {"embeddings": [[0.1]]}
    client_class = mocker.patch("ollama.Client", return_value=client)

    OllamaEmbed("mxbai-embed-large:latest").embed("string to embed", key="caller-key")

    _assert_bearer_token(client_class, "caller-key")


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
    client.chat.assert_called_once()


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
    _install_async_chat(
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
    _install_async_chat(
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


def test_tool_conversion_kwargs_empty_schema():
    tool = llm.Tool(
        name="t",
        input_schema={"type": "object", "properties": {}},
        implementation=lambda **kwargs: "x",
    )
    ollama_tool = _llm_tool_to_ollama_tool(tool)

    assert ollama_tool.function.name == "t"
    params = ollama_tool.function.parameters
    assert dict(params.properties or {}) == {}
    assert not params.required


def test_tool_conversion_name_and_description_override():
    tool = _make_kwargs_tool(name="custom_name", description="Custom description")
    ollama_tool = _llm_tool_to_ollama_tool(tool)

    assert ollama_tool.function.name == "custom_name"
    assert ollama_tool.function.description == "Custom description"


def test_thinking_surfaces_as_reasoning_part(mocker, mock_ollama_client):
    """A chunk's thinking content lands as a ReasoningPart on the response."""
    from llm.parts import ReasoningPart

    _install_sync_chat(
        mock_ollama_client,
        chunks=[
            _ollama_chunk(thinking="Let me think..."),
            _ollama_chunk("Final answer", done=True, usage=True),
        ],
    )

    response = get_model("deepseek-r1:70b").prompt("Dummy Prompt")
    assert response.text() == "Final answer"

    parts = response.messages()[-1].parts
    reasoning = [p for p in parts if isinstance(p, ReasoningPart)]
    assert len(reasoning) == 1
    assert reasoning[0].text == "Let me think..."


def test_hide_reasoning_suppresses_reasoning_events(mocker, mock_ollama_client):
    """prompt.hide_reasoning drops reasoning events without altering the request."""
    from llm.parts import ReasoningPart

    client = _install_sync_chat(
        mock_ollama_client,
        chunks=[
            _ollama_chunk(thinking="Let me think..."),
            _ollama_chunk("Final answer", done=True, usage=True),
        ],
    )

    response = get_model("deepseek-r1:70b").prompt(
        "Dummy Prompt",
        hide_reasoning=True,
        think=True,
    )
    assert response.text() == "Final answer"
    parts = response.messages()[-1].parts
    assert not any(isinstance(p, ReasoningPart) for p in parts)
    # Request still asks the model to think — hide_reasoning only suppresses display.
    _, kwargs = client.chat.call_args
    assert kwargs.get("think") is True


def test_streamed_response_json_matches_non_streamed(mocker, mock_ollama_client):
    """Streaming reassembles the payload Ollama would have returned unstreamed."""
    _install_sync_chat(
        mock_ollama_client,
        chunks=[
            _ollama_chunk(thinking="Let me "),
            _ollama_chunk(thinking="think..."),
            _ollama_chunk(tool_calls=[("multiply", {"a": 6, "b": 7})]),
            _ollama_chunk("Answer: "),
            _ollama_chunk("42", done=True, usage=True),
        ],
    )
    streamed = get_model("deepseek-r1:70b").prompt("Dummy Prompt")
    streamed.text()

    _install_sync_chat(
        mock_ollama_client,
        response=_ollama_chunk(
            "Answer: 42",
            thinking="Let me think...",
            tool_calls=[("multiply", {"a": 6, "b": 7})],
            done=True,
            usage=True,
        ),
    )
    non_streamed = get_model("deepseek-r1:70b").prompt("Dummy Prompt", stream=False)
    non_streamed.text()

    assert streamed.response_json == non_streamed.response_json
    message = streamed.response_json["message"]
    assert message["content"] == "Answer: 42"
    assert message["thinking"] == "Let me think..."
    assert message["tool_calls"][0]["function"]["name"] == "multiply"
    # Envelope comes from the final chunk — the only one carrying usage counts.
    assert streamed.response_json["prompt_eval_count"] == 1


@pytest.mark.asyncio
async def test_async_streaming_reassembles_response_json(mocker, mock_ollama_client):
    """The async pump drives the same accumulator down a separate code path."""
    _install_async_chat(
        mocker,
        chunks=[
            _ollama_chunk("Test response 1"),
            _ollama_chunk("Test response 2", done=True, usage=True),
        ],
    )

    response = get_async_model("llama2:7b").prompt("Dummy Prompt")
    await response.text()

    message = response.response_json["message"]
    assert message["content"] == "Test response 1Test response 2"


def test_tool_call_reply_round_trip(mocker, mock_ollama_client):
    """response.reply() round-trips a tool call + result back into the next request."""
    from llm import ToolResult

    mock_ollama_client.chat.side_effect = [
        iter(
            [
                _ollama_chunk(tool_calls=[("multiply", {"a": 6, "b": 7})]),
                _ollama_chunk("", done=True, usage=True),
            ],
        ),
        iter(
            [
                _ollama_chunk("The answer is 42", done=True, usage=True),
            ],
        ),
    ]

    first = get_model("llama2:7b").prompt("What is 6 times 7?")
    first.text()
    tool_calls = first.tool_calls()
    assert len(tool_calls) == 1
    assert tool_calls[0].tool_call_id is not None

    second = first.reply(
        tool_results=[
            ToolResult(
                name="multiply",
                output="42",
                tool_call_id=tool_calls[0].tool_call_id,
            ),
        ],
    )
    assert second.text() == "The answer is 42"

    # Verify the second turn's chat call carried the assistant tool_call and the
    # tool result back to Ollama in wire format.
    second_call_messages = mock_ollama_client.chat.call_args_list[1].kwargs["messages"]
    roles = [m["role"] for m in second_call_messages]
    assert roles == ["user", "assistant", "tool"]
    assistant_msg = second_call_messages[1]
    assert assistant_msg["tool_calls"]
    assert assistant_msg["tool_calls"][0].function.name == "multiply"
    assert assistant_msg["tool_calls"][0].function.arguments == {"a": 6, "b": 7}
    tool_msg = second_call_messages[2]
    assert tool_msg == {"role": "tool", "content": "42", "name": "multiply"}


def test_build_messages_handles_multi_attachment_history(mock_ollama_client):
    """A conversation with attachments across turns survives the new Part chain."""
    from pathlib import Path

    from llm import Attachment

    png_bytes = (Path(__file__).parent / "data" / "box.png").read_bytes()
    att_a = Attachment(content=png_bytes, type="image/png")
    att_b = Attachment(content=png_bytes + b"\x00", type="image/png")

    model = get_model("llama2:7b")
    conversation = model.conversation()

    mock_ollama_client.chat.side_effect = [
        iter([_ollama_chunk("first reply", done=True, usage=True)]),
        iter([_ollama_chunk("second reply", done=True, usage=True)]),
    ]

    conversation.prompt("look at this", attachments=[att_a]).text()
    conversation.prompt("and this", attachments=[att_b]).text()

    second_messages = mock_ollama_client.chat.call_args_list[1].kwargs["messages"]
    user_messages = [m for m in second_messages if m["role"] == "user"]
    assert len(user_messages) == 2
    assert user_messages[0]["images"] == [att_a.base64_content()]
    assert user_messages[1]["images"] == [att_b.base64_content()]
