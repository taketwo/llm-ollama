from unittest.mock import AsyncMock, Mock

import llm
import ollama
import pydantic
import pytest
from helpers import assert_bearer_token
from llm import Attachment, ToolResult, get_async_model, get_model
from llm.parts import ReasoningPart

from llm_ollama.models.chat import (
    AsyncOllamaChatModel,
    OllamaChatModel,
    _llm_tool_to_ollama_tool,
)


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


def test_tool_conversion_no_input_schema_uses_signature():
    """A tool with a real signature and no input_schema keeps its signature-derived parameters."""

    def search(query: str, max_results: int = 3) -> str:
        """Search."""
        return "x"

    tool = llm.Tool(name="search", description="Search", implementation=search)
    ollama_tool = _llm_tool_to_ollama_tool(tool)

    params = ollama_tool.function.parameters
    assert set(params.properties.keys()) == {"query", "max_results"}
    assert params.required == ["query", "max_results"]


def test_chat_key_reaches_client_as_bearer_token(mocker):
    """A per-call key travels from prompt() through to the client's auth header."""
    client = Mock()
    client.chat.return_value = iter([_ollama_chunk("ok", done=True, usage=True)])
    client_class = mocker.patch("ollama.Client", return_value=client)

    OllamaChatModel(
        model_id="llama2:7b", ollama_name="llama2:7b", capabilities=["completion"]
    ).prompt("Dummy Prompt", key="caller-key").text()

    assert_bearer_token(client_class, "caller-key")


@pytest.mark.asyncio
async def test_async_chat_key_reaches_client_as_bearer_token(mocker):
    """The async client is built from the resolved key just as the sync one is."""

    async def mock_chat(*_args, **_kwargs):
        yield _ollama_chunk("ok", done=True, usage=True)

    client = AsyncMock()
    client.chat.return_value = mock_chat()
    client_class = mocker.patch("ollama.AsyncClient", return_value=client)

    await (
        AsyncOllamaChatModel(
            model_id="llama2:7b", ollama_name="llama2:7b", capabilities=["completion"]
        )
        .prompt("Dummy Prompt", key="caller-key")
        .text()
    )

    assert_bearer_token(client_class, "caller-key")


@pytest.mark.parametrize("stream", [True, False])
def test_sync_chat_sends_ollama_name(mock_ollama_client, stream):
    """The server is asked for the model by its Ollama name, not its llm ID."""
    reply = _ollama_chunk("ok", done=True, usage=True)
    if stream:
        _install_sync_chat(mock_ollama_client, chunks=[reply])
    else:
        _install_sync_chat(mock_ollama_client, response=reply)
    model = OllamaChatModel(
        model_id="ollama/llama2:7b",
        ollama_name="llama2:7b",
        capabilities=["completion"],
    )

    model.prompt("Dummy Prompt", stream=stream).text()

    assert mock_ollama_client.chat.call_args.kwargs["model"] == "llama2:7b"


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [True, False])
async def test_async_chat_sends_ollama_name(mocker, stream):
    """The server is asked for the model by its Ollama name, not its llm ID."""
    reply = _ollama_chunk("ok", done=True, usage=True)
    if stream:
        client = _install_async_chat(mocker, chunks=[reply])
    else:
        client = _install_async_chat(mocker, response=reply)
    model = AsyncOllamaChatModel(
        model_id="ollama/llama2:7b",
        ollama_name="llama2:7b",
        capabilities=["completion"],
    )

    await model.prompt("Dummy Prompt", stream=stream).text()

    assert client.chat.call_args.kwargs["model"] == "llama2:7b"


def test_sync_streaming_yields_text(mocker, mock_ollama_client):
    """A streamed sync response yields concatenated text content."""
    _install_sync_chat(
        mock_ollama_client,
        chunks=[
            _ollama_chunk("Test response 1"),
            _ollama_chunk("Test response 2", done=True, usage=True),
        ],
    )

    response = get_model("ollama/llama2:7b").prompt("Dummy Prompt")

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

    response = get_async_model("ollama/llama2:7b").prompt("Dummy Prompt")

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

    response = get_model("ollama/llama2:7b").prompt("Dummy Prompt")
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

    response = get_async_model("ollama/llama2:7b").prompt("Dummy Prompt")
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

    response = get_model("ollama/llama2:7b").prompt("Dummy Prompt", stream=False)
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

    response = get_async_model("ollama/llama2:7b").prompt("Dummy Prompt", stream=False)
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


def test_tool_conversion_without_implementation_raises():
    tool = llm.Tool(name="t")

    with pytest.raises(ValueError, match="'t' has no implementation"):
        _llm_tool_to_ollama_tool(tool)


def test_tool_conversion_kwargs_empty_schema():
    tool = llm.Tool(
        name="t",
        input_schema={"type": "object", "properties": {}},
        implementation=lambda **_kwargs: "x",
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
    _install_sync_chat(
        mock_ollama_client,
        chunks=[
            _ollama_chunk(thinking="Let me think..."),
            _ollama_chunk("Final answer", done=True, usage=True),
        ],
    )

    response = get_model("ollama/deepseek-r1:70b").prompt("Dummy Prompt")
    assert response.text() == "Final answer"

    parts = response.messages()[-1].parts
    reasoning = [p for p in parts if isinstance(p, ReasoningPart)]
    assert len(reasoning) == 1
    assert reasoning[0].text == "Let me think..."


def test_hide_reasoning_suppresses_reasoning_events(mocker, mock_ollama_client):
    """prompt.hide_reasoning drops reasoning events without altering the request."""
    client = _install_sync_chat(
        mock_ollama_client,
        chunks=[
            _ollama_chunk(thinking="Let me think..."),
            _ollama_chunk("Final answer", done=True, usage=True),
        ],
    )

    response = get_model("ollama/deepseek-r1:70b").prompt(
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
    streamed = get_model("ollama/deepseek-r1:70b").prompt("Dummy Prompt")
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
    non_streamed = get_model("ollama/deepseek-r1:70b").prompt(
        "Dummy Prompt", stream=False
    )
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

    response = get_async_model("ollama/llama2:7b").prompt("Dummy Prompt")
    await response.text()

    message = response.response_json["message"]
    assert message["content"] == "Test response 1Test response 2"


def test_tool_call_reply_round_trip(mocker, mock_ollama_client):
    """response.reply() round-trips a tool call + result back into the next request."""
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

    first = get_model("ollama/llama2:7b").prompt("What is 6 times 7?")
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


def test_build_messages_handles_multi_attachment_history(mock_ollama_client, box_png):
    """A conversation with attachments across turns survives the new Part chain."""
    att_a = Attachment(content=box_png, type="image/png")
    att_b = Attachment(content=box_png + b"\x00", type="image/png")

    model = get_model("ollama/llama2:7b")
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


def test_build_messages_drops_reasoning_from_history(mock_ollama_client):
    """A prior turn's reasoning stays out of the next request; its text goes back."""
    mock_ollama_client.chat.side_effect = [
        iter(
            [
                _ollama_chunk(thinking="Let me think..."),
                _ollama_chunk("Final answer", done=True, usage=True),
            ],
        ),
        iter([_ollama_chunk("Follow-up answer", done=True, usage=True)]),
    ]

    conversation = get_model("ollama/deepseek-r1:70b").conversation()
    first = conversation.prompt("Dummy Prompt")
    first.text()
    assert any(isinstance(p, ReasoningPart) for p in first.messages()[-1].parts)
    conversation.prompt("Follow-up prompt").text()

    second_messages = mock_ollama_client.chat.call_args_list[1].kwargs["messages"]
    assert second_messages == [
        {"role": "user", "content": "Dummy Prompt"},
        {"role": "assistant", "content": "Final answer"},
        {"role": "user", "content": "Follow-up prompt"},
    ]


@pytest.mark.parametrize(
    ("model_id", "offers_think"),
    [
        ("ollama/deepseek-r1:70b", True),
        ("ollama/gemma4:e2b", True),
        ("ollama/llama2:7b", False),
    ],
)
def test_think_option_is_offered_only_by_thinking_models(
    mock_ollama_client,
    model_id,
    offers_think,
):
    """The option schema lists ``think`` only for models that report thinking."""
    for model in (get_model(model_id), get_async_model(model_id)):
        properties = model.Options.model_json_schema()["properties"]

        assert ("think" in properties) is offers_think


def test_sync_think_true_is_rejected_without_thinking(mock_ollama_client):
    """Asking a model without thinking to think fails before any request."""
    model = get_model("ollama/llama2:7b")

    with pytest.raises(
        pydantic.ValidationError,
        match="this model does not support thinking",
    ):
        model.prompt("Dummy Prompt", think=True)

    mock_ollama_client.chat.assert_not_called()


@pytest.mark.asyncio
async def test_async_think_true_is_rejected_without_thinking(
    mocker,
    mock_ollama_client,
):
    """Asking a model without thinking to think fails before any request."""
    client = _install_async_chat(mocker, response=_ollama_chunk("ok", done=True))
    model = get_async_model("ollama/llama2:7b")

    with pytest.raises(
        pydantic.ValidationError,
        match="this model does not support thinking",
    ):
        model.prompt("Dummy Prompt", think=True)

    client.chat.assert_not_called()


@pytest.mark.parametrize("think", [False, "false"])
def test_sync_think_false_is_not_sent_without_thinking(mock_ollama_client, think):
    """Turning thinking off on a model without it is accepted and not sent."""
    _install_sync_chat(
        mock_ollama_client,
        response=_ollama_chunk("ok", done=True, usage=True),
    )

    get_model("ollama/llama2:7b").prompt(
        "Dummy Prompt", stream=False, think=think
    ).text()

    assert "think" not in mock_ollama_client.chat.call_args.kwargs


@pytest.mark.asyncio
@pytest.mark.parametrize("think", [False, "false"])
async def test_async_think_false_is_not_sent_without_thinking(
    mocker,
    mock_ollama_client,
    think,
):
    """Turning thinking off on a model without it is accepted and not sent."""
    client = _install_async_chat(
        mocker,
        response=_ollama_chunk("ok", done=True, usage=True),
    )

    await (
        get_async_model("ollama/llama2:7b")
        .prompt(
            "Dummy Prompt",
            stream=False,
            think=think,
        )
        .text()
    )

    assert "think" not in client.chat.call_args.kwargs


def test_sync_think_false_is_sent_with_thinking(mock_ollama_client):
    """Turning thinking off on a thinking model reaches the server."""
    _install_sync_chat(
        mock_ollama_client,
        response=_ollama_chunk("ok", done=True, usage=True),
    )

    get_model("ollama/deepseek-r1:70b").prompt(
        "Dummy Prompt",
        stream=False,
        think=False,
    ).text()

    assert mock_ollama_client.chat.call_args.kwargs["think"] is False


@pytest.mark.asyncio
async def test_async_think_false_is_sent_with_thinking(mocker, mock_ollama_client):
    """Turning thinking off on a thinking model reaches the server."""
    client = _install_async_chat(
        mocker,
        response=_ollama_chunk("ok", done=True, usage=True),
    )

    await (
        get_async_model("ollama/deepseek-r1:70b")
        .prompt(
            "Dummy Prompt",
            stream=False,
            think=False,
        )
        .text()
    )

    assert client.chat.call_args.kwargs["think"] is False
