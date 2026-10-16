"""Chat models served by Ollama."""

import json
from collections.abc import AsyncGenerator, Iterator
from typing import ClassVar, cast

import llm
import ollama
from llm.parts import (
    AttachmentPart,
    Message,
    ReasoningPart,
    StreamEvent,
    TextPart,
    ToolCallPart,
    ToolResultPart,
)
from llm.utils import monotonic_ulid
from ollama._utils import convert_function_to_tool
from pydantic import Field

from llm_ollama.auth import get_async_client, get_client
from llm_ollama.models.base import BaseMixin


class ChatMixin(BaseMixin):
    """Request building shared by the sync and async Ollama chat models."""

    can_stream: bool = True
    supports_schema: bool = True
    supports_tools: bool = True
    attachment_types: ClassVar[set[str]] = {
        "image/png",
        "image/jpeg",
        "image/webp",
        "image/gif",
    }

    class Options(llm.Options):
        """Parameters that can be set when the model is run by Ollama.

        See: https://github.com/ollama/ollama/blob/main/docs/modelfile.md#parameter
        """

        num_ctx: int | None = Field(
            default=None,
            description="Sets the size of the context window used to generate the next token. (Default: 2048)",
        )
        repeat_last_n: int | None = Field(
            default=None,
            description="Sets how far back for the model to look back to prevent repetition. (Default: 64, 0 = disabled, -1 = num_ctx)",
        )
        repeat_penalty: float | None = Field(
            default=None,
            description="Sets how strongly to penalize repetitions. A higher value (e.g., 1.5) will penalize repetitions more strongly, while a lower value (e.g., 0.9) will be more lenient. (Default: 1.1)",
        )
        temperature: float | None = Field(
            default=None,
            description="The temperature of the model. Increasing the temperature will make the model answer more creatively. (Default: 0.8)",
        )
        seed: int | None = Field(
            default=None,
            description="Sets the random number seed to use for generation. Setting this to a specific number will make the model generate the same text for the same prompt. (Default: 0)",
        )
        stop: list[str] | None = Field(
            default=None,
            description="Sets the stop sequences to use. When this pattern is encountered the LLM will stop generating text and return.",
        )
        num_predict: int | None = Field(
            default=None,
            description="Maximum number of tokens to predict when generating text. (Default: -1, infinite generation)",
        )
        top_k: int | None = Field(
            default=None,
            description="Reduces the probability of generating nonsense. A higher value (e.g. 100) will give more diverse answers, while a lower value (e.g. 10) will be more conservative. (Default: 40)",
        )
        top_p: float | None = Field(
            default=None,
            description="Works together with top-k. A higher value (e.g., 0.95) will lead to more diverse text, while a lower value (e.g., 0.5) will generate more focused and conservative text. (Default: 0.9)",
        )
        min_p: float | None = Field(
            default=None,
            description="Alternative to the top_p, and aims to ensure a balance of quality and variety. The parameter p represents the minimum probability for a token to be considered, relative to the probability of the most likely token. (Default: 0.0)",
        )
        json_object: bool | None = Field(
            default=None,
            description="Output a valid JSON object {...}. Prompt must mention JSON.",
        )
        think: bool | None = Field(
            default=None,
            description="Enable the model's thinking process.",
        )

    def __init__(
        self,
        model_id: str,
        *,
        supports_tools: bool = True,
    ) -> None:
        """Create the model; ``supports_tools`` says whether it accepts tool definitions."""
        self.model_id = model_id
        self.supports_tools = supports_tools

    def build_messages(
        self,
        prompt: llm.Prompt,
        conversation: llm.Conversation | llm.AsyncConversation | None,
    ) -> list[dict]:
        """Convert ``prompt.messages`` into the ``messages`` list of an Ollama chat call."""
        # `conversation` is unused: under the llm 0.32 contract the framework
        # pre-bakes prior turns into prompt.messages, so walking conversation
        # here would double-emit history. The argument stays on the signature
        # for API compatibility.
        del conversation

        messages: list[dict] = []
        for message in prompt.messages:
            messages.extend(_convert_message(message))
        return messages

    def _prepare_chat_kwargs(self, prompt: llm.Prompt) -> tuple[dict, dict]:
        """Build the ``options`` and ``kwargs`` dicts for an Ollama chat call.

        Splits prompt options into Ollama "options" (model parameters) and top-level
        chat kwargs (``think``, ``format``, ``tools``), so both the sync and async
        ``execute`` methods can prepare the request identically.
        """
        options = prompt.options.model_dump(exclude_none=True)
        think = options.pop("think", None)
        json_object = options.pop("json_object", None)
        kwargs: dict = {}
        if think is not None:
            kwargs["think"] = think
        if json_object:
            kwargs["format"] = "json"
        elif prompt.schema:
            kwargs["format"] = prompt.schema
        if prompt.tools:
            # supported_server_side_tools is empty, so llm rejects ServerSideTool earlier.
            kwargs["tools"] = [
                _llm_tool_to_ollama_tool(cast("llm.Tool", tool))
                for tool in prompt.tools
            ]
        return options, kwargs


class OllamaChatModel(ChatMixin, llm.KeyModel):
    """A chat model served by Ollama."""

    def execute(
        self,
        prompt: llm.Prompt,
        stream: bool,  # noqa: FBT001
        response: llm.Response,
        conversation: llm.Conversation | None = None,
        key: str | None = None,
    ) -> Iterator[StreamEvent]:
        """Run the prompt on the Ollama server, yielding events as they arrive."""
        messages = self.build_messages(prompt, conversation)
        options, kwargs = self._prepare_chat_kwargs(prompt)
        accumulator = _ChunkAccumulator(response, hide_reasoning=prompt.hide_reasoning)
        client = get_client(key=key)
        if stream:
            response_stream = client.chat(
                model=self.model_id,
                messages=messages,
                stream=True,
                options=options,
                **kwargs,
            )
            for chunk in response_stream:
                yield from accumulator.consume(chunk)
        else:
            ollama_response = client.chat(
                model=self.model_id,
                messages=messages,
                options=options,
                **kwargs,
            )
            yield from accumulator.consume(ollama_response)
        accumulator.finalize()


class AsyncOllamaChatModel(ChatMixin, llm.AsyncKeyModel):
    """An asynchronous chat model served by Ollama."""

    async def execute(
        self,
        prompt: llm.Prompt,
        stream: bool,  # noqa: FBT001
        response: llm.AsyncResponse,
        conversation: llm.AsyncConversation | None = None,
        key: str | None = None,
    ) -> AsyncGenerator[StreamEvent, None]:
        """Execute the Ollama model asynchronously.

        Parameters
        ----------
        prompt : llm.Prompt
            The prompt for the model.
        stream : bool
            Whether to stream the response.
        response : llm.AsyncResponse
            The response object to populate.
        conversation : llm.AsyncConversation | None, optional
            The conversation context.
        key : str | None, optional
            API key for the Ollama server.

        """
        messages = self.build_messages(prompt, conversation)
        options, kwargs = self._prepare_chat_kwargs(prompt)
        accumulator = _ChunkAccumulator(response, hide_reasoning=prompt.hide_reasoning)
        client = get_async_client(key=key)
        if stream:
            response_stream = await client.chat(
                model=self.model_id,
                messages=messages,
                stream=True,
                options=options,
                **kwargs,
            )
            async for chunk in response_stream:
                for event in accumulator.consume(chunk):
                    yield event
        else:
            ollama_response = await client.chat(
                model=self.model_id,
                messages=messages,
                options=options,
                **kwargs,
            )
            for event in accumulator.consume(ollama_response):
                yield event
        accumulator.finalize()


class _ChunkAccumulator:
    """Turns Ollama chat chunks into StreamEvents and rebuilds the raw payload.

    One accumulator serves one ``execute()`` call, whose response it holds and mutates.
    Reassembling the streamed pieces gives ``response_json`` the same shape whether or
    not the caller streamed.
    """

    def __init__(
        self,
        response: "llm.Response | llm.AsyncResponse",
        *,
        hide_reasoning: bool = False,
    ) -> None:
        self.response = response
        self.hide_reasoning = hide_reasoning
        self._content: list[str] = []
        self._thinking: list[str] = []
        self._tool_calls: list[dict] = []
        self._final_chunk: ollama.ChatResponse | None = None

    def consume(self, chunk: ollama.ChatResponse) -> list[StreamEvent]:
        """Accumulate one chunk and return the StreamEvents it produces.

        Registers any tool calls the chunk carries on the response.
        """
        events: list[StreamEvent] = []
        if chunk.message.content:
            self._content.append(chunk.message.content)
            events.append(StreamEvent(type="text", chunk=chunk.message.content))
        if chunk.message.thinking:
            self._thinking.append(chunk.message.thinking)
            if not self.hide_reasoning:
                events.append(
                    StreamEvent(type="reasoning", chunk=chunk.message.thinking),
                )
        for tool_call in chunk.message.tool_calls or ():
            self._tool_calls.append(tool_call.model_dump())
            tool_call_id = f"tc_{str(monotonic_ulid()).lower()}"
            arguments = dict(tool_call.function.arguments or {})
            events.append(
                StreamEvent(
                    type="tool_call_name",
                    chunk=tool_call.function.name,
                    tool_call_id=tool_call_id,
                ),
            )
            events.append(
                StreamEvent(
                    type="tool_call_args",
                    chunk=json.dumps(arguments),
                    tool_call_id=tool_call_id,
                ),
            )
            self.response.add_tool_call(
                llm.ToolCall(
                    name=tool_call.function.name,
                    arguments=arguments,
                    tool_call_id=tool_call_id,
                ),
            )
        if chunk.done:
            self._final_chunk = chunk
        return events

    def finalize(self) -> None:
        """Write the reassembled payload and token usage onto the response.

        A stream cut short before its ``done`` chunk leaves both unset; the text already
        emitted is unaffected.
        """
        if self._final_chunk is None:
            return
        payload = self._final_chunk.model_dump()
        payload["message"]["content"] = "".join(self._content)
        payload["message"]["thinking"] = "".join(self._thinking) or None
        payload["message"]["tool_calls"] = self._tool_calls or None
        self.response.response_json = payload
        self.response.set_usage(
            input=self._final_chunk.prompt_eval_count,
            output=self._final_chunk.eval_count,
        )


def _convert_message(message: Message) -> list[dict]:
    """Convert one ``llm`` message into the Ollama chat messages it stands for.

    Returns the message's tool results as standalone ``"tool"`` messages, followed by
    the message itself unless it has no text, images or tool calls.
    """
    text_chunks: list[str] = []
    images: list[str] = []
    tool_calls: list[ollama.Message.ToolCall] = []
    tool_results: list[ToolResultPart] = []
    for part in message.parts:
        if isinstance(part, TextPart):
            text_chunks.append(part.text)
        elif isinstance(part, AttachmentPart) and part.attachment is not None:
            images.append(part.attachment.base64_content())
        elif isinstance(part, ToolCallPart):
            tool_calls.append(
                ollama.Message.ToolCall(
                    function=ollama.Message.ToolCall.Function(
                        name=part.name,
                        arguments=part.arguments or {},
                    ),
                ),
            )
        elif isinstance(part, ToolResultPart):
            tool_results.append(part)
        elif isinstance(part, ReasoningPart):
            # Ollama does not accept reasoning input back; thinking
            # models keep their own state. Drop silently.
            continue

    # ToolResultParts always become standalone {"role": "tool", ...}
    # messages on the wire — Ollama keys them by tool name, not by
    # being grouped into a parent message.
    wire_messages: list[dict] = [
        {
            "role": "tool",
            "content": tool_result.output,
            "name": tool_result.name,
        }
        for tool_result in tool_results
    ]

    if not text_chunks and not images and not tool_calls:
        return wire_messages

    wire: dict = {"role": message.role, "content": "".join(text_chunks)}
    if images:
        wire["images"] = images
    if tool_calls:
        wire["tool_calls"] = tool_calls
    wire_messages.append(wire)
    return wire_messages


def _llm_tool_to_ollama_tool(tool: llm.Tool) -> ollama.Tool:
    """Convert an llm.Tool to an ollama.Tool.

    tool.input_schema, when set, overrides the parameters that would otherwise be
    derived from tool.implementation's signature — this is how tools whose
    implementation is **kwargs-bound carry their real parameter schema.

    Parameters
    ----------
    tool : llm.Tool
        An llm.Tool instance with a callable implementation.

    Raises
    ------
    ValueError
        If ``tool`` has no implementation.

    """
    if tool.implementation is None:
        raise ValueError(f"Tool {tool.name!r} has no implementation")
    ollama_tool = convert_function_to_tool(tool.implementation)
    assert ollama_tool.function is not None  # noqa: S101
    ollama_tool.function.name = tool.name
    if tool.description:
        ollama_tool.function.description = tool.description
    if tool.input_schema:
        ollama_tool.function.parameters = ollama.Tool.Function.Parameters(
            type=tool.input_schema.get("type"),
            required=tool.input_schema.get("required"),
            properties={
                k: ollama.Tool.Function.Parameters.Property(
                    **{
                        f: v
                        for f, v in p.items()
                        if f in ("type", "description", "enum", "items")
                    },
                )
                for k, p in tool.input_schema.get("properties", {}).items()
            },
        )
    return ollama_tool
