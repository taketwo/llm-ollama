import json
import os
import warnings
from collections import defaultdict
from collections.abc import AsyncGenerator

import llm
import ollama
from llm.parts import (
    AttachmentPart,
    ReasoningPart,
    StreamEvent,
    TextPart,
    ToolCallPart,
    ToolResultPart,
)
from llm.utils import dicts_to_table_string, monotonic_ulid
from ollama._utils import convert_function_to_tool
from pydantic import Field, TypeAdapter, ValidationError

from llm_ollama.auth import get_async_client, get_client
from llm_ollama.cache import Cache

cache = Cache(llm.user_dir() / "llm-ollama" / "cache")


@llm.hookimpl
def register_commands(cli):
    @cli.group(name="ollama")
    def ollama_group() -> None:
        """Commands for working with models hosted on Ollama server."""

    @ollama_group.command()
    def models() -> None:
        """List models that are available on Ollama server."""
        to_print = [
            {
                "model": model["model"],
                "digest": model["digest"][:12],
                "capabilities": ", ".join(
                    sorted(
                        _get_ollama_model_capabilities(model["digest"], model["model"])
                    ),
                ),
            }
            for model in _get_ollama_models()
        ]
        to_print.sort(key=lambda x: x["model"])
        done = dicts_to_table_string(["model", "digest", "capabilities"], to_print)
        print("\n".join(done))


@llm.hookimpl
def register_tools(register):
    register(
        llm.Tool(
            name="ollama_web_search",
            description="Search the web for information",
            implementation=ollama.web_search,
        ),
    )
    register(
        llm.Tool(
            name="ollama_web_fetch",
            description="Fetch the contents of a web page",
            implementation=ollama.web_fetch,
        ),
    )


@llm.hookimpl
def register_models(register, model_aliases=None):
    models = defaultdict(list)
    for model in _get_ollama_models():
        name, digest = model["model"], model["digest"]
        models[digest].append(name)
        if name.endswith(":latest"):
            models[digest].append(name[: -len(":latest")])
    for digest, names in models.items():
        name, aliases = _pick_primary_name(names)
        capabilities = _get_ollama_model_capabilities(digest, name)
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
    for model in _get_ollama_models():
        models[model["digest"]].append(model["model"])
        if model["model"].endswith(":latest"):
            models[model["digest"]].append(model["model"][: -len(":latest")])
    for names in models.values():
        name, aliases = _pick_primary_name(names)
        register(OllamaEmbed(name), aliases=aliases)


class _SharedOllama:
    can_stream: bool = True
    supports_schema: bool = True
    supports_tools: bool = True
    attachment_types = {
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
        supports_tools: bool = True,
    ) -> None:
        self.model_id = model_id
        self.supports_tools = supports_tools

    def __str__(self) -> str:
        return f"Ollama: {self.model_id}"

    def build_messages(self, prompt, conversation):
        # `conversation` is unused: under the llm 0.32 contract the framework
        # pre-bakes prior turns into prompt.messages, so walking conversation
        # here would double-emit history. The argument stays on the signature
        # for API compatibility.
        del conversation

        messages: list[dict] = []
        for message in prompt.messages:
            text_chunks: list[str] = []
            images: list[str] = []
            tool_calls: list[ollama.Message.ToolCall] = []
            tool_results: list[ToolResultPart] = []
            for part in message.parts:
                if isinstance(part, TextPart):
                    text_chunks.append(part.text)
                elif isinstance(part, AttachmentPart):
                    if part.attachment is not None:
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
            for tool_result in tool_results:
                messages.append(
                    {
                        "role": "tool",
                        "content": tool_result.output,
                        "name": tool_result.name,
                    },
                )

            if not text_chunks and not images and not tool_calls:
                continue

            wire: dict = {"role": message.role, "content": "".join(text_chunks)}
            if images:
                wire["images"] = images
            if tool_calls:
                wire["tool_calls"] = tool_calls
            messages.append(wire)

        return messages

    def set_usage(self, response, usage):
        if not usage:
            return
        input_tokens = usage.pop("prompt_tokens")
        output_tokens = usage.pop("completion_tokens")
        response.set_usage(input=input_tokens, output=output_tokens)

    def _prepare_chat_kwargs(self, prompt) -> tuple[dict, dict]:
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
            kwargs["tools"] = [_llm_tool_to_ollama_tool(tool) for tool in prompt.tools]
        return options, kwargs

    def _interpret_chunk(
        self,
        chunk: ollama.ChatResponse,
        response: "llm.Response | llm.AsyncResponse",
        *,
        hide_reasoning: bool = False,
    ) -> tuple[list[StreamEvent], dict | None]:
        """Translate one Ollama chat chunk (streaming or non-streaming) into a list
        of StreamEvents to yield and an optional usage dict to register at end of
        stream.

        Side effect: registers each captured tool call on ``response`` via
        ``add_tool_call`` with the same synthesized ``tool_call_id`` carried on
        the emitted events. The framework dedups by id when assembling parts,
        so the call survives in ``response.tool_calls()`` while also taking part
        in the StreamEvent ordering used by ``response.stream_events()`` and
        ``response.to_dict()``.

        ``hide_reasoning`` suppresses reasoning events without altering the
        request: the model still thinks, only the visible trace is dropped. The
        ``-o think`` option is the user-facing knob for actually disabling the
        reasoning step.
        """
        events: list[StreamEvent] = []
        if chunk.message.content:
            events.append(StreamEvent(type="text", chunk=chunk.message.content))
        if chunk.message.thinking and not hide_reasoning:
            events.append(StreamEvent(type="reasoning", chunk=chunk.message.thinking))
        for tool_call in chunk.message.tool_calls or ():
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
            response.add_tool_call(
                llm.ToolCall(
                    name=tool_call.function.name,
                    arguments=arguments,
                    tool_call_id=tool_call_id,
                ),
            )
        usage = None
        if chunk.done:
            usage = {
                "prompt_tokens": chunk.prompt_eval_count,
                "completion_tokens": chunk.eval_count,
            }
        return events, usage


class Ollama(_SharedOllama, llm.Model):
    def execute(
        self,
        prompt: llm.Prompt,
        stream: bool,
        response: llm.Response,
        conversation=None,
    ):
        messages = self.build_messages(prompt, conversation)
        response._prompt_json = {"messages": messages}
        options, kwargs = self._prepare_chat_kwargs(prompt)
        usage = None
        if stream:
            response_stream = get_client().chat(
                model=self.model_id,
                messages=messages,
                stream=True,
                options=options,
                **kwargs,
            )
            for chunk in response_stream:
                events, chunk_usage = self._interpret_chunk(
                    chunk,
                    response,
                    hide_reasoning=prompt.hide_reasoning,
                )
                if chunk_usage is not None:
                    usage = chunk_usage
                yield from events
        else:
            ollama_response = get_client().chat(
                model=self.model_id,
                messages=messages,
                options=options,
                **kwargs,
            )
            response.response_json = ollama_response.model_dump()
            events, usage = self._interpret_chunk(
                ollama_response,
                response,
                hide_reasoning=prompt.hide_reasoning,
            )
            yield from events
        self.set_usage(response, usage)


class AsyncOllama(_SharedOllama, llm.AsyncModel):
    async def execute(
        self,
        prompt: llm.Prompt,
        stream: bool,
        response: llm.AsyncResponse,
        conversation: llm.AsyncConversation | None = None,
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

        """
        messages = self.build_messages(prompt, conversation)
        response._prompt_json = {"messages": messages}
        options, kwargs = self._prepare_chat_kwargs(prompt)
        usage = None
        if stream:
            response_stream = await get_async_client().chat(
                model=self.model_id,
                messages=messages,
                stream=True,
                options=options,
                **kwargs,
            )
            async for chunk in response_stream:
                events, chunk_usage = self._interpret_chunk(
                    chunk,
                    response,
                    hide_reasoning=prompt.hide_reasoning,
                )
                if chunk_usage is not None:
                    usage = chunk_usage
                for event in events:
                    yield event
        else:
            ollama_response = await get_async_client().chat(
                model=self.model_id,
                messages=messages,
                options=options,
                **kwargs,
            )
            response.response_json = ollama_response.model_dump()
            events, usage = self._interpret_chunk(
                ollama_response,
                response,
                hide_reasoning=prompt.hide_reasoning,
            )
            for event in events:
                yield event
        self.set_usage(response, usage)


class OllamaEmbed(llm.EmbeddingModel):
    supports_text = True
    supports_binary = False
    batch_size = 8

    def __init__(self, model_id):
        self.model_id = model_id
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
                )

    def __str__(self) -> str:
        return f"Ollama: {self.model_id}"

    def embed_batch(self, items):
        result = get_client().embed(
            model=self.model_id,
            input=items,
            truncate=self.truncate,
        )
        yield from result["embeddings"]


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


def _get_ollama_models() -> list[dict]:
    """Get a list of models available on Ollama.

    Returns
    -------
    list[dict]
        A list of models available on Ollama. If the Ollama server is down, an empty
        list is returned.

    """
    try:
        return get_client().list()["models"]
    except:
        return []


@cache("model_capabilities", key="digest")
def _get_ollama_model_capabilities(digest: str, model: str) -> list[str]:
    """Get a list of capabilities for a given Ollama model.

    This function may raise an exception if the Ollama server is down or the model does
    not exist.

    Returns
    -------
    list[str]
        A list of capabilities for the given model.

    """
    return get_client().show(model).capabilities or []


def _llm_tool_to_ollama_tool(tool: llm.Tool) -> ollama.Tool:
    """Convert an llm.Tool to an ollama.Tool.

    Uses ollama's convert_function_to_tool for the initial conversion, then
    overrides the parameters with tool.input_schema when it contains properties.
    This handles tools whose implementation is **kwargs-bound and carries the
    real parameter schema in input_schema rather than the function signature.

    Parameters
    ----------
    tool : llm.Tool
        An llm.Tool instance with a callable implementation.

    Returns
    -------
    ollama.Tool
        An ollama.Tool instance.

    """
    assert tool.implementation is not None
    ollama_tool = convert_function_to_tool(tool.implementation)
    assert ollama_tool.function is not None
    ollama_tool.function.name = tool.name
    if tool.description:
        ollama_tool.function.description = tool.description
    if tool.input_schema.get("properties"):
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
                for k, p in tool.input_schema["properties"].items()
            },
        )
    return ollama_tool
