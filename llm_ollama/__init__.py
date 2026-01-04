import contextlib
import os
import warnings
from collections import defaultdict
from collections.abc import AsyncGenerator

import llm
import ollama
from llm.utils import dicts_to_table_string
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
                    _get_ollama_model_capabilities(model["digest"], model["model"]),
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
def register_models(register):
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
        logprobs: bool | None = Field(
            default=None,
            description="Include log probabilities for each token",
        )
        top_logprobs: int | None = Field(
            default=None,
            description="Number of top log probabilities to return per token",
            ge=0,
            le=20,
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
        messages = []
        if not conversation:
            if prompt.system:
                messages.append({"role": "system", "content": prompt.system})
            messages.append({"role": "user", "content": prompt.prompt})
            if prompt.attachments:
                messages[-1]["images"] = [
                    attachment.base64_content() for attachment in prompt.attachments
                ]
            return messages

        current_system = None
        for prev_response in conversation.responses:
            if (
                prev_response.prompt.system
                and prev_response.prompt.system != current_system
            ):
                messages.append(
                    {"role": "system", "content": prev_response.prompt.system},
                )
                current_system = prev_response.prompt.system
            messages.append({"role": "user", "content": prev_response.prompt.prompt})
            if prev_response.prompt.attachments:
                messages[-1]["images"] = [
                    attachment.base64_content()
                    for attachment in prev_response.prompt.attachments
                ]

            messages.append(
                {"role": "assistant", "content": prev_response.text_or_raise()}
            )
        if prompt.system and prompt.system != current_system:
            messages.append({"role": "system", "content": prompt.system})
        messages.append({"role": "user", "content": prompt.prompt})
        if prompt.attachments:
            messages[-1]["images"] = [
                attachment.base64_content() for attachment in prompt.attachments
            ]
        messages.extend(
            {
                "role": "tool",
                "content": tool_result.output,
                "name": tool_result.name,
            }
            for tool_result in prompt.tool_results
        )

        return messages

    def set_usage(self, response, usage):
        if not usage:
            return
        input_tokens = usage.pop("prompt_tokens")
        output_tokens = usage.pop("completion_tokens")
        response.set_usage(input=input_tokens, output=output_tokens)


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
        options = prompt.options.model_dump(exclude_none=True)
        think = options.pop("think", None)
        json_object = options.pop("json_object", None)
        logprobs = options.pop("logprobs", None)
        top_logprobs = options.pop("top_logprobs", None)
        kwargs = {}
        usage = None
        if think is not None:
            kwargs["think"] = think
        if json_object:
            kwargs["format"] = "json"
        elif prompt.schema:
            kwargs["format"] = prompt.schema
        if logprobs is not None:
            kwargs["logprobs"] = logprobs
        if top_logprobs is not None:
            kwargs["top_logprobs"] = top_logprobs
        if prompt.tools:
            kwargs["tools"] = [_llm_tool_to_ollama_tool(tool) for tool in prompt.tools]
        if stream:
            response_stream = get_client().chat(
                model=self.model_id,
                messages=messages,
                stream=True,
                options=options,
                **kwargs,
            )
            collected_logprobs = []
            for chunk in response_stream:
                if chunk.message.tool_calls:
                    for tool_call in chunk.message.tool_calls:
                        response.add_tool_call(
                            llm.ToolCall(
                                name=tool_call.function.name,
                                arguments=tool_call.function.arguments,
                            ),
                        )
                with contextlib.suppress(KeyError):
                    if chunk["done"]:
                        usage = {
                            "prompt_tokens": chunk["prompt_eval_count"],
                            "completion_tokens": chunk["eval_count"],
                        }
                    yield chunk["message"]["content"]
                if hasattr(chunk, "logprobs") and chunk.logprobs:
                    collected_logprobs.extend(chunk.logprobs)

            if response.response_json is None:
                response.response_json = {}
            if collected_logprobs:
                response.response_json["logprobs"] = _logprobs_to_dicts(
                    collected_logprobs
                )
        else:
            ollama_response = get_client().chat(
                model=self.model_id,
                messages=messages,
                options=options,
                **kwargs,
            )
            response.response_json = ollama_response.dict()
            usage = {
                "prompt_tokens": response.response_json["prompt_eval_count"],
                "completion_tokens": response.response_json["eval_count"],
            }
            yield response.response_json["message"]["content"]
            if ollama_response.message.tool_calls:
                for tool_call in ollama_response.message.tool_calls:
                    response.add_tool_call(
                        llm.ToolCall(
                            name=tool_call.function.name,
                            arguments=tool_call.function.arguments,
                        ),
                    )
        self.set_usage(response, usage)


class AsyncOllama(_SharedOllama, llm.AsyncModel):
    async def execute(
        self,
        prompt: llm.Prompt,
        stream: bool,
        response: llm.AsyncResponse,
        conversation: llm.AsyncConversation | None = None,
    ) -> AsyncGenerator[str, None]:
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
        options = prompt.options.model_dump(exclude_none=True)
        think = options.pop("think", None)
        json_object = options.pop("json_object", None)
        logprobs = options.pop("logprobs", None)
        top_logprobs = options.pop("top_logprobs", None)
        kwargs = {}
        usage = None
        if think is not None:
            kwargs["think"] = think
        if json_object:
            kwargs["format"] = "json"
        elif prompt.schema:
            kwargs["format"] = prompt.schema
        if logprobs is not None:
            kwargs["logprobs"] = logprobs
        if top_logprobs is not None:
            kwargs["top_logprobs"] = top_logprobs
        if prompt.tools:
            kwargs["tools"] = [_llm_tool_to_ollama_tool(tool) for tool in prompt.tools]

        try:
            if stream:
                response_stream = await get_async_client().chat(
                    model=self.model_id,
                    messages=messages,
                    stream=True,
                    options=options,
                    **kwargs,
                )
                collected_logprobs = []
                async for chunk in response_stream:
                    with contextlib.suppress(KeyError):
                        yield chunk["message"]["content"]
                        if chunk["done"]:
                            usage = {
                                "prompt_tokens": chunk["prompt_eval_count"],
                                "completion_tokens": chunk["eval_count"],
                            }

                    if hasattr(chunk, "logprobs") and chunk.logprobs:
                        collected_logprobs.extend(chunk.logprobs)

                if response.response_json is None:
                    response.response_json = {}

                # Store collected logprobs in response
                if collected_logprobs:
                    response.response_json["logprobs"] = _logprobs_to_dicts(
                        collected_logprobs
                    )

            else:
                ollama_response = await get_async_client().chat(
                    model=self.model_id,
                    messages=messages,
                    options=options,
                    **kwargs,
                )
                response.response_json = ollama_response.dict()
                usage = {
                    "prompt_tokens": response.response_json["prompt_eval_count"],
                    "completion_tokens": response.response_json["eval_count"],
                }
                yield response.response_json["message"]["content"]

                if hasattr(ollama_response, "logprobs") and ollama_response.logprobs:
                    response.response_json["logprobs"] = _logprobs_to_dicts(
                        ollama_response.logprobs
                    )

            self.set_usage(response, usage)

        except Exception as e:
            raise RuntimeError(f"Async execution failed: {e}") from e


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


def _logprobs_to_dicts(
    logprobs: list[ollama._types.Logprob] | list[dict],
) -> list[dict]:
    """Convert Logprob objects to dictionaries for JSON serialization.

    Parameters
    ----------
    logprobs : list[ollama._types.Logprob] or list[dict]
        A list of Logprob objects (or dictionaries). The function handles
        Logprob objects that have a `model_dump` method, a `dict` method,
        or are already dictionaries.

    Returns
    -------
    list[dict]
        A list of dictionaries representing the logprobs.
    """
    if not logprobs:
        return []
    if hasattr(logprobs[0], "model_dump"):
        return [lp.model_dump() for lp in logprobs]
    elif hasattr(logprobs[0], "dict"):
        return [lp.dict() for lp in logprobs]
    else:
        # Assume it's already a dictionary
        return logprobs


def _llm_tool_to_ollama_tool(tool: llm.Tool) -> ollama.Tool:
    """Convert an llm.Tool to an ollama.Tool.

    Uses ollama's convert_function_to_tool to properly convert the function
    signature to a schema that matches ollama's expectations.

    Parameters
    ----------
    tool : llm.Tool
        An llm.Tool instance with a callable implementation.

    Returns
    -------
    ollama.Tool
        An ollama.Tool instance.

    """
    ollama_tool = convert_function_to_tool(tool.implementation)
    if tool.name != tool.implementation.__name__:
        ollama_tool.function.name = tool.name
    if tool.description:
        ollama_tool.function.description = tool.description
    return ollama_tool
