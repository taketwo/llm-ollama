from collections import defaultdict
from typing import List, Optional, Tuple

import click
import llm
import ollama
from pydantic import Field


@llm.hookimpl
def register_commands(cli):
    @cli.group(name="ollama")
    def ollama_group():
        "Commands for working with models hosted on Ollama"

    @ollama_group.command(name="list-models")
    def list_models():
        """List models that are available locally on Ollama server."""
        for model in _get_ollama_models():
            click.echo(model["name"])


@llm.hookimpl
def register_models(register):
    models = defaultdict(list)
    for model in _get_ollama_models():
        models[model["digest"]].append(model["name"])
        if model["name"].endswith(":latest"):
            models[model["digest"]].append(model["name"][: -len(":latest")])
    for names in models.values():
        name, aliases = _pick_primary_name(names)
        register(Ollama(name), aliases=aliases)


class Ollama(llm.Model):
    can_stream: bool = True

    class Options(llm.Options):
        """
        Parameters that can be set when the model is run by Ollama.

        See: https://github.com/ollama/ollama/blob/main/docs/modelfile.md#parameter
        """

        mirostat: Optional[int] = Field(
            default=None,
            description=("Enable Mirostat sampling for controlling perplexity."),
        )
        mirostat_eta: Optional[float] = Field(
            default=None,
            description=(
                "Influences how quickly the algorithm responds to feedback from the generated text."
            ),
        )
        mirostat_tau: Optional[float] = Field(
            default=None,
            description=(
                "Controls the balance between coherence and diversity of the output."
            ),
        )
        num_ctx: Optional[int] = Field(
            default=None,
            description="The size of the context window used to generate the next token.",
        )
        temperature: Optional[float] = Field(
            default=None,
            description=(
                "The temperature of the model. Increasing the temperature will make the model answer more creatively."
            ),
        )
        seed: Optional[int] = Field(
            default=None,
            description=(
                "Sets the random number seed to use for generation. Setting this to a specific number will make the model generate the same text for the same prompt."
            ),
        )
        stop: Optional[str] = Field(
            default=None,
            description=(
                "Sets the stop sequences to use. When this pattern is encountered the LLM will stop generating text and return."
            ),
        )
        tfs_z: Optional[float] = Field(
            default=None,
            description=(
                "Tail free sampling is used to reduce the impact of less probable tokens from the output."
            ),
        )
        num_predict: Optional[int] = Field(
            default=None,
            description=("Maximum number of tokens to predict when generating text."),
        )
        top_k: Optional[int] = Field(
            default=None,
            description=("Reduces the probability of generating nonsense."),
        )
        top_p: Optional[float] = Field(
            default=None,
            description=(
                "Works together with top-k. A higher value (e.g., 0.95) will lead to more diverse text, while a lower value (e.g., 0.5) will generate more focused and conservative text."
            ),
        )

    def __init__(
        self,
        model_id: str,
    ) -> None:
        self.model_id = model_id

    def __str__(self) -> str:
        return f"Ollama: {self.model_id}"

    def execute(
        self,
        prompt: llm.Prompt,
        stream: bool,
        response: llm.Response,
        conversation=None,
    ):
        messages = self.build_messages(prompt, conversation)
        options = prompt.options.model_dump(exclude_none=True)
        response._prompt_json = {"messages": messages}

        if stream:
            response_stream = ollama.chat(
                model=self.model_id,
                messages=messages,
                stream=True,
                options=options,
            )
            for chunk in response_stream:
                try:
                    yield chunk["message"]["content"]
                except KeyError:
                    pass
        else:
            response.response_json = ollama.chat(
                model=self.model_id,
                messages=messages,
                options=options,
            )
            yield response.response_json["message"]["content"]

    def build_messages(self, prompt, conversation):
        messages = []
        if not conversation:
            if prompt.system:
                messages.append({"role": "system", "content": prompt.system})
            messages.append({"role": "user", "content": prompt.prompt})
            return messages
        current_system = None
        for prev_response in conversation.responses:
            if (
                prev_response.prompt.system
                and prev_response.prompt.system != current_system
            ):
                messages.append(
                    {"role": "system", "content": prev_response.prompt.system}
                )
                current_system = prev_response.prompt.system
            messages.append({"role": "user", "content": prev_response.prompt.prompt})
            messages.append({"role": "assistant", "content": prev_response.text()})
        if prompt.system and prompt.system != current_system:
            messages.append({"role": "system", "content": prompt.system})
        messages.append({"role": "user", "content": prompt.prompt})
        return messages


def _pick_primary_name(names: List[str]) -> Tuple[str, List[str]]:
    """Pick the primary model name from a list of names.

    The picking algorithm prefers names with the most specific tag, e.g. "llama2:7b"
    over "llama2:latest" over "llama2".

    Parameters
    ----------
    names : list[str]
        A non-empty list of model names.

    Returns
    -------
    tuple[str, list[str, ...]]
        The primary model name and a list with the secondary names.

    """
    if len(names) == 1:
        return names[0], ()
    sorted_names = sorted(
        names,
        key=lambda name: (
            ":" not in name,
            name.endswith(":latest"),
            name,
        ),
    )
    return sorted_names[0], tuple(sorted_names[1:])


def _get_ollama_models():
    """Get a list of models available on Ollama.

    Returns
    -------
    list[dict]
        A list of models available on Ollama. If the Ollama server is down, an empty
        list is returned.

    """
    try:
        return ollama.list()["models"]
    except:
        return []
