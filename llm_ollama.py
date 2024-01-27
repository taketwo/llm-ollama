from typing import Optional

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
        for model in ollama.list()["models"]:
            click.echo(model["name"])


@llm.hookimpl
def register_models(register):
    for model in ollama.list()["models"]:
        register(Ollama(model["name"]))


class Ollama(llm.Model):
    can_stream: bool = True

    class Options(llm.Options):
        temperature: Optional[float] = Field(
            description=(
                "The temperature of the model. Increasing the temperature will make the model answer more creatively."
            ),
            ge=0,
            le=1,
            default=0.8,
        )
        # TODO: Implement more options

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
        response._prompt_json = {"messages": messages}
        options = {}
        if prompt.options.temperature:
            options["temperature"] = prompt.options.temperature

        if stream:
            response_stream = ollama.chat(
                model=self.model_id,
                messages=messages,
                stream=True,
                options=options,
            )
            for chunk in response_stream:
                yield chunk["message"]["content"]
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
