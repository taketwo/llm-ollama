import json
from typing import Optional

import httpx
import llm
from pydantic import Field


@llm.hookimpl
def register_models(register):
    register(Ollama("llama2"))


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
        body = {
            "model": self.model_id,
            "messages": messages,
            "options": {},
            "stream": False,
        }
        if prompt.options.temperature:
            body["options"]["temperature"] = prompt.options.temperature

        if stream:
            body["stream"] = True
            with httpx.Client() as client, client.stream(
                method="POST",
                url="http://localhost:11434/api/chat",
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                },
                json=body,
                timeout=None,
            ) as r:
                for line in r.iter_lines():
                    try:
                        response.response_json = json.loads(line)
                        yield response.response_json["message"]["content"]
                    except json.decoder.JSONDecodeError:
                        raise llm.ModelError(
                            "Failed to parse JSON returned by the server",
                        )
                    except KeyError:
                        raise llm.ModelError(
                            "JSON returned by the server does not have expected structure",
                        )
        else:
            with httpx.Client() as client:
                api_response = client.post(
                    "http://localhost:11434/api/chat",
                    headers={
                        "Content-Type": "application/json",
                        "Accept": "application/json",
                    },
                    json=body,
                    timeout=None,
                )
                api_response.raise_for_status()
                yield api_response.json()["message"]["content"]
                response.response_json = api_response.json()

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
