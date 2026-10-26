"""Decision models served by Ollama."""

import json
from collections.abc import AsyncGenerator, Iterator
from typing import Literal, cast

import llm
import ollama
from llm.parts import Message, TextPart
from pydantic import Field, ValidationInfo, field_validator

from llm_ollama.auth import get_async_client, get_client
from llm_ollama.models.base import BaseMixin

# The ID of the one question each request asks; the same as llm-typesafe's, so the
# logged responses of both plugins have the same shape
_QUESTION_ID = "evaluation"


class _DecisionOptions(llm.Options):
    """What a decision model is asked and how it reads the prompt."""

    answer_type: Literal["noul", "choice", "score"] = Field(
        default="noul",
        description="noul: probability that the answer is yes; choice: the option that fits best; score: a degree on ordered levels.",
    )
    criteria: dict | list | str | None = Field(
        default=None,
        description='JSON. noul: optional {"true": "...", "false": "..."}; choice: {"name": "description or null", ...}; score: ["lowest level", ..., "highest level"].',
        # Checked when omitted too, since choice and score require it
        validate_default=True,
    )
    input_format: Literal["text", "json"] = Field(
        default="text",
        description="Read the prompt as text, or parse it as a JSON string, object or array.",
    )

    # A field validator rather than a model validator, so that errors name the field.
    # It reads answer_type from info.data, which holds only the fields declared above
    # criteria that validated.
    @field_validator("criteria")
    @classmethod
    def _check_criteria(
        cls, criteria: dict | list | str | None, info: ValidationInfo
    ) -> dict | list | str | None:
        """Parse ``criteria`` given as JSON text and check its shape for ``answer_type``.

        Counts and empty descriptions are left to the server.
        """
        if isinstance(criteria, str):
            try:
                criteria = json.loads(criteria)
            except json.JSONDecodeError as error:
                raise ValueError(f"criteria is not valid JSON: {error}") from error
        answer_type = info.data.get("answer_type")
        if answer_type == "noul":
            if criteria is not None and not (
                isinstance(criteria, dict)
                and criteria.keys() <= {"true", "false"}
                and all(isinstance(value, str) for value in criteria.values())
            ):
                raise ValueError(
                    'noul criteria must be omitted or an object with the keys "true" and/or "false" and string values'
                )
        elif answer_type == "choice":
            if not (
                isinstance(criteria, dict)
                and all(
                    value is None or isinstance(value, str)
                    for value in criteria.values()
                )
            ):
                raise ValueError(
                    "choice criteria must be an object that maps each choice to a description string or null"
                )
        elif answer_type == "score" and not (
            isinstance(criteria, list)
            and all(isinstance(level, str) for level in criteria)
        ):
            raise ValueError(
                "score criteria must be a list of level description strings, from lowest to highest"
            )
        return criteria


class DecisionMixin(BaseMixin):
    """Request and result handling shared by the sync and async Ollama decision models."""

    supports_conversation = False
    Options = _DecisionOptions

    def __init__(self, *, model_id: str, ollama_name: str) -> None:
        """Create the model llm knows as ``model_id`` and Ollama as ``ollama_name``."""
        self.model_id = model_id
        self.ollama_name = ollama_name

    def _prepare_systemone_kwargs(self, prompt: llm.Prompt) -> dict:
        """Build the arguments of an Ollama ``systemone`` call, other than the model.

        The system prompt is the question and the user prompt is the state. An empty
        state is passed on for the server to judge.

        Raises
        ------
        llm.ModelError
            If the system prompt is missing or blank, or if the state is not a JSON
            string, object or array when ``input_format`` is ``json``.

        """
        # llm rejects history, attachments and tools before execute, so the prompt
        # holds nothing but system and user text.
        instructions = _join_text(prompt.messages, "system")
        if not instructions.strip():
            raise llm.ModelError(
                "Pass the question as the system prompt: -s/--system on the command line, system= in Python"
            )
        state = _join_text(prompt.messages, "user")

        options = cast("_DecisionOptions", prompt.options)
        if options.input_format == "json":
            state = _parse_json_state(state)
        question: dict = {"type": options.answer_type, "instructions": instructions}
        if options.criteria is not None:
            question["criteria"] = options.criteria
        return {"state": state, "questions": {_QUESTION_ID: question}}

    def _record_result(
        self,
        response: llm.Response | llm.AsyncResponse,
        result: ollama.SystemOneResponse,
    ) -> str:
        """Store the ``systemone`` result and token usage on ``response``.

        Returns the answer to the question as JSON text.
        """
        response.response_json = result.model_dump()
        response.set_usage(
            input=result.usage.input_tokens,
            output=result.usage.output_tokens,
        )
        answer = result.answers[_QUESTION_ID]
        return json.dumps(answer.model_dump(), ensure_ascii=False)


class OllamaDecisionModel(DecisionMixin, llm.KeyModel):
    """A decision model served by Ollama."""

    def execute(
        self,
        prompt: llm.Prompt,
        stream: bool,  # noqa: FBT001
        response: llm.Response,
        conversation: llm.Conversation | None = None,
        key: str | None = None,
    ) -> Iterator[str]:
        """Put the prompt's question to the Ollama server, yielding the answer as JSON text."""
        # Unused: the model cannot stream, and llm rejects history before execute.
        del stream, conversation

        kwargs = self._prepare_systemone_kwargs(prompt)
        result = get_client(key=key).systemone(model=self.ollama_name, **kwargs)
        yield self._record_result(response, result)


class AsyncOllamaDecisionModel(DecisionMixin, llm.AsyncKeyModel):
    """An asynchronous decision model served by Ollama."""

    async def execute(
        self,
        prompt: llm.Prompt,
        stream: bool,  # noqa: FBT001
        response: llm.AsyncResponse,
        conversation: llm.AsyncConversation | None = None,
        key: str | None = None,
    ) -> AsyncGenerator[str, None]:
        """Put the prompt's question to the Ollama server, yielding the answer as JSON text."""
        # Unused: the model cannot stream, and llm rejects history before execute.
        del stream, conversation

        kwargs = self._prepare_systemone_kwargs(prompt)
        result = await get_async_client(key=key).systemone(
            model=self.ollama_name, **kwargs
        )
        yield self._record_result(response, result)


def _join_text(messages: list[Message], role: str) -> str:
    """Join the text parts of the ``role`` messages with blank lines."""
    return "\n\n".join(
        part.text
        for message in messages
        if message.role == role
        for part in message.parts
        if isinstance(part, TextPart)
    )


def _parse_json_state(text: str) -> str | dict | list:
    """Parse ``text`` as a JSON string, object or array, or raise ``llm.ModelError``."""
    try:
        state = json.loads(text)
    except json.JSONDecodeError as error:
        raise llm.ModelError(f"The prompt is not valid JSON: {error}") from error
    if not isinstance(state, (str, dict, list)):
        raise llm.ModelError("The JSON prompt must be a string, object or array")
    return state
