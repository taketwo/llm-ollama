import inspect
import json
from dataclasses import dataclass
from unittest.mock import AsyncMock, Mock

import llm
import ollama
import pydantic
import pytest
from helpers import assert_bearer_token
from llm.parts import Message, TextPart

from llm_ollama.models.decision import (
    AsyncOllamaDecisionModel,
    DecisionMixin,
    OllamaDecisionModel,
)

QUESTION = "Is this a person's name?"

_MODEL = OllamaDecisionModel(model_id="ollama/tev1:0.8b", ollama_name="tev1:0.8b")


def _prepare_kwargs(state="Alice", system=QUESTION, **options):
    """Build the systemone arguments for a prompt with the given state, question and options."""
    prompt = llm.Prompt(
        state, _MODEL, system=system, options=DecisionMixin.Options(**options)
    )
    return _MODEL._prepare_systemone_kwargs(prompt)


def _prepare_kwargs_from_messages(*messages):
    prompt = llm.Prompt("", _MODEL, messages=list(messages))
    return _MODEL._prepare_systemone_kwargs(prompt)


def _get_question(kwargs):
    return kwargs["questions"]["evaluation"]


def test_noul_without_criteria_is_the_default():
    assert _prepare_kwargs() == {
        "state": "Alice",
        "questions": {"evaluation": {"type": "noul", "instructions": QUESTION}},
    }


@pytest.mark.parametrize(
    ("answer_type", "criteria"),
    [
        ("noul", {"true": "A person's name", "false": "Anything else"}),
        ("noul", {"true": "A person's name"}),
        ("noul", {"false": "Anything else"}),
        ("noul", {}),
        ("choice", {"person": "A person's name", "place": None}),
        ("score", ["Not a name", "Possibly a name", "Surely a name"]),
    ],
)
def test_criteria_reach_the_question(answer_type, criteria):
    kwargs = _prepare_kwargs(answer_type=answer_type, criteria=criteria)

    assert _get_question(kwargs) == {
        "type": answer_type,
        "instructions": QUESTION,
        "criteria": criteria,
    }


@pytest.mark.parametrize(
    ("answer_type", "criteria"),
    [
        ("noul", {"true": "A person's name", "false": "Anything else"}),
        ("choice", {"person": "A person's name", "place": None}),
        ("score", ["Not a name", "Surely a name"]),
    ],
)
def test_criteria_given_as_json_text_are_parsed(answer_type, criteria):
    kwargs = _prepare_kwargs(answer_type=answer_type, criteria=json.dumps(criteria))

    assert _get_question(kwargs)["criteria"] == criteria


def test_criteria_that_are_not_json_are_rejected():
    with pytest.raises(pydantic.ValidationError, match="criteria is not valid JSON"):
        DecisionMixin.Options(answer_type="choice", criteria="{not json")


@pytest.mark.parametrize(
    ("answer_type", "criteria", "expected"),
    [
        ("noul", ["true", "false"], "noul criteria must be omitted or an object"),
        ("noul", {"maybe": "Unsure"}, 'keys "true" and/or "false"'),
        ("noul", {"true": "Yes", "other": "No"}, 'keys "true" and/or "false"'),
        ("noul", {"true": 1}, "string values"),
        ("noul", {"true": None}, "string values"),
        ("noul", '["a"]', "noul criteria must be omitted or an object"),
        ("choice", None, "choice criteria must be an object"),
        ("choice", ["a", "b"], "choice criteria must be an object"),
        ("choice", {"a": 1, "b": None}, "choice criteria must be an object"),
        ("choice", {"a": ["x"]}, "choice criteria must be an object"),
        ("choice", '["a", "b"]', "choice criteria must be an object"),
        ("choice", "null", "choice criteria must be an object"),
        ("score", None, "score criteria must be a list"),
        ("score", {"low": "Low", "high": "High"}, "score criteria must be a list"),
        ("score", ["Low", 2], "score criteria must be a list"),
        ("score", ["Low", None], "score criteria must be a list"),
        ("score", '{"low": "Low"}', "score criteria must be a list"),
    ],
)
def test_criteria_of_the_wrong_shape_are_rejected(answer_type, criteria, expected):
    with pytest.raises(pydantic.ValidationError, match=expected):
        DecisionMixin.Options(answer_type=answer_type, criteria=criteria)


@pytest.mark.parametrize(
    ("answer_type", "criteria"),
    [
        ("choice", {"only": "The one option"}),
        ("choice", {}),
        ("choice", {"": "Nameless", "b": ""}),
        ("score", ["Only level"]),
        ("score", []),
        ("score", [f"Level {i}" for i in range(11)]),
        ("score", ["", ""]),
    ],
)
def test_counts_and_empty_strings_are_left_to_the_server(answer_type, criteria):
    kwargs = _prepare_kwargs(answer_type=answer_type, criteria=criteria)

    assert _get_question(kwargs)["criteria"] == criteria


def test_criteria_errors_name_the_field():
    with pytest.raises(pydantic.ValidationError) as excinfo:
        DecisionMixin.Options(answer_type="choice")

    assert [error["loc"] for error in excinfo.value.errors()] == [("criteria",)]


def test_unknown_answer_type_is_rejected_without_checking_criteria():
    with pytest.raises(pydantic.ValidationError) as excinfo:
        DecisionMixin.Options(answer_type="rank", criteria=["Low", "High"])

    assert [error["loc"] for error in excinfo.value.errors()] == [("answer_type",)]


def test_system_and_user_parts_are_joined_with_blank_lines():
    kwargs = _prepare_kwargs_from_messages(
        Message(
            role="system", parts=[TextPart(text="First."), TextPart(text="Second.")]
        ),
        Message(role="system", parts=[TextPart(text="Third.")]),
        Message(role="user", parts=[TextPart(text="One"), TextPart(text="Two")]),
        Message(role="user", parts=[TextPart(text="Three")]),
    )

    assert kwargs["state"] == "One\n\nTwo\n\nThree"
    assert _get_question(kwargs)["instructions"] == "First.\n\nSecond.\n\nThird."


def test_text_input_format_leaves_json_text_alone():
    assert _prepare_kwargs(state='{"name": "Alice"}')["state"] == '{"name": "Alice"}'


@pytest.mark.parametrize(
    ("state", "expected"),
    [
        ('{"name": "Alice", "age": 30}', {"name": "Alice", "age": 30}),
        ('["Alice", "Bob"]', ["Alice", "Bob"]),
        ('"Alice"', "Alice"),
        ("{}", {}),
        ("[]", []),
    ],
)
def test_json_input_format_parses_the_state(state, expected):
    assert _prepare_kwargs(state=state, input_format="json")["state"] == expected


@pytest.mark.parametrize("state", ["{Alice}", ""])
def test_json_input_format_rejects_invalid_json(state):
    with pytest.raises(llm.ModelError, match="not valid JSON"):
        _prepare_kwargs(state=state, input_format="json")


@pytest.mark.parametrize("state", ["42", "1.5", "true", "null"])
def test_json_input_format_rejects_scalars_other_than_strings(state):
    with pytest.raises(llm.ModelError, match="string, object or array"):
        _prepare_kwargs(state=state, input_format="json")


@pytest.mark.parametrize("system", [None, "", "  \n\t"])
def test_missing_question_is_rejected(system):
    with pytest.raises(llm.ModelError, match="-s/--system"):
        _prepare_kwargs(system=system)


def test_empty_state_is_passed_through():
    assert _prepare_kwargs(state="")["state"] == ""


def _build_result(answer=None, *, input_tokens=115, output_tokens=0):
    """Build the ``systemone`` result of a model that gave ``answer`` to the one question."""
    return ollama.SystemOneResponse.model_validate(
        {
            "model": "tev1:0.8b",
            "answers": {"evaluation": answer or {"type": "noul", "noul": 0.86}},
            "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
        }
    )


async def _resolve(value):
    """Await ``value`` if it is awaitable (an async response), so one test serves both."""
    return await value if inspect.isawaitable(value) else value


@dataclass
class _MockedDecision:
    """A decision model whose Ollama client is a mock."""

    model: OllamaDecisionModel | AsyncOllamaDecisionModel
    client: Mock
    # The patched ollama.Client or ollama.AsyncClient, which returns client
    client_class: Mock


@pytest.fixture(
    params=[OllamaDecisionModel, AsyncOllamaDecisionModel], ids=["sync", "async"]
)
def decision(request, mocker):
    """Provide a sync and an async decision model whose client answers ``noul``."""
    is_async = request.param is AsyncOllamaDecisionModel
    client = AsyncMock() if is_async else Mock()
    client.systemone.return_value = _build_result()
    client_class = mocker.patch(
        "ollama.AsyncClient" if is_async else "ollama.Client", return_value=client
    )
    model = request.param(model_id="ollama/tev1:0.8b", ollama_name="tev1:0.8b")
    return _MockedDecision(model=model, client=client, client_class=client_class)


@pytest.mark.asyncio
async def test_request_goes_to_the_server_by_ollama_name(decision):
    response = decision.model.prompt("Alice", system=QUESTION)

    await _resolve(response.text())

    decision.client.systemone.assert_called_once_with(
        model="tev1:0.8b",
        state="Alice",
        questions={"evaluation": {"type": "noul", "instructions": QUESTION}},
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        (
            {"type": "noul", "noul": 0.86},
            '{"type": "noul", "noul": 0.86}',
        ),
        (
            {
                "type": "choice",
                "choice": "имя",
                "probabilities": {"имя": 0.75, "место": 0.25},
                "confidence": 0.5,
            },
            '{"type": "choice", "choice": "имя", "probabilities": {"имя": 0.75, "место": 0.25}, "confidence": 0.5}',
        ),
    ],
)
async def test_answer_is_yielded_as_json_text(decision, answer, expected):
    decision.client.systemone.return_value = _build_result(answer)
    response = decision.model.prompt("Alice", system=QUESTION)

    assert await _resolve(response.text()) == expected


@pytest.mark.asyncio
async def test_usage_reports_input_and_output_tokens(decision):
    decision.client.systemone.return_value = _build_result(
        input_tokens=115, output_tokens=7
    )
    response = decision.model.prompt("Alice", system=QUESTION)

    usage = await _resolve(response.usage())

    assert (usage.input, usage.output) == (115, 7)


@pytest.mark.asyncio
async def test_full_result_is_stored_as_response_json(decision):
    response = decision.model.prompt("Alice", system=QUESTION)

    await _resolve(response.text())

    assert response.response_json == {
        "model": "tev1:0.8b",
        "answers": {"evaluation": {"type": "noul", "noul": 0.86}},
        "usage": {"input_tokens": 115, "output_tokens": 0},
    }


@pytest.mark.asyncio
async def test_key_reaches_client_as_bearer_token(decision):
    response = decision.model.prompt("Alice", system=QUESTION, key="caller-key")

    await _resolve(response.text())

    assert_bearer_token(decision.client_class, "caller-key")


@pytest.mark.asyncio
async def test_server_error_propagates(decision):
    decision.client.systemone.side_effect = ollama.ResponseError(
        "state must not be empty", 400
    )
    response = decision.model.prompt("", system=QUESTION)

    with pytest.raises(ollama.ResponseError, match="state must not be empty"):
        await _resolve(response.text())


@pytest.mark.asyncio
async def test_conversation_cannot_continue(decision):
    conversation = decision.model.conversation()
    await _resolve(conversation.prompt("Alice", system=QUESTION).text())

    with pytest.raises(llm.ConversationNotSupported):
        conversation.prompt("Bob", system=QUESTION)
