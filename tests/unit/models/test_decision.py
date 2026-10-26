import json

import llm
import pydantic
import pytest
from llm.parts import Message, TextPart

from llm_ollama.models.decision import DecisionMixin

QUESTION = "Is this a person's name?"

_MODEL = DecisionMixin()


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
