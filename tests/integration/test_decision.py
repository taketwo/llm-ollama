import json

import ollama
import pytest
from llm import get_async_model, get_model

pytestmark = pytest.mark.decision

REFUND_REQUEST = (
    "I was charged twice for the same order. Please refund the extra payment."
)
REFUND_QUESTION = "Is the customer asking for a refund?"

# The probabilities in the answers depend on the model and its build, so the tests check
# the shape of each answer and which way it leans on clear-cut inputs


@pytest.mark.parametrize(
    ("state", "leans_yes"),
    [
        (REFUND_REQUEST, True),
        ("The weather in Lisbon is sunny, so I am going to the beach.", False),
    ],
    ids=["yes", "no"],
)
def test_noul(model_id, state, leans_yes):
    """A noul question answers with the probability that the answer is yes."""
    response = get_model(model_id).prompt(state, system=REFUND_QUESTION)
    answer = json.loads(response.text())
    assert answer["type"] == "noul"
    assert (answer["noul"] > 0.5) == leans_yes


def test_choice(model_id):
    """A choice question picks one of the criteria and rates all of them."""
    criteria = {"billing": "Payments and refunds", "technical": "Software problems"}
    response = get_model(model_id).prompt(
        REFUND_REQUEST,
        system="Which department should handle this message?",
        answer_type="choice",
        criteria=criteria,
    )
    answer = json.loads(response.text())
    assert answer["type"] == "choice"
    assert answer["choice"] == "billing"
    assert answer["probabilities"].keys() == criteria.keys()


def test_score(model_id):
    """A score question rates the state between the lowest and highest level."""
    levels = ["Calm", "Annoyed", "Furious"]
    response = get_model(model_id).prompt(
        "This is outrageous! I am absolutely furious, I want my money back NOW!!!",
        system="How upset is the customer?",
        answer_type="score",
        criteria=levels,
    )
    answer = json.loads(response.text())
    assert answer["type"] == "score"
    assert 1 < answer["score"] <= len(levels) - 1
    assert answer["legend"] == {str(i): level for i, level in enumerate(levels)}


def test_usage_and_response_json(model_id):
    """A question reports its input tokens and keeps the server's whole result."""
    model = get_model(model_id)
    response = model.prompt(REFUND_REQUEST, system=REFUND_QUESTION)
    answer = json.loads(response.text())
    usage = response.usage()
    assert usage.input > 0
    assert usage.output == 0
    assert response.response_json["model"] == model.ollama_name
    assert list(response.response_json["answers"].values()) == [answer]


def test_server_error(model_id):
    """A question the server rejects raises its error."""
    response = get_model(model_id).prompt(
        REFUND_REQUEST,
        system="How upset is the customer?",
        answer_type="score",
        criteria=["Calm"],  # The server requires 2 to 10 levels
    )
    with pytest.raises(ollama.ResponseError) as excinfo:
        response.text()
    assert excinfo.value.status_code == 400


@pytest.mark.asyncio
async def test_async_usage_and_response_json(model_id):
    """Same as test_usage_and_response_json, through the async model."""
    model = get_async_model(model_id)
    response = model.prompt(REFUND_REQUEST, system=REFUND_QUESTION)
    answer = json.loads(await response.text())
    usage = await response.usage()
    assert usage.input > 0
    assert usage.output == 0
    assert response.response_json["model"] == model.ollama_name
    assert list(response.response_json["answers"].values()) == [answer]
