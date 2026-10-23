import json

import pytest
from llm import get_async_model, get_model, schema_dsl

pytestmark = pytest.mark.completion

stream_modes = pytest.mark.parametrize(
    "stream",
    [True, False],
    ids=["stream", "no-stream"],
)


@stream_modes
def test_prompt(model_id, stream):
    """A prompt returns text and its token usage, streamed or not.

    Streamed and non-streamed responses arrive in different shapes, and only a real
    server shows which fields each one fills.
    """
    response = get_model(model_id).prompt("Say hello in one word.", stream=stream)
    assert len(response.text()) > 0
    usage = response.usage()
    assert usage.input > 0
    assert usage.output > 0


@pytest.mark.asyncio
@stream_modes
async def test_async_prompt(model_id, stream):
    """Same as test_prompt, through the async model."""
    response = get_async_model(model_id).prompt("Say hello in one word.", stream=stream)
    assert len(await response.text()) > 0
    usage = await response.usage()
    assert usage.input > 0
    assert usage.output > 0


@pytest.mark.asyncio
async def test_async_prompt_with_schema(model_id):
    """A prompt with a schema returns JSON that follows it."""
    model = get_async_model(model_id)
    response = model.prompt(
        "Describe a nice dog with a surprising name",
        schema=schema_dsl("name, age int, bio"),
    )
    response_text = await response.text()
    assert len(response_text) > 0
    json_response = json.loads(response_text)
    assert "name" in json_response
    assert "bio" in json_response
    assert "age" in json_response
    assert isinstance(json_response["age"], int)


def test_json_object(model_id):
    """``json_object`` makes the model answer in JSON."""
    response = get_model(model_id).prompt(
        "Describe a cat as a JSON object.",
        json_object=True,
    )
    json.loads(response.text())


def test_conversation(model_id):
    """A prompt in a conversation sees the turns before it."""
    conversation = get_model(model_id).conversation()
    # A conversation adds a turn to its history only once the response completes
    conversation.prompt("My cat is called Biscotti. Reply with OK.").text()
    response = conversation.prompt("What is my cat called? Answer with the name only.")
    assert "biscotti" in response.text().lower()
