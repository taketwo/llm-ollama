import json

import pytest
from llm import get_async_model, schema_dsl

pytestmark = pytest.mark.completion


@pytest.mark.asyncio
async def test_async_model_prompt(model_id):
    """A prompt returns text."""
    model = get_async_model(model_id)
    response = model.prompt("a short poem about tea")
    response_text = await response.text()
    assert len(response_text) > 0


@pytest.mark.asyncio
async def test_async_model_prompt_with_schema(model_id):
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
