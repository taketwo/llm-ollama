import json

import pytest
from llm import get_async_model, get_model, schema_dsl


@pytest.mark.integration
@pytest.mark.asyncio
async def test_async_model_prompt(integration_model):
    """Tests actual run."""
    model = get_async_model(integration_model)
    response = model.prompt("a short poem about tea")
    response_text = await response.text()
    assert len(response_text) > 0


@pytest.mark.integration
@pytest.mark.asyncio
async def test_async_model_prompt_with_schema(integration_model):
    """Tests actual run."""
    model = get_async_model(integration_model)
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


@pytest.mark.integration
def test_tools(integration_model):
    """Test tool execution."""

    def multiply(a: int, b: int):
        "Multiply two integers"
        return int(a) * int(b)

    model = get_model(integration_model)
    chain = model.chain(
        "What is 12345 * 4312? Use the available tools to compute the answer.",
        tools=[multiply],
        options={"temperature": 0, "think": False},
    )
    result = chain.text()
    assert "53231640" in result or "53,231,640" in result


@pytest.mark.integration
@pytest.mark.asyncio
async def test_async_tools(integration_model):
    """Async streaming + tool execution end-to-end against a real Ollama server.

    The async streaming branch's tool-call capture was a recent fix and is not
    otherwise exercised live. Mirrors test_tools so the only difference is sync
    vs async.
    """

    def multiply(a: int, b: int):
        "Multiply two integers"
        return int(a) * int(b)

    model = get_async_model(integration_model)
    chain = model.chain(
        "What is 12345 * 4312? Use the available tools to compute the answer.",
        tools=[multiply],
        options={"temperature": 0, "think": False},
    )
    result = await chain.text()
    assert "53231640" in result or "53,231,640" in result


@pytest.mark.integration
def test_thinking_surfaces_reasoning_part(integration_model):
    """A thinking-capable model produces a ReasoningPart when think=True.

    Catches breakage in the chunk.message.thinking → reasoning StreamEvent
    mapping that mocks cannot see, e.g. if Ollama relocates the field or
    changes its shape across versions.
    """
    from llm.parts import ReasoningPart

    model = get_model(integration_model)
    response = model.prompt(
        "Compute 7 times 13 step by step. Show your reasoning.",
        options={"temperature": 0, "think": True},
    )
    assert len(response.text()) > 0
    parts = response.messages()[-1].parts
    reasoning_parts = [p for p in parts if isinstance(p, ReasoningPart)]
    assert reasoning_parts, "Expected at least one ReasoningPart with think=True"
    assert any(p.text for p in reasoning_parts), "Expected non-empty reasoning text"
