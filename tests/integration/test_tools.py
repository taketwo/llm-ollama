import pytest
from llm import get_async_model, get_model

pytestmark = pytest.mark.tools


def multiply(a: int, b: int):
    "Multiply two integers"
    return int(a) * int(b)


def test_tools(model_id):
    """A chain calls the tool and answers with its result."""
    model = get_model(model_id)
    chain = model.chain(
        "What is 12345 * 4312? Use the available tools to compute the answer.",
        tools=[multiply],
        options={"temperature": 0, "think": False},
    )
    result = chain.text()
    assert "53231640" in result or "53,231,640" in result


@pytest.mark.asyncio
async def test_async_tools(model_id):
    """Same as test_tools, through the async model."""
    model = get_async_model(model_id)
    chain = model.chain(
        "What is 12345 * 4312? Use the available tools to compute the answer.",
        tools=[multiply],
        options={"temperature": 0, "think": False},
    )
    result = await chain.text()
    assert "53231640" in result or "53,231,640" in result
