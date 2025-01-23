import pytest

from llm import get_async_model


@pytest.mark.integration
@pytest.mark.asyncio
async def test_async_model_prompt():
    """Tests actual run. Needs llama3.2"""
    model = get_async_model("llama3.2:latest")
    response = model.prompt("a short poem about tea")
    response_text = await response.text()
    assert len(response_text) > 0
