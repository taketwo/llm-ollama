import pytest
from llm import get_model
from llm.parts import ReasoningPart

pytestmark = pytest.mark.thinking


def test_thinking_surfaces_reasoning_part(model_id):
    """A thinking-capable model produces a ReasoningPart when think=True.

    Catches breakage in the chunk.message.thinking → reasoning StreamEvent
    mapping that mocks cannot see, e.g. if Ollama relocates the field or
    changes its shape across versions.
    """
    model = get_model(model_id)
    response = model.prompt(
        "Compute 7 times 13 step by step. Show your reasoning.",
        options={"temperature": 0, "think": True},
    )
    assert len(response.text()) > 0
    parts = response.messages()[-1].parts
    reasoning_parts = [p for p in parts if isinstance(p, ReasoningPart)]
    assert reasoning_parts, "Expected at least one ReasoningPart with think=True"
    assert any(p.text for p in reasoning_parts), "Expected non-empty reasoning text"


def test_think_false_surfaces_no_reasoning_part(model_id):
    """With think=False the model answers without reasoning.

    Without this, a ``think`` option that never reached the server would pass the test
    above on a model that thinks by default.
    """
    model = get_model(model_id)
    response = model.prompt(
        "Compute 7 times 13 step by step. Show your reasoning.",
        options={"temperature": 0, "think": False},
    )
    assert len(response.text()) > 0
    parts = response.messages()[-1].parts
    assert not any(isinstance(p, ReasoningPart) for p in parts)
