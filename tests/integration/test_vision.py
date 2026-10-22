import json

import pytest
from llm import Attachment, get_model, schema_dsl

pytestmark = pytest.mark.vision


def test_attachments(model_id, box_png):
    """Test attachment handling in conversations."""
    model = get_model(model_id)

    response = model.prompt(
        "What is written in the image?",
        system="""Role: Test assistant for conversation history.
Task: Warn about missing images/history. Keep replies short. Images are english text.""",
        attachments=[Attachment(content=box_png, type="image/png")],
        schema=schema_dsl("text"),
        think=False,
    )
    response_text = response.text()
    assert len(response_text) > 0
    json_response = json.loads(response_text)
    assert json_response["text"].lower() == "shark"


def test_conversation_attachments(model_id, box_png):
    """Test attachment handling in conversations."""
    model = get_model(model_id)

    convo = model.conversation()

    # Get the image into history without cheating.
    assert (
        "shark"
        not in convo.prompt(
            "Is image available, yes or no?",
            system="""Role: Test assistant for conversation history.
Task: Warn about missing images/history. Keep replies short. Images are one basic color.""",
            attachments=[Attachment(content=box_png, type="image/png")],
            think=False,
        )
        .text()
        .lower()
    )

    # Is image history available?
    response = convo.prompt(
        "Based on the history, what word is written in the image?",
        schema=schema_dsl("text"),
        think=False,
    )
    response_text = response.text()
    assert len(response_text) > 0
    json_response = json.loads(response_text)
    assert json_response["text"].lower() == "shark"
