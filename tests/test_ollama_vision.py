import json
import re

import pytest
from llm import Attachment, get_model, schema_dsl


@pytest.mark.vision
def test_attachments(vision_model):
    """Test attachment handling in conversations."""
    model = get_model(vision_model)

    # Get the image into history without cheating.
    box = open("tests/data/box.png", "rb").read()
    response = model.prompt(
        "What is written in the image?",
        system="""Role: Test assistant for conversation history.
Task: Warn about missing images/history. Keep replies short. Images are english text.""",
        attachments=[Attachment(content=box, type="image/png")],
        schema=schema_dsl("text"),
    )
    response_text = response.text()
    assert len(response_text) > 0
    json_response = json.loads(response_text)
    assert "shark" == json_response["text"].lower()


@pytest.mark.vision
def test_conversation_attachments(vision_model):
    """Test attachment handling in conversations."""
    model = get_model(vision_model)

    convo = model.conversation()

    # Get the image into history without cheating.
    box = open("tests/data/box.png", "rb").read()
    assert (
        "shark"
        not in convo.prompt(
            "Is image available, yes or no?",
            system="""Role: Test assistant for conversation history.
Task: Warn about missing images/history. Keep replies short. Images are one basic color.""",
            attachments=[Attachment(content=box, type="image/png")],
        )
        .text()
        .lower()
    )

    # Is image history available?
    response = convo.prompt(
        "Based on the history, what word is written in the image?",
        schema=schema_dsl("text"),
    )
    response_text = response.text()
    assert len(response_text) > 0
    json_response = json.loads(response_text)
    assert "shark" == json_response["text"].lower()
