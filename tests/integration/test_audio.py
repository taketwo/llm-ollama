import pytest
from llm import Attachment, get_model

pytestmark = pytest.mark.audio

# Every prompt turns thinking off, because gemma4 ignores media while it thinks, which
# it does by default, and sets temperature 0, because otherwise gemma4 now and then
# hears no speech in the clip


@pytest.mark.parametrize(
    ("audio_fixture", "audio_type"),
    [("shark_wav", "audio/wav"), ("shark_mp3", "audio/mpeg")],
    ids=["wav", "mp3"],
)
def test_attachments(request, model_id, audio_fixture, audio_type):
    """The model hears the word spoken in a WAV or MP3 attachment.

    Ollama tells audio from images by the file's signature, which differs between the
    two formats.
    """
    audio = request.getfixturevalue(audio_fixture)
    response = get_model(model_id).prompt(
        "What word is spoken in the audio?",
        attachments=[Attachment(content=audio, type=audio_type)],
        options={"temperature": 0, "think": False},
    )
    assert "shark" in response.text().lower()


def test_conversation_attachments(model_id, shark_wav):
    """Audio from an earlier turn stays in the conversation's history."""
    conversation = get_model(model_id).conversation()

    # Get the audio into history without the word in the reply
    first_response = conversation.prompt(
        "Is there speech in the audio? Answer yes or no.",
        attachments=[Attachment(content=shark_wav, type="audio/wav")],
        options={"temperature": 0, "think": False},
    )
    assert "shark" not in first_response.text().lower()

    response = conversation.prompt(
        "What word is spoken in the audio?",
        options={"temperature": 0, "think": False},
    )
    assert "shark" in response.text().lower()
