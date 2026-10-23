import pytest
from llm import get_embedding_model

pytestmark = pytest.mark.embedding


def test_embed(model_id):
    """A text embeds as a non-empty vector of floats."""
    vector = get_embedding_model(model_id).embed("Sharks swim in the sea.")
    assert len(vector) > 0
    assert all(isinstance(value, float) for value in vector)


def test_embed_multi(model_id):
    """Several texts embed as one vector each, all of the same length."""
    model = get_embedding_model(model_id)
    vectors = list(model.embed_multi(["one", "two", "three"]))
    assert len(vectors) == 3
    assert len({len(vector) for vector in vectors}) == 1
