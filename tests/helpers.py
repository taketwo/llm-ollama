"""Plain assertion/setup helpers shared across test modules."""


def assert_bearer_token(client_class, token):
    _, kwargs = client_class.call_args
    assert kwargs["headers"]["Authorization"] == f"Bearer {token}"
