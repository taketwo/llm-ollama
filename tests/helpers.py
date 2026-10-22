"""Plain assertion/setup helpers shared across test modules."""


def assert_bearer_token(client_class, token):
    _, kwargs = client_class.call_args
    assert kwargs["headers"]["Authorization"] == f"Bearer {token}"


def make_show_fail(client, name, error):
    """Make the mocked ``client.show(name)`` raise ``error``; other names are unaffected.

    Returns the previous ``show`` side effect, which restores the model when reassigned.
    """
    show = client.show.side_effect

    def failing_show(shown_name):
        if shown_name == name:
            raise error
        return show(shown_name)

    client.show.side_effect = failing_show
    return show
