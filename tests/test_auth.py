from unittest.mock import Mock, patch

import pytest

from llm_ollama.auth import get_async_client, get_client


@pytest.fixture
def mock_basic_auth():
    with patch("llm_ollama.auth.httpx.BasicAuth") as mock:
        yield mock


@pytest.fixture
def mock_ollama_client():
    with patch("llm_ollama.auth.ollama.Client") as mock:
        yield mock


@pytest.fixture
def mock_ollama_async_client():
    with patch("llm_ollama.auth.ollama.AsyncClient") as mock:
        yield mock


def parametrize_clients():
    """Decorator to apply client parameterization to test methods."""
    return pytest.mark.parametrize(
        ("get_client_func", "mock_fixture"),
        [
            (get_client, "mock_ollama_client"),
            (get_async_client, "mock_ollama_async_client"),
        ],
    )


class TestAuthentication:
    """Tests for Ollama client authentication."""

    @parametrize_clients()
    def test_no_environment_variable(
        self,
        get_client_func,
        mock_fixture,
        request,
        monkeypatch,
    ):
        """Test client creation when OLLAMA_HOST is not set."""
        monkeypatch.delenv("OLLAMA_HOST", raising=False)
        mock_client_class = request.getfixturevalue(mock_fixture)

        get_client_func()

        mock_client_class.assert_called_once_with()

    @parametrize_clients()
    def test_host_without_auth(
        self,
        get_client_func,
        mock_fixture,
        request,
        monkeypatch,
    ):
        """Test client creation with host but without authentication."""
        monkeypatch.setenv("OLLAMA_HOST", "http://localhost:11434")
        mock_client_class = request.getfixturevalue(mock_fixture)

        get_client_func()

        mock_client_class.assert_called_once_with(host="http://localhost:11434")

    @parametrize_clients()
    def test_host_with_auth(
        self,
        get_client_func,
        mock_fixture,
        request,
        mock_basic_auth,
        monkeypatch,
    ):
        """Test client creation with host and authentication."""
        monkeypatch.setenv("OLLAMA_HOST", "http://user:pass@example.com:8080")
        mock_client_class = request.getfixturevalue(mock_fixture)
        mock_auth_instance = Mock()
        mock_basic_auth.return_value = mock_auth_instance

        get_client_func()

        mock_basic_auth.assert_called_once_with(username="user", password="pass")
        mock_client_class.assert_called_once_with(
            host="http://example.com:8080",
            auth=mock_auth_instance,
        )


@pytest.mark.parametrize(
    ("host_url", "expected_host", "expected_user", "expected_pass"),
    [
        ("http://user:pass@localhost:11434", "http://localhost:11434", "user", "pass"),
        (
            "https://admin:secret@secure.example.com",
            "https://secure.example.com",
            "admin",
            "secret",
        ),
        (
            "http://user%40domain:p%40ssw0rd@example.com:8080",
            "http://example.com:8080",
            "user@domain",
            "p@ssw0rd",
        ),
    ],
)
def test_various_auth_formats(
    host_url,
    expected_host,
    expected_user,
    expected_pass,
    mock_basic_auth,
    mock_ollama_client,
    monkeypatch,
):
    """Test parsing various URL formats with authentication."""
    monkeypatch.setenv("OLLAMA_HOST", host_url)
    mock_auth_instance = Mock()
    mock_basic_auth.return_value = mock_auth_instance

    get_client()

    mock_basic_auth.assert_called_once_with(
        username=expected_user,
        password=expected_pass,
    )
    mock_ollama_client.assert_called_once_with(
        host=expected_host,
        auth=mock_auth_instance,
    )
