from unittest.mock import ANY, Mock, patch

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


@pytest.fixture
def mock_get_key():
    with patch("llm_ollama.auth.llm.get_key", return_value=None) as mock:
        yield mock


def parametrize_clients():
    """Decorator to run the same test for both sync and async clients."""
    return pytest.mark.parametrize(
        ("get_client_func", "mock_fixture"),
        [
            (get_client, "mock_ollama_client"),
            (get_async_client, "mock_ollama_async_client"),
        ],
    )


class TestClientCreation:
    """Tests for Ollama client creation.

    Each test is parameterized over sync/async to verify parity. mock_get_key suppresses
    real LLM key lookup unless the test is specifically about API key behaviour.
    """

    @parametrize_clients()
    def test_defaults(
        self,
        get_client_func,
        mock_fixture,
        request,
        monkeypatch,
        mock_get_key,
    ):
        monkeypatch.delenv("OLLAMA_HOST", raising=False)
        monkeypatch.delenv("OLLAMA_HEADERS", raising=False)
        monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
        mock_client_class = request.getfixturevalue(mock_fixture)
        get_client_func()
        mock_client_class.assert_called_once_with(timeout=ANY, headers={})

    @parametrize_clients()
    def test_host_without_credentials(
        self,
        get_client_func,
        mock_fixture,
        request,
        monkeypatch,
        mock_get_key,
    ):
        monkeypatch.setenv("OLLAMA_HOST", "http://localhost:11434")
        monkeypatch.delenv("OLLAMA_HEADERS", raising=False)
        monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
        mock_client_class = request.getfixturevalue(mock_fixture)
        get_client_func()
        mock_client_class.assert_called_once_with(
            host="http://localhost:11434",
            headers={},
            timeout=ANY,
        )

    @parametrize_clients()
    def test_host_with_basic_auth(
        self,
        get_client_func,
        mock_fixture,
        request,
        mock_basic_auth,
        monkeypatch,
        mock_get_key,
    ):
        monkeypatch.setenv("OLLAMA_HOST", "http://user:pass@example.com:8080")
        monkeypatch.delenv("OLLAMA_HEADERS", raising=False)
        monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
        mock_client_class = request.getfixturevalue(mock_fixture)
        mock_auth_instance = Mock()
        mock_basic_auth.return_value = mock_auth_instance
        get_client_func()
        mock_basic_auth.assert_called_once_with(username="user", password="pass")
        mock_client_class.assert_called_once_with(
            host="http://example.com:8080",
            auth=mock_auth_instance,
            headers={},
            timeout=ANY,
        )

    @parametrize_clients()
    def test_custom_headers(
        self,
        get_client_func,
        mock_fixture,
        request,
        monkeypatch,
        mock_get_key,
    ):
        monkeypatch.delenv("OLLAMA_HOST", raising=False)
        monkeypatch.setenv("OLLAMA_HEADERS", "X-Custom-Header=value")
        monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
        mock_client_class = request.getfixturevalue(mock_fixture)
        get_client_func()
        mock_client_class.assert_called_once_with(
            headers={"X-Custom-Header": "value"},
            timeout=ANY,
        )

    @parametrize_clients()
    def test_api_key_injected_as_bearer_token(
        self,
        get_client_func,
        mock_fixture,
        request,
        monkeypatch,
        mock_get_key,
    ):
        monkeypatch.delenv("OLLAMA_HOST", raising=False)
        monkeypatch.delenv("OLLAMA_HEADERS", raising=False)
        monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
        mock_client_class = request.getfixturevalue(mock_fixture)
        mock_get_key.return_value = "test-key"
        get_client_func()
        mock_get_key.assert_called_once_with(alias="ollama", env="OLLAMA_API_KEY")
        mock_client_class.assert_called_once_with(
            timeout=ANY,
            headers={"Authorization": "Bearer test-key"},
        )

    @parametrize_clients()
    def test_ollama_headers_authorization_takes_precedence_over_api_key(
        self,
        get_client_func,
        mock_fixture,
        request,
        monkeypatch,
        mock_get_key,
    ):
        """Explicit Authorization in OLLAMA_HEADERS wins; get_key is not called at all."""
        monkeypatch.delenv("OLLAMA_HOST", raising=False)
        monkeypatch.setenv("OLLAMA_HEADERS", "Authorization=Bearer explicit-token")
        monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
        mock_client_class = request.getfixturevalue(mock_fixture)
        mock_get_key.return_value = "api-key"
        get_client_func()
        mock_get_key.assert_not_called()
        mock_client_class.assert_called_once_with(
            timeout=ANY,
            headers={"Authorization": "Bearer explicit-token"},
        )


@pytest.mark.parametrize(
    ("host_env", "expected_host", "expected_user", "expected_pass"),
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
def test_various_basic_auth_formats(
    host_env,
    expected_host,
    expected_user,
    expected_pass,
    mock_basic_auth,
    mock_ollama_client,
    mock_get_key,
    monkeypatch,
):
    """Test parsing various URL formats with basic authentication."""
    monkeypatch.delenv("OLLAMA_HEADERS", raising=False)
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
    monkeypatch.setenv("OLLAMA_HOST", host_env)
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
        headers={},
        timeout=ANY,
    )


@pytest.mark.parametrize(
    ("headers_env", "expected_headers"),
    [
        ("", {}),
        ("Authorization=Bearer TOKEN", {"Authorization": "Bearer TOKEN"}),
        (
            "Authorization=Bearer TOKEN,User-Agent=ollama-client",
            {"Authorization": "Bearer TOKEN", "User-Agent": "ollama-client"},
        ),
        (
            "X-API-Key=secret,Content-Type=application/json",
            {"X-API-Key": "secret", "Content-Type": "application/json"},
        ),
        (
            "Header With Spaces=value,Another-Header=another value",
            {"Header With Spaces": "value", "Another-Header": "another value"},
        ),
        ("Authorization:Bearer TOKEN", ValueError),
    ],
)
def test_various_ollama_headers_formats(
    headers_env,
    expected_headers,
    mock_ollama_client,
    mock_get_key,
    monkeypatch,
):
    """Test parsing various OLLAMA_HEADERS formats."""
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
    monkeypatch.setenv("OLLAMA_HEADERS", headers_env)
    if expected_headers is ValueError:
        with pytest.raises(ValueError, match="Invalid OLLAMA_HEADERS format"):
            get_client()
    else:
        get_client()
        mock_ollama_client.assert_called_once_with(
            timeout=ANY,
            headers=expected_headers,
        )
