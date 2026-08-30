from collections.abc import Callable
from typing import NamedTuple
from unittest.mock import ANY, Mock, patch

import pytest

from llm_ollama.auth import _parse_auth_from_url, get_async_client, get_client


pytestmark = pytest.mark.usefixtures("bare_env")


@pytest.fixture(autouse=True)
def mock_get_key():
    """Suppress the real key lookup, which would otherwise read the developer's keys."""
    with patch("llm_ollama.auth.llm.get_key", return_value=None) as mock:
        yield mock


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


class _ClientUnderTest(NamedTuple):
    """A client factory paired with the ollama class it is expected to construct."""

    create: Callable
    cls: Mock


@pytest.fixture(
    params=[
        (get_client, "mock_ollama_client"),
        (get_async_client, "mock_ollama_async_client"),
    ],
    ids=["sync", "async"],
)
def client(request):
    """The factory under test, parameterized over sync and async to verify parity."""
    create, mock_name = request.param
    return _ClientUnderTest(create, request.getfixturevalue(mock_name))


class TestClientCreation:
    """Tests for Ollama client creation."""

    def test_defaults(self, client):
        client.create()
        client.cls.assert_called_once_with(timeout=ANY, headers={})

    def test_host_without_credentials(self, client, monkeypatch):
        monkeypatch.setenv("OLLAMA_HOST", "http://localhost:11434")
        client.create()
        client.cls.assert_called_once_with(
            host="http://localhost:11434",
            headers={},
            timeout=ANY,
        )

    def test_host_with_basic_auth(self, client, mock_basic_auth, monkeypatch):
        monkeypatch.setenv("OLLAMA_HOST", "http://user:pass@example.com:8080")
        mock_basic_auth.return_value = mock_auth_instance = Mock()
        client.create()
        mock_basic_auth.assert_called_once_with(username="user", password="pass")
        client.cls.assert_called_once_with(
            host="http://example.com:8080",
            auth=mock_auth_instance,
            headers={},
            timeout=ANY,
        )

    def test_custom_headers(self, client, monkeypatch):
        monkeypatch.setenv("OLLAMA_HEADERS", "X-Custom-Header=value")
        client.create()
        client.cls.assert_called_once_with(
            headers={"X-Custom-Header": "value"},
            timeout=ANY,
        )

    def test_api_key_injected_as_bearer_token(self, client, mock_get_key):
        mock_get_key.return_value = "test-key"
        client.create()
        mock_get_key.assert_called_once_with(alias="ollama", env="OLLAMA_API_KEY")
        client.cls.assert_called_once_with(
            timeout=ANY,
            headers={"Authorization": "Bearer test-key"},
        )

    def test_ollama_headers_authorization_takes_precedence_over_api_key(
        self,
        client,
        mock_get_key,
        monkeypatch,
    ):
        """Explicit Authorization in OLLAMA_HEADERS wins; get_key is not called at all."""
        monkeypatch.setenv("OLLAMA_HEADERS", "Authorization=Bearer explicit-token")
        mock_get_key.return_value = "api-key"
        client.create()
        mock_get_key.assert_not_called()
        client.cls.assert_called_once_with(
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
    monkeypatch,
):
    """Test parsing various URL formats with basic authentication."""
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
    "url",
    [
        "unix:///var/run/ollama.sock",
        "/var/run/ollama.sock",
        "localhost:11434",
    ],
)
def test_parse_auth_from_url_passes_through_hostless_url(url):
    """A URL with no hostname round-trips unchanged with no auth.

    These shapes (unix socket, bare path, host:port without scheme) have no place to
    carry credentials, so the parser must not rebuild the netloc — doing so would inject
    the literal string "None" as the host.
    """
    assert _parse_auth_from_url(url) == (url, None)


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
    monkeypatch,
):
    """Test parsing various OLLAMA_HEADERS formats."""
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
