"""Authentication functionality for Ollama clients."""

import os
from typing import Optional, Tuple, Type, TypedDict, TypeVar, Union, TYPE_CHECKING
from urllib.parse import unquote, urlparse

import httpx
import ollama

if TYPE_CHECKING:
    from typing_extensions import NotRequired

# Timeout configuration for Ollama HTTP clients
# - No overall timeout (None) allows slow models/hardware to generate responses
# - Short connect timeout (1s) quickly fails when Ollama host is unreachable
# Ref: https://github.com/taketwo/llm-ollama/issues/52
DEFAULT_REQUEST_TIMEOUT = None
CONNECT_TIMEOUT = 1.0

T = TypeVar("T", bound=Union[ollama.Client, ollama.AsyncClient])


def get_client() -> ollama.Client:
    """Create an Ollama client with host and authentication set based on OLLAMA_HOST."""
    return _create_client(ollama.Client)


def get_async_client() -> ollama.AsyncClient:
    """Create an asynchronous Ollama client with host and authentication set based on OLLAMA_HOST."""
    return _create_client(ollama.AsyncClient)


def _parse_auth_from_url(url: str) -> Tuple[str, Optional[httpx.BasicAuth]]:
    """Parse URL and extract credentials if present.

    Parameters
    ----------
    url : str
        The URL to parse in the format http://username:password@host:port.

    Returns
    -------
    Tuple[str, Optional[httpx.BasicAuth]]
        A tuple containing the clean URL without credentials and an httpx.BasicAuth
        object if credentials were found, or None if no credentials were present.

    """
    parsed = urlparse(url)
    auth = None
    if parsed.username and parsed.password:
        auth = httpx.BasicAuth(
            username=unquote(parsed.username),
            password=unquote(parsed.password),
        )
    netloc = parsed.hostname
    if parsed.port:
        netloc = f"{netloc}:{parsed.port}"
    return parsed._replace(netloc=netloc).geturl(), auth


def _parse_headers_from_env() -> dict[str, str]:
    """Parse OLLAMA_HEADERS environment variable to extract custom HTTP headers.

    The variable should either be unset/empty, or contain a comma-separated list of
    key=value pairs. Malformed entries will raise a ValueError.

    """
    raw_headers = os.getenv("OLLAMA_HEADERS")
    collector: dict[str, str] = {}
    if raw_headers:
        for pair in raw_headers.split(","):
            if "=" not in pair:
                raise ValueError(
                    f"Invalid OLLAMA_HEADERS format: '{pair}' is missing '=' separator. "
                    f"Expected format: 'key1=value1,key2=value2'",
                )
            key, value = pair.split("=", 1)
            collector[key] = value
    return collector


def _parse_auth_from_env() -> Tuple[Optional[str], Optional[httpx.BasicAuth]]:
    """Parse OLLAMA_HOST environment variable and extract credentials."""
    host = os.getenv("OLLAMA_HOST")
    auth = None
    if host is not None:
        host, auth = _parse_auth_from_url(host)
    return host, auth


class ClientParams(TypedDict):
    host: "NotRequired[str]"
    timeout: "NotRequired[httpx.Timeout]"
    auth: "NotRequired[httpx.Auth]"
    headers: "dict[str, str]"


def _create_client(client_class: Type[T]):
    """Create a client with host, authentication, and headers set based on environment variables."""
    host, auth = _parse_auth_from_env()
    kwargs: ClientParams = {
        "timeout": httpx.Timeout(DEFAULT_REQUEST_TIMEOUT, connect=CONNECT_TIMEOUT),
        "headers": _parse_headers_from_env(),
    }
    if host is not None:
        kwargs["host"] = host
    if auth is not None:
        kwargs["auth"] = auth
    return client_class(**kwargs)
