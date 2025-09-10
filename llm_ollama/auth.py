"""Authentication functionality for Ollama clients."""

import os
from typing import NotRequired, Optional, Tuple, Type, TypedDict, TypeVar, Union
from urllib.parse import unquote, urlparse

import httpx
import ollama

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


def _parse_headers_from_env() -> Optional[dict[str, str]]:
    """
    Parses the headers to pass to Ollama in a comma-separated string - e.g.

    OLLAMA_HEADERS='Authorization=Bearer TOKEN,User-Agent=ollama-client'

    Flexible approach to allow settting HTTP headers from a variety of deployment environments - Ollama
    behind OpenWebUI with an auth token, Cloudflare tunnel with a service token, etc.
    """

    raw_headers = os.getenv("OLLAMA_HEADERS")
    if raw_headers is not None:
        collector: dict[str, str] = {}
        headers = raw_headers.split(",")
        for pair in headers:
            key, value = pair.split("=")
            collector[key] = value
        return collector

    else:
        return None


def _parse_auth_from_env() -> Tuple[
    Optional[str], Optional[httpx.BasicAuth], Optional[dict[str, str]]
]:
    """Parse OLLAMA_HOST environment variable and extract credentials and custom headers if present."""
    host = os.getenv("OLLAMA_HOST")
    auth = None
    if host is not None:
        host, auth = _parse_auth_from_url(host)
    headers = _parse_headers_from_env()

    return host, auth, headers


class ClientParams(TypedDict):
    host: NotRequired[str]
    timeout: NotRequired[httpx.Timeout]
    auth: NotRequired[httpx.Auth]
    headers: NotRequired[dict[str, str]]


def _create_client(client_class: Type[T]):
    """Create a client with host and authentication set based on OLLAMA_HOST."""
    host, auth, headers = _parse_auth_from_env()
    kwargs: ClientParams = {
        "timeout": httpx.Timeout(DEFAULT_REQUEST_TIMEOUT, connect=CONNECT_TIMEOUT)
    }
    if host is not None:
        kwargs["host"] = host
    if auth is not None:
        kwargs["auth"] = auth
    if headers is not None:
        kwargs["headers"] = headers
    return client_class(**kwargs)
