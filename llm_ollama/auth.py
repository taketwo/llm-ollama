"""Authentication functionality for Ollama clients."""

import os
from typing import NotRequired, Optional, Tuple, Type, TypeVar, Union, TypedDict
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


def _parse_auth_from_env() -> Tuple[Optional[str], Optional[httpx.BasicAuth]]:
    """Parse OLLAMA_HOST environment variable and extract credentials if present."""
    host = os.getenv("OLLAMA_HOST")
    if not host:
class ClientParams(TypedDict):
    host: str
    timeout: NotRequired[httpx.Timeout]
    auth: NotRequired[httpx.Auth]

def _create_client(client_class):
    """Create a client with host and authentication set based on OLLAMA_HOST."""
    host, auth = _parse_auth_from_env()
    kwargs: ClientParams = {
        "timeout": httpx.Timeout(DEFAULT_REQUEST_TIMEOUT, connect=CONNECT_TIMEOUT),
        "host": host
    }
    if host:
        kwargs["host"] = host
    if auth:
        kwargs["auth"] = auth
    return client_class(**kwargs)
