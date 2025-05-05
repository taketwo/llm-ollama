"""Authentication functionality for Ollama clients."""

import os
from typing import Optional, Tuple
from urllib.parse import unquote, urlparse

import httpx
import ollama


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
        return None, None
    return _parse_auth_from_url(host)


def _create_client(client_class):
    """Create a client with host and authentication set based on OLLAMA_HOST."""
    host, auth = _parse_auth_from_env()
    kwargs = {}
    if host:
        kwargs["host"] = host
    if auth:
        kwargs["auth"] = auth
    return client_class(**kwargs)
