"""This plugin's own llm tools: authenticated wrappers around ollama's web tools."""

import llm
import ollama

from llm_ollama.auth import get_client


@llm.hookimpl
def register_tools(register):
    register(
        llm.Tool(
            name="ollama_web_search",
            description="Search the web for information",
            implementation=_ollama_web_search,
        ),
    )
    register(
        llm.Tool(
            name="ollama_web_fetch",
            description="Fetch the contents of a web page",
            implementation=_ollama_web_fetch,
        ),
    )


# Docstrings below are copied verbatim from ollama.Client.web_search/web_fetch


def _ollama_web_search(
    query: str,
    max_results: int = 3,
) -> ollama.WebSearchResponse:
    """
    Performs a web search

    Args:
      query: The query to search for
      max_results: The maximum number of results to return (default: 3)

    Returns:
      WebSearchResponse with the search results
    Raises:
      ValueError: If OLLAMA_API_KEY environment variable is not set
    """
    return get_client().web_search(query, max_results)


def _ollama_web_fetch(url: str) -> ollama.WebFetchResponse:
    """
    Fetches the content of a web page for the provided URL.

    Args:
      url: The URL to fetch

    Returns:
      WebFetchResponse with the fetched result
    """
    return get_client().web_fetch(url)
