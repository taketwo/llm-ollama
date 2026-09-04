"""This plugin's own llm tools."""

import llm
import ollama


@llm.hookimpl
def register_tools(register):
    register(
        llm.Tool(
            name="ollama_web_search",
            description="Search the web for information",
            implementation=ollama.web_search,
        ),
    )
    register(
        llm.Tool(
            name="ollama_web_fetch",
            description="Fetch the contents of a web page",
            implementation=ollama.web_fetch,
        ),
    )
