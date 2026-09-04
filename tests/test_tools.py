from llm import get_tools


def test_registered_tools():
    tool_names = get_tools().keys()
    assert "ollama_web_search" in tool_names
    assert "ollama_web_fetch" in tool_names
