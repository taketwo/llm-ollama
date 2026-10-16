"""Fixtures for unit tests.

The autouse fixtures isolate each test from the developer's environment and llm user
directory, which integration tests rely on, so they live here rather than in the root
conftest.
"""

from pathlib import Path

import ollama
import pytest

from llm_ollama import registry


@pytest.fixture(autouse=True)
def bare_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Clear the variables client creation reads, so a test sets only what it is about."""
    for name in ("OLLAMA_HOST", "OLLAMA_HEADERS", "OLLAMA_API_KEY"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture(autouse=True)
def _isolated_user_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Give each test an empty llm user directory and capability cache."""
    monkeypatch.setenv("LLM_USER_PATH", str(tmp_path))
    # The cache resolved its directory when registry was imported, before LLM_USER_PATH
    # was set here
    monkeypatch.setattr(registry.cache, "cache_dir", tmp_path)


@pytest.fixture
def mock_ollama_client(mocker):
    """Mock ``ollama.Client`` as a server with the models below and return the client."""
    return_value = {
        "models": [
            {
                "model": "stable-code:3b",
                "digest": "aa5ab8afb86208e1c097028d63074f0142ce6079420ea6f68f219933361fd869",
                "model_info": {},
                "capabilities": ["completion"],
            },
            {
                "model": "llama2:7b",
                "digest": "78e26419b4469263f75331927a00a0284ef6544c1975b826b15abdaef17bb962",
                "model_info": {},
                "capabilities": ["completion"],
            },
            {
                "model": "llama2:7b-q4_K_M",
                "digest": "78e26419b4469263f75331927a00a0284ef6544c1975b826b15abdaef17bb962",
                "model_info": {},
                "capabilities": ["completion"],
            },
            {
                "model": "llama2:latest",
                "digest": "78e26419b4469263f75331927a00a0284ef6544c1975b826b15abdaef17bb962",
                "model_info": {},
                "capabilities": ["completion"],
            },
            {
                "model": "phi3:latest",
                "digest": "e2fd6321a5fe6bb3ac8a4e6f1cf04477fd2dea2924cf53237a995387e152ee9c",
                "model_info": {},
                "capabilities": ["completion"],
            },
            {
                "model": "mxbai-embed-large:latest",
                "digest": "468836162de7f81e041c43663fedbbba921dcea9b9fefea135685a39b2d83dd8",
                "model_info": {},
                "capabilities": ["embedding"],
            },
            {
                "model": "deepseek-r1:70b",
                "digest": "0c1615a8ca32ef41e433aa420558b4685f9fc7f3fd74119860a8e2e389cd7942",
                "model_info": {},
                "capabilities": ["completion", "thinking"],
            },
            {
                "model": "deepseek-r1:70b-llama-distill-q4_K_M",
                "digest": "0c1615a8ca32ef41e433aa420558b4685f9fc7f3fd74119860a8e2e389cd7942",
                "model_info": {},
                "capabilities": ["completion", "thinking"],
            },
        ],
    }
    client = mocker.patch("ollama.Client").return_value
    client.list.return_value = return_value
    client.show.side_effect = lambda name: next(
        ollama.ShowResponse(**m) for m in return_value["models"] if m["model"] == name
    )
    return client
