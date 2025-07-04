from pathlib import Path

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add command-line options for integration test control."""
    group = parser.getgroup("integration")
    group.addoption(
        "--integration",
        action="store_true",
        help="force enable integration tests",
    )
    group.addoption(
        "--no-integration",
        action="store_true",
        help="force disable integration tests",
    )
    group.addoption(
        "--model",
        action="store",
        default="llama3.2",
        help="specify a model to use for integration tests (has to be capable of completion and tool usage)",
    )


def pytest_configure(config: pytest.Config) -> None:
    """Configure integration testing setup.

    Performs two tasks:
    1. Registers the 'integration' marker to avoid pytest warnings about unknown markers
    2. Validates CLI options, ensuring --integration and --no-integration are not used
       together
    """
    config.addinivalue_line(
        "markers",
        "integration: mark test as requiring Ollama server",
    )

    if config.getoption("--integration") and config.getoption("--no-integration"):
        raise pytest.UsageError(
            "--integration and --no-integration are mutually exclusive",
        )


def pytest_sessionstart(session: pytest.Session) -> None:
    """Determine if integration tests should run.

    The decision is based on command-line options and availability of Ollama server
    and necessary model(s).
    """
    model = session.config.getoption("--model")
    enabled = False
    if not session.config.getoption("--no-integration"):
        try:
            models = __import__("ollama").list()["models"]
            if any(model in m.model for m in models):
                enabled = True
            elif session.config.getoption("--integration"):
                pytest.exit(
                    f"Integration tests forced but model {model} is not available",
                    1,
                )
        except Exception as e:
            if session.config.getoption("--integration"):
                pytest.exit(
                    f"Integration tests forced but Ollama discovery failed: {e}",
                    1,
                )

    session.config._integration = {
        "enabled": enabled,
        "model": model,
    }


def pytest_collection_modifyitems(
    config: pytest.Config,
    items: list[pytest.Item],
) -> None:
    """Skip integration tests if integration is disabled."""
    if not config._integration["enabled"]:
        skip_integration = pytest.mark.skip(reason="Integration tests disabled")
        for item in items:
            if "integration" in item.keywords:
                item.add_marker(skip_integration)


@pytest.fixture(scope="session")
def integration_config(pytestconfig: pytest.Config) -> dict:
    """Get integration test configuration."""
    return pytestconfig._integration


@pytest.fixture(scope="session")
def integration_model(integration_config: dict) -> str:
    """Get name of a model to use for integration tests."""
    return integration_config["model"]


@pytest.fixture(autouse=False)
def _isolated_cache(tmp_path: Path) -> None:
    """Set isolated cache directory for each test."""
    import llm_ollama

    llm_ollama.cache.set_dir(tmp_path / "test_cache")
