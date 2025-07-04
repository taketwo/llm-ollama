from pathlib import Path

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add command-line options for integration test control.

    Creates a separate option group for integration testing with two mutually exclusive
    flags:

        --integration     Force run integration tests, fail if Ollama server unavailable
        --no-integration  Skip integration tests regardless of Ollama server status
    """
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

    The decision is based on command-line options and Ollama server availability.
    """
    if session.config.getoption("--no-integration"):
        enabled = False
    else:
        enabled = True
        try:
            __import__("ollama").list()
        except Exception as e:
            if session.config.getoption("--integration"):
                pytest.exit(
                    f"Integration tests forced but Ollama discovery failed: {e}",
                    1,
                )
            enabled = False

    session.config._integration = {"enabled": enabled}


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


@pytest.fixture(autouse=False)
def _isolated_cache(tmp_path: Path) -> None:
    """Set isolated cache directory for each test."""
    import llm_ollama

    llm_ollama.cache.set_dir(tmp_path / "test_cache")
