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
    # Separate options for vision tests, since the models capable of tools and vision are large.
    group.addoption(
        "--vision",
        action="store_true",
        help="force enable vision tests",
    )
    group.addoption(
        "--no-vision",
        action="store_true",
        help="force disable vision tests",
    )
    group.addoption(
        "--vision-model",
        action="store",
        default="llama3.2-vision:11b",
        help="specify a vision model to use for integration tests (has to be capable of vision)",
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
    config.addinivalue_line(
        "markers",
        "vision: mark test as requiring Ollama server with vision model",
    )

    if config.getoption("--integration") and config.getoption("--no-integration"):
        raise pytest.UsageError(
            "--integration and --no-integration are mutually exclusive",
        )
    if config.getoption("--vision") and config.getoption("--no-vision"):
        raise pytest.UsageError(
            "--vision and --no-vision are mutually exclusive",
        )


def pytest_sessionstart(session: pytest.Session) -> None:
    """Determine if integration tests should run.

    The decision is based on command-line options and availability of Ollama server
    and necessary model(s).
    """

    def is_enabled(model_option: str, force_on: str, force_off: str):
        model = session.config.getoption(model_option)
        enabled = False
        if not session.config.getoption(force_off):
            try:
                models = __import__("ollama").list()["models"]
                if any(model in m.model for m in models):
                    enabled = True
                elif session.config.getoption(force_on):
                    pytest.exit(
                        f"Integration tests forced but model {model} is not available",
                        1,
                    )
            except Exception as e:
                if session.config.getoption(force_on):
                    pytest.exit(
                        f"Integration tests forced but Ollama discovery failed: {e}",
                        1,
                    )
        return {
            "enabled": enabled,
            "model": model,
        }

    session.config._integration = is_enabled(
        "--model", "--integration", "--no-integration"
    )
    session.config._vision = is_enabled("--vision-model", "--vision", "--no-vision")


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
    elif not config._vision["enabled"]:
        skip_vision = pytest.mark.skip(reason="Vision tests disabled")
        for item in items:
            if "vision" in item.keywords:
                item.add_marker(skip_vision)


@pytest.fixture(scope="session")
def integration_config(pytestconfig: pytest.Config) -> dict:
    """Get integration test configuration."""
    return pytestconfig._integration


@pytest.fixture(scope="session")
def integration_model(integration_config: dict) -> str:
    """Get name of a model to use for integration tests."""
    return integration_config["model"]


@pytest.fixture(scope="session")
def vision_config(pytestconfig: pytest.Config) -> dict:
    """Get vision test configuration."""
    return pytestconfig._vision


@pytest.fixture(scope="session")
def vision_model(vision_config: dict) -> str:
    """Get name of a model to use for vision tests."""
    return vision_config["model"]


@pytest.fixture(autouse=False)
def _isolated_cache(tmp_path: Path) -> None:
    """Set isolated cache directory for each test."""
    import llm_ollama

    llm_ollama.cache.set_dir(tmp_path / "test_cache")
