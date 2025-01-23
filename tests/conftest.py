import pytest
from _pytest.fixtures import SubRequest


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


@pytest.fixture(autouse=True)
def _check_ollama(request: SubRequest) -> None:
    """Automatically check Ollama server availability for integration tests.

    This fixture runs automatically for any test marked with @pytest.mark.integration.
    It implements the following logic:
        * If --no-integration specified: skip test
        * If --integration specified: fail if Ollama server unavailable
        * Otherwise: skip if Ollama server unavailable
    """
    if not request.node.get_closest_marker("integration"):
        return

    if request.config.getoption("--no-integration"):
        pytest.skip("Integration tests disabled with --no-integration")

    try:
        __import__("ollama").list()
    except Exception as e:
        if request.config.getoption("--integration"):
            raise RuntimeError(
                "--integration specified but Ollama server not available",
            ) from e
        pytest.skip("Ollama server not available")
