from dataclasses import dataclass
from pathlib import Path

import pytest

from llm_ollama.auth import get_client
from llm_ollama.registry import _to_model_id


@dataclass(frozen=True)
class ModelKind:
    """A kind of model that integration tests need from the Ollama server.

    Each kind gets a marker of the same name, ``--<name>`` and ``--no-<name>`` flags
    that force its tests on or off, and a ``--<name>-model`` option that picks the model.
    """

    name: str
    default_model: str
    capabilities: tuple[str, ...]  # what the model must support, in Ollama's terms


MODEL_KINDS = (
    ModelKind("chat", "qwen3:8b", ("completion", "tools", "thinking")),
    ModelKind("vision", "gemma3:latest", ("completion", "vision")),
)


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("integration", "integration tests against an Ollama server")
    for kind in MODEL_KINDS:
        group.addoption(
            f"--{kind.name}",
            action="store_true",
            help=f"force enable {kind.name} tests",
        )
        group.addoption(
            f"--no-{kind.name}",
            action="store_true",
            help=f"force disable {kind.name} tests",
        )
        group.addoption(
            f"--{kind.name}-model",
            default=kind.default_model,
            help=f"model for {kind.name} tests, which must support "
            f"{', '.join(kind.capabilities)} (default: %(default)s)",
        )


def pytest_configure(config: pytest.Config) -> None:
    """Register a marker for each model kind and reject contradictory flags."""
    for kind in MODEL_KINDS:
        config.addinivalue_line(
            "markers",
            f"{kind.name}: needs a {kind.name} model on the Ollama server",
        )
        if config.getoption(f"--{kind.name}") and config.getoption(f"--no-{kind.name}"):
            raise pytest.UsageError(
                f"--{kind.name} and --no-{kind.name} are mutually exclusive",
            )


# Last, so that -k and -m have already deselected tests
@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(
    config: pytest.Config,
    items: list[pytest.Item],
) -> None:
    """Skip the tests of each model kind that is not enabled.

    A kind is enabled when the Ollama server has its model, unless ``--no-<kind>`` is
    given. With ``--<kind>``, the run stops if the server cannot be queried or lacks the
    model. Kinds without selected tests are not checked, so a run without integration
    tests does not contact the server.
    """
    tests_by_kind = {
        kind: [item for item in items if item.get_closest_marker(kind.name)]
        for kind in MODEL_KINDS
    }
    kinds_to_check = [
        kind
        for kind, tests in tests_by_kind.items()
        if tests and not config.getoption(f"--no-{kind.name}")
    ]
    enabled = _find_served_kinds(config, kinds_to_check) if kinds_to_check else set()
    for kind, tests in tests_by_kind.items():
        if kind in enabled:
            continue
        skip = pytest.mark.skip(reason=f"{kind.name} tests disabled")
        for test in tests:
            test.add_marker(skip)


def _find_served_kinds(
    config: pytest.Config,
    kinds: list[ModelKind],
) -> set[ModelKind]:
    """Return those of ``kinds`` whose model the Ollama server has.

    Stops the run if a kind given as ``--<kind>`` is not among them.
    """
    try:
        server_names = {model["model"] for model in get_client().list()["models"]}
    except Exception as error:  # noqa: BLE001
        for kind in kinds:
            if config.getoption(f"--{kind.name}"):
                pytest.exit(
                    f"--{kind.name} given but Ollama discovery failed: {error}", 1
                )
        return set()
    served_kinds = set()
    for kind in kinds:
        model_name = config.getoption(f"--{kind.name}-model")
        # The plugin also registers a model tagged :latest under its bare name
        if model_name in server_names or f"{model_name}:latest" in server_names:
            served_kinds.add(kind)
        elif config.getoption(f"--{kind.name}"):
            pytest.exit(
                f"--{kind.name} given but model {model_name} is not available", 1
            )
    return served_kinds


@pytest.fixture
def model_id(request: pytest.FixtureRequest) -> str:
    """The ID of the model named by the ``--<kind>-model`` option of the test's kind.

    The test must carry exactly one model kind marker.
    """
    (kind,) = [
        candidate
        for candidate in MODEL_KINDS
        if request.node.get_closest_marker(candidate.name)
    ]
    return _to_model_id(request.config.getoption(f"--{kind.name}-model"))


@pytest.fixture
def box_png() -> bytes:
    """A PNG image of the word "shark" in black on white."""
    return (Path(__file__).parent / "data" / "box.png").read_bytes()
