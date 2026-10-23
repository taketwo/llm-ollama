from dataclasses import dataclass
from pathlib import Path

import ollama
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
    ModelKind("completion", "gemma4:e2b", ("completion",)),
    ModelKind("tools", "gemma4:e2b", ("completion", "tools")),
    ModelKind("thinking", "gemma4:e2b", ("completion", "thinking")),
    ModelKind("vision", "gemma4:e2b", ("completion", "vision")),
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
            f"{kind.name}: needs a model that supports "
            f"{', '.join(kind.capabilities)} on the Ollama server",
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
    """Skip the tests of each model kind that is not enabled, with the reason.

    A kind is enabled when the Ollama server has its model and the model reports the
    kind's capabilities, unless ``--no-<kind>`` is given. With ``--<kind>``, the run
    stops if the kind cannot be enabled. Kinds without selected tests are not checked,
    so a run without integration tests does not contact the server.
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
    reasons = _find_unserved_kinds(config, kinds_to_check) if kinds_to_check else {}
    for kind, tests in tests_by_kind.items():
        if config.getoption(f"--no-{kind.name}"):
            reason = f"--no-{kind.name} given"
        elif kind in reasons:
            reason = reasons[kind]
            if config.getoption(f"--{kind.name}"):
                pytest.exit(f"--{kind.name} given but {reason}", 1)
        else:
            continue
        skip = pytest.mark.skip(reason=f"{kind.name} tests disabled: {reason}")
        for test in tests:
            test.add_marker(skip)


def _find_unserved_kinds(
    config: pytest.Config,
    kinds: list[ModelKind],
) -> dict[ModelKind, str]:
    """Map each of ``kinds`` that the Ollama server cannot serve to the reason."""
    try:
        client = get_client()
        server_names = {model["model"] for model in client.list()["models"]}
    except Exception as error:  # noqa: BLE001
        return dict.fromkeys(kinds, f"Ollama discovery failed: {error}")
    model_names = {kind: config.getoption(f"--{kind.name}-model") for kind in kinds}
    capabilities_by_model: dict[str, list[str]] = {}
    reasons_by_model: dict[str, str] = {}
    for model_name in set(model_names.values()):
        # The plugin also registers a model tagged :latest under its bare name
        if (
            model_name not in server_names
            and f"{model_name}:latest" not in server_names
        ):
            reasons_by_model[model_name] = f"{model_name} is not available"
            continue
        # Asks the server rather than the plugin's capability cache, so that a stale
        # cache fails the tests instead of skipping them
        try:
            capabilities_by_model[model_name] = (
                client.show(model_name).capabilities or []
            )
        except (ollama.ResponseError, ConnectionError) as error:
            reasons_by_model[model_name] = f"{model_name} is not available: {error}"
    reasons = {}
    for kind, model_name in model_names.items():
        if model_name in reasons_by_model:
            reasons[kind] = reasons_by_model[model_name]
            continue
        missing = [
            capability
            for capability in kind.capabilities
            if capability not in capabilities_by_model[model_name]
        ]
        if missing:
            reasons[kind] = f"{model_name} does not report {', '.join(missing)}"
    return reasons


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
