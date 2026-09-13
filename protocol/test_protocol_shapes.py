"""Structural tests for openneuro.protocol.

These only assert *shape*: that a plain object satisfying the described
methods/attributes is recognized as an ApplicationConnector / gateway
without subclassing anything from openneuro.protocol, and that the
protocol modules themselves stay free of I/O, application-name literals,
and dependencies on runtime/applications/agent_adapter/embodiment.

Domain types (ApplicationId, ActionRequest, etc.) are not assumed to exist
in this sandbox; local stubs stand in for them wherever a runtime value is
actually needed.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

from openneuro.protocol import ApplicationConnector, ApplicationGateway
from openneuro.protocol.gateway import _Priority


# ---------------------------------------------------------------------------
# Local stubs standing in for absent domain types.
# ---------------------------------------------------------------------------


class _FakeActionRequest:
    def __init__(self, request_id: str = "req-1") -> None:
        self.id = request_id
        self.application_id = "app-1"
        self.action_name = "noop"
        self.arguments: dict = {}
        self.session_id = "sess-1"


class _FakeActionResult:
    def __init__(self, request_id: str, success: bool = True) -> None:
        self.request_id = request_id
        self.success = success
        self.message = "ok"


class _FakeConnector:
    """Satisfies ApplicationConnector purely by shape, no inheritance."""

    def __init__(self) -> None:
        self.application_id = "app-1"

    async def start(self, gateway: "ApplicationGateway") -> None:
        return None

    async def dispatch_action(self, request: _FakeActionRequest) -> _FakeActionResult:
        return _FakeActionResult(request.id)

    async def stop(self) -> None:
        return None


class _FakeGateway:
    """Satisfies ApplicationGateway purely by shape, no inheritance."""

    def register_actions(self, actions: list) -> None:
        return None

    def unregister_actions(self, action_names: list) -> None:
        return None

    def publish_state(self, state) -> None:
        return None

    def publish_event(self, message: str, silent: bool, priority=_Priority.LOW) -> str:
        return "event-1"

    def request_attention(self, request) -> None:
        return None

    def disconnect(self) -> None:
        return None


class _NotAConnector:
    """Missing dispatch_action / stop — should NOT satisfy the protocol."""

    def __init__(self) -> None:
        self.application_id = "app-1"

    async def start(self, gateway) -> None:
        return None


class _NotAGateway:
    """Missing several required methods — should NOT satisfy the protocol."""

    def publish_event(self, message: str, silent: bool, priority=None) -> str:
        return "event-1"


# ---------------------------------------------------------------------------
# Structural typing tests.
# ---------------------------------------------------------------------------


def test_fake_connector_satisfies_protocol_without_subclassing():
    connector = _FakeConnector()
    assert isinstance(connector, ApplicationConnector)
    assert _FakeConnector.__mro__[-1] is object  # no Protocol in its bases


def test_fake_gateway_satisfies_protocol_without_subclassing():
    gateway = _FakeGateway()
    assert isinstance(gateway, ApplicationGateway)
    assert _FakeGateway.__mro__[-1] is object


def test_incomplete_object_does_not_satisfy_connector_protocol():
    assert not isinstance(_NotAConnector(), ApplicationConnector)


def test_incomplete_object_does_not_satisfy_gateway_protocol():
    assert not isinstance(_NotAGateway(), ApplicationGateway)


@pytest.mark.asyncio
async def test_fake_connector_and_gateway_interact_end_to_end():
    connector = _FakeConnector()
    gateway = _FakeGateway()

    await connector.start(gateway)
    request = _FakeActionRequest(request_id="req-42")
    result = await connector.dispatch_action(request)
    assert result.request_id == "req-42"
    assert result.success is True

    event_id = gateway.publish_event("hello", silent=False)
    assert event_id == "event-1"

    await connector.stop()


def test_gateway_default_priority_is_low():
    import inspect as _inspect

    sig = _inspect.signature(_FakeGateway.publish_event)
    assert sig.parameters["priority"].default is _Priority.LOW


# ---------------------------------------------------------------------------
# Contract-hygiene tests: no I/O, no app-name literals, no forbidden imports.
# ---------------------------------------------------------------------------

_PROTOCOL_DIR = Path(__file__).resolve().parents[2] / "openneuro" / "protocol"
_FORBIDDEN_MODULE_PREFIXES = (
    "openneuro.runtime",
    "openneuro.applications",
    "openneuro.agent_adapter",
    "openneuro.embodiment",
)
_FORBIDDEN_IO_MODULES = {"socket", "requests", "httpx", "aiohttp", "urllib"}
_APPLICATION_NAME_LITERALS = {"github", "discord", "browser", "simulation"}


def _source_files():
    return sorted(_PROTOCOL_DIR.glob("*.py"))


@pytest.mark.parametrize("path", _source_files(), ids=lambda p: p.name)
def test_module_has_no_forbidden_imports(path: Path):
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module] if node.module else []
        else:
            continue
        for name in names:
            if not name:
                continue
            assert not name.startswith(_FORBIDDEN_MODULE_PREFIXES), (
                f"{path.name} imports forbidden module {name}"
            )
            assert name.split(".")[0] not in _FORBIDDEN_IO_MODULES, (
                f"{path.name} imports I/O module {name}"
            )


@pytest.mark.parametrize("path", _source_files(), ids=lambda p: p.name)
def test_module_has_no_application_name_literals(path: Path):
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            lowered = node.value.lower()
            for literal in _APPLICATION_NAME_LITERALS:
                assert literal not in lowered, (
                    f"{path.name} contains application-name literal {literal!r}"
                )


def test_connector_module_has_no_concrete_gateway_or_applications_module():
    # Guards the "do NOT implement a concrete gateway / applications module
    # here" requirement at the file-layout level.
    names = {p.name for p in _source_files()}
    assert names == {"__init__.py", "connector.py", "gateway.py"}
