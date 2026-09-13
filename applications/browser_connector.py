"""BrowserConnector — optional external connector for OpenNeuro.

Isolated-worker note: see the module docstring in ``github_connector.py``
for why this file carries small structural fallbacks for the canonical
domain/protocol types instead of importing production modules that other
workers own and this sandbox does not contain.

Base-repo inspection: the base Open-LLM-VTuber repository does not bundle a
browser-automation library (no Playwright/Selenium in its dependencies).
Per the task contract, this connector does not introduce a heavy browser
framework just for this task. Instead it defines a small structural
``BrowserDriver`` seam: if the composition root has a real driver available
(Playwright, a CDP wrapper, whatever the deployment provides) it can inject
it; if nothing is injected, this connector opportunistically checks whether
Playwright happens to be importable, and otherwise disables itself cleanly
-- it still registers with the gateway with zero actions and every dispatch
fails with a clear, non-crashing message.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional, Protocol

from loguru import logger

try:  # pragma: no cover - exercised only once real modules exist
    from openneuro.domain import (  # type: ignore
        ActionDefinition,
        ActionRequest,
        ActionResult,
        ApplicationId,
        Priority,
    )
except ImportError:  # pragma: no cover - isolated-worker fallback
    ApplicationId = str  # type: ignore[assignment,misc]

    class Priority(str, Enum):
        LOW = "LOW"
        MEDIUM = "MEDIUM"
        HIGH = "HIGH"
        CRITICAL = "CRITICAL"

    @dataclass
    class ActionDefinition:  # type: ignore[no-redef]
        application_id: "ApplicationId"
        name: str
        description: str
        schema: dict
        risk: str
        registered_at: datetime

    @dataclass
    class ActionRequest:  # type: ignore[no-redef]
        id: str
        application_id: "ApplicationId"
        action_name: str
        arguments: dict
        session_id: str
        requested_at: datetime

    @dataclass
    class ActionResult:  # type: ignore[no-redef]
        request_id: str
        success: bool
        message: str
        completed_at: datetime

try:  # pragma: no cover - exercised only once real modules exist
    from openneuro.applications.protocol import (  # type: ignore
        ApplicationConnector,
        ApplicationGateway,
    )
except ImportError:  # pragma: no cover - isolated-worker fallback

    class ApplicationGateway(Protocol):  # type: ignore[no-redef]
        def register_actions(self, actions: list) -> None: ...
        def unregister_actions(self, action_names: list) -> None: ...
        def publish_state(self, state: Any) -> None: ...
        def publish_event(
            self, message: str, silent: bool, priority: "Priority" = Priority.LOW
        ) -> Any: ...
        def request_attention(self, request: Any) -> None: ...
        def disconnect(self) -> None: ...

    class ApplicationConnector(Protocol):  # type: ignore[no-redef]
        application_id: "ApplicationId"

        async def start(self, gateway: "ApplicationGateway") -> None: ...
        async def dispatch_action(self, request: "ActionRequest") -> "ActionResult": ...
        async def stop(self) -> None: ...


# --------------------------------------------------------------------------
# Browser driver seam (dependency-injected; no framework required to exist).
# --------------------------------------------------------------------------


class BrowserDriver(Protocol):
    """The bounded slice of browser control OpenNeuro needs.

    Any real automation library can be adapted to this shape by the
    composition root. This connector never imports a specific browser
    framework directly except for an optional, best-effort Playwright
    auto-detection used only when nothing is injected.
    """

    async def navigate(self, url: str) -> None: ...

    async def read_text(self, selector: Optional[str] = None) -> str: ...

    async def click(self, selector: str) -> None: ...


def _playwright_available() -> bool:
    return importlib.util.find_spec("playwright") is not None


_ACTIONS = [
    {
        "name": "navigate",
        "description": "Navigate the browser to a URL.",
        "schema": {"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]},
        "risk": "medium",
    },
    {
        "name": "read_text",
        "description": "Read visible text from the current page, optionally scoped to a CSS selector.",
        "schema": {"type": "object", "properties": {"selector": {"type": "string"}}},
        "risk": "low",
    },
    {
        "name": "click",
        "description": "Click an element identified by a CSS selector.",
        "schema": {"type": "object", "properties": {"selector": {"type": "string"}}, "required": ["selector"]},
        "risk": "medium",
    },
]


class BrowserConnector:
    """ApplicationConnector wrapping a bounded set of browser actions.

    Degrades cleanly: if no ``BrowserDriver`` is injected and no supported
    automation library is importable, the connector still starts (so the
    runtime doesn't need to special-case its absence) but registers no
    actions and every dispatch returns a clear failure.
    """

    def __init__(
        self,
        application_id: "ApplicationId",
        driver: Optional[BrowserDriver] = None,
    ) -> None:
        self.application_id = application_id
        self._driver = driver
        self._gateway: Optional["ApplicationGateway"] = None
        self._enabled = driver is not None or _playwright_available()
        self._registered_names: list[str] = []

    @property
    def enabled(self) -> bool:
        return self._enabled

    async def start(self, gateway: "ApplicationGateway") -> None:
        self._gateway = gateway
        if not self._enabled:
            logger.info(
                "BrowserConnector[{}]: no browser driver available, starting "
                "disabled with zero registered actions.",
                self.application_id,
            )
            gateway.register_actions([])
            self._registered_names = []
            return

        actions = self._build_action_definitions()
        gateway.register_actions(actions)
        self._registered_names = [a["name"] for a in _ACTIONS]

    async def stop(self) -> None:
        if self._gateway is not None and self._registered_names:
            self._gateway.unregister_actions(self._registered_names)
        self._registered_names = []
        self._gateway = None

    async def dispatch_action(self, request: "ActionRequest") -> "ActionResult":
        if not self._enabled or self._driver is None:
            return ActionResult(
                request_id=request.id,
                success=False,
                message="BrowserConnector is disabled: no browser automation available.",
                completed_at=_now(),
            )
        try:
            if request.action_name == "navigate":
                await self._driver.navigate(request.arguments["url"])
                return ActionResult(request.id, True, "Navigated.", _now())
            if request.action_name == "read_text":
                text = await self._driver.read_text(request.arguments.get("selector"))
                return ActionResult(request.id, True, text[:4000], _now())
            if request.action_name == "click":
                await self._driver.click(request.arguments["selector"])
                return ActionResult(request.id, True, "Clicked.", _now())
            return ActionResult(
                request.id, False, f"Unknown browser action '{request.action_name}'.", _now()
            )
        except Exception as exc:  # never let a connector crash the runtime
            logger.warning("BrowserConnector dispatch failed: {}", exc)
            return ActionResult(request.id, False, f"Browser action failed: {exc}", _now())

    def _build_action_definitions(self) -> list:
        now = _now()
        return [
            ActionDefinition(
                application_id=self.application_id,
                name=a["name"],
                description=a["description"],
                schema=a["schema"],
                risk=a["risk"],
                registered_at=now,
            )
            for a in _ACTIONS
        ]


def _now() -> datetime:
    return datetime.now(timezone.utc)
