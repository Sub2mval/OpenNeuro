"""DiscordConnector — optional external connector for OpenNeuro.

Posts messages to a Discord channel via an incoming webhook: a single
authenticated HTTP POST, no OAuth flow and no persistent bot/gateway
connection required. Mirrors the structure of ``github_connector.py`` (same
isolated-worker fallback pattern for domain/protocol types, same "missing
credentials -> clean non-crashing failure" behavior).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Optional, Protocol

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
# HTTP client seam (dependency-injected so tests never touch the network).
# --------------------------------------------------------------------------


class DiscordHTTPClient(Protocol):
    async def post(self, url: str, *, json: dict) -> "DiscordHTTPResponse": ...

    async def aclose(self) -> None: ...


@dataclass
class DiscordHTTPResponse:
    status_code: int
    text_body: str = ""


class HttpxDiscordClient:
    """Default DiscordHTTPClient backed by httpx."""

    def __init__(self) -> None:
        import httpx  # local import: keeps httpx optional for pure-fake tests

        self._client = httpx.AsyncClient(timeout=10.0)

    async def post(self, url: str, *, json: dict) -> DiscordHTTPResponse:
        resp = await self._client.post(url, json=json)
        return DiscordHTTPResponse(status_code=resp.status_code, text_body=resp.text)

    async def aclose(self) -> None:
        await self._client.aclose()


# --------------------------------------------------------------------------
# Settings (injected, never hardcoded).
# --------------------------------------------------------------------------


@dataclass
class DiscordConnectorSettings:
    """Credentials/config for DiscordConnector, injected by the composition root.

    ``webhook_url`` being unset is a valid, supported state: the connector
    still starts and registers its action, it just answers every dispatch
    with a clear "not configured" failure.
    """

    webhook_url: Optional[str] = None

    @property
    def is_configured(self) -> bool:
        return bool(self.webhook_url)


# --------------------------------------------------------------------------
# Connector
# --------------------------------------------------------------------------

_ACTIONS = [
    {
        "name": "post_message",
        "description": "Post a message to the configured Discord channel via webhook.",
        "schema": {
            "type": "object",
            "properties": {"content": {"type": "string"}},
            "required": ["content"],
        },
        "risk": "low",
    },
]


class DiscordConnector:
    """ApplicationConnector for a single Discord channel (via webhook).

    - Registers one action (post_message).
    - No polling: incoming webhooks are one-way (post only), so there is no
      "new event" stream to pull from Discord this way.
    - Every real API call is skipped, with a clear failure message, when no
      webhook_url is configured -- the connector never crashes core runtime
      for lack of config.
    """

    def __init__(
        self,
        application_id: "ApplicationId",
        settings: DiscordConnectorSettings,
        http_client_factory: Optional[Callable[[], DiscordHTTPClient]] = None,
    ) -> None:
        self.application_id = application_id
        self._settings = settings
        self._http_client_factory = http_client_factory or (lambda: HttpxDiscordClient())
        self._gateway: Optional["ApplicationGateway"] = None
        self._http: Optional[DiscordHTTPClient] = None

    def _build_action_definitions(self) -> list:
        now = datetime.now(timezone.utc)
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

    async def start(self, gateway: "ApplicationGateway") -> None:
        self._gateway = gateway
        gateway.register_actions(self._build_action_definitions())

        if not self._settings.is_configured:
            logger.info(
                "DiscordConnector[{}]: no webhook_url configured, actions will fail fast.",
                self.application_id,
            )
            return

        self._http = self._http_client_factory()

    async def dispatch_action(self, request: "ActionRequest") -> "ActionResult":
        now = datetime.now(timezone.utc)

        if request.action_name != "post_message":
            return ActionResult(
                request_id=request.id,
                success=False,
                message=f"unknown action: {request.action_name}",
                completed_at=now,
            )

        if not self._settings.is_configured or self._http is None:
            return ActionResult(
                request_id=request.id,
                success=False,
                message="DiscordConnector not configured (missing webhook_url)",
                completed_at=now,
            )

        content = request.arguments.get("content")
        if not content:
            return ActionResult(
                request_id=request.id,
                success=False,
                message="post_message: missing required field 'content'",
                completed_at=now,
            )

        resp = await self._http.post(self._settings.webhook_url, json={"content": content})
        if resp.status_code in (200, 204):
            return ActionResult(
                request_id=request.id, success=True, message="message posted", completed_at=now
            )
        return ActionResult(
            request_id=request.id,
            success=False,
            message=f"discord webhook failed: {resp.status_code} {resp.text_body}",
            completed_at=now,
        )

    async def stop(self) -> None:
        if self._http is not None:
            await self._http.aclose()
            self._http = None
        if self._gateway is not None:
            self._gateway.unregister_actions([a["name"] for a in _ACTIONS])
            self._gateway.disconnect()
