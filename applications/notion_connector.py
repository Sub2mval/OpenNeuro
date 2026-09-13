"""NotionConnector — optional external connector for OpenNeuro.

Appends a paragraph of text to a configured Notion page via the Notion API
(``PATCH /v1/blocks/{page_id}/children``). Appending to an existing page
(rather than creating a database row) is deliberate: it needs only an
integration token and a page id, no database schema to match. Mirrors the
structure of ``github_connector.py``.
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


class NotionHTTPClient(Protocol):
    async def patch(self, url: str, *, headers: dict, json: dict) -> "NotionHTTPResponse": ...

    async def aclose(self) -> None: ...


@dataclass
class NotionHTTPResponse:
    status_code: int
    text_body: str = ""


class HttpxNotionClient:
    """Default NotionHTTPClient backed by httpx."""

    def __init__(self) -> None:
        import httpx  # local import: keeps httpx optional for pure-fake tests

        self._client = httpx.AsyncClient(timeout=10.0)

    async def patch(self, url: str, *, headers: dict, json: dict) -> NotionHTTPResponse:
        resp = await self._client.patch(url, headers=headers, json=json)
        return NotionHTTPResponse(status_code=resp.status_code, text_body=resp.text)

    async def aclose(self) -> None:
        await self._client.aclose()


# --------------------------------------------------------------------------
# Settings (injected, never hardcoded).
# --------------------------------------------------------------------------


@dataclass
class NotionConnectorSettings:
    """Credentials/config for NotionConnector, injected by the composition root.

    ``token``/``page_id`` being unset is a valid, supported state: the
    connector still starts and registers its action, it just answers every
    dispatch with a clear "not configured" failure.
    """

    token: Optional[str] = None
    page_id: Optional[str] = None
    base_url: str = "https://api.notion.com/v1"
    notion_version: str = "2022-06-28"

    @property
    def is_configured(self) -> bool:
        return bool(self.token and self.page_id)


# --------------------------------------------------------------------------
# Connector
# --------------------------------------------------------------------------

_ACTIONS = [
    {
        "name": "append_note",
        "description": "Append a paragraph of text to the configured Notion page.",
        "schema": {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
        "risk": "low",
    },
]


class NotionConnector:
    """ApplicationConnector for a single Notion page.

    - Registers one action (append_note).
    - No polling: this connector only ever writes to the configured page.
    - Every real API call is skipped, with a clear failure message, when
      credentials are absent -- the connector never crashes core runtime for
      lack of config.
    """

    def __init__(
        self,
        application_id: "ApplicationId",
        settings: NotionConnectorSettings,
        http_client_factory: Optional[Callable[[], NotionHTTPClient]] = None,
    ) -> None:
        self.application_id = application_id
        self._settings = settings
        self._http_client_factory = http_client_factory or (lambda: HttpxNotionClient())
        self._gateway: Optional["ApplicationGateway"] = None
        self._http: Optional[NotionHTTPClient] = None

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
                "NotionConnector[{}]: no token/page_id configured, actions will fail fast.",
                self.application_id,
            )
            return

        self._http = self._http_client_factory()

    async def dispatch_action(self, request: "ActionRequest") -> "ActionResult":
        now = datetime.now(timezone.utc)

        if request.action_name != "append_note":
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
                message="NotionConnector not configured (missing token/page_id)",
                completed_at=now,
            )

        text = request.arguments.get("text")
        if not text:
            return ActionResult(
                request_id=request.id,
                success=False,
                message="append_note: missing required field 'text'",
                completed_at=now,
            )

        url = f"{self._settings.base_url}/blocks/{self._settings.page_id}/children"
        headers = {
            "Authorization": f"Bearer {self._settings.token}",
            "Notion-Version": self._settings.notion_version,
            "Content-Type": "application/json",
        }
        body = {
            "children": [
                {
                    "object": "block",
                    "type": "paragraph",
                    "paragraph": {"rich_text": [{"type": "text", "text": {"content": text}}]},
                }
            ]
        }
        resp = await self._http.patch(url, headers=headers, json=body)
        if resp.status_code == 200:
            return ActionResult(
                request_id=request.id, success=True, message="note appended", completed_at=now
            )
        return ActionResult(
            request_id=request.id,
            success=False,
            message=f"notion API call failed: {resp.status_code} {resp.text_body}",
            completed_at=now,
        )

    async def stop(self) -> None:
        if self._http is not None:
            await self._http.aclose()
            self._http = None
        if self._gateway is not None:
            self._gateway.unregister_actions([a["name"] for a in _ACTIONS])
            self._gateway.disconnect()
