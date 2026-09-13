"""GitHubConnector — optional external connector for OpenNeuro.

Isolated-worker note
---------------------
This file is produced by worker 12 in a sandbox that does not contain the
other OpenNeuro workers' production modules (domain models, the
ApplicationConnector/ApplicationGateway protocol, etc.). To stay usable on
its own while still conforming exactly to the canonical contract, it tries
to import the real canonical modules first and only falls back to minimal
structural stand-ins when they are absent. The final integrator should
delete the fallback branch once the real modules are merged in -- nothing
here is a competing production implementation of another worker's module,
it exists solely so this file can be imported and unit-tested in isolation.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Awaitable, Callable, Optional, Protocol

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


class GitHubHTTPClient(Protocol):
    """Minimal async HTTP surface GitHubConnector needs.

    A thin wrapper is used instead of importing httpx directly everywhere so
    tests can inject a fake without any network dependency.
    """

    async def get(self, url: str, *, headers: dict, params: Optional[dict] = None) -> "GitHubHTTPResponse": ...

    async def post(self, url: str, *, headers: dict, json: dict) -> "GitHubHTTPResponse": ...

    async def aclose(self) -> None: ...


@dataclass
class GitHubHTTPResponse:
    status_code: int
    json_body: Any = None
    text_body: str = ""


class HttpxGitHubClient:
    """Default GitHubHTTPClient backed by httpx (already an OLLV dependency)."""

    def __init__(self) -> None:
        import httpx  # local import: keeps httpx optional for pure-fake tests

        self._client = httpx.AsyncClient(timeout=10.0)

    async def get(self, url: str, *, headers: dict, params: Optional[dict] = None) -> GitHubHTTPResponse:
        resp = await self._client.get(url, headers=headers, params=params)
        try:
            body = resp.json()
        except Exception:
            body = None
        return GitHubHTTPResponse(status_code=resp.status_code, json_body=body, text_body=resp.text)

    async def post(self, url: str, *, headers: dict, json: dict) -> GitHubHTTPResponse:
        resp = await self._client.post(url, headers=headers, json=json)
        try:
            body = resp.json()
        except Exception:
            body = None
        return GitHubHTTPResponse(status_code=resp.status_code, json_body=body, text_body=resp.text)

    async def aclose(self) -> None:
        await self._client.aclose()


# --------------------------------------------------------------------------
# Settings (injected, never hardcoded).
# --------------------------------------------------------------------------


@dataclass
class GitHubConnectorSettings:
    """Credentials/config for GitHubConnector, injected by the composition root.

    ``token`` and ``repo`` being unset is a valid, supported state: the
    connector still starts and registers its actions, it just answers every
    dispatch with a clear "not configured" failure and skips polling.
    """

    token: Optional[str] = None
    repo: Optional[str] = None  # "owner/name"
    base_url: str = "https://api.github.com"
    poll_interval_seconds: float = 30.0

    @property
    def is_configured(self) -> bool:
        return bool(self.token and self.repo)


# --------------------------------------------------------------------------
# Connector
# --------------------------------------------------------------------------

_ACTIONS = [
    {
        "name": "comment_on_issue",
        "description": "Post a comment on a GitHub issue or pull request.",
        "schema": {
            "type": "object",
            "properties": {
                "issue_number": {"type": "integer"},
                "body": {"type": "string"},
            },
            "required": ["issue_number", "body"],
        },
        "risk": "medium",
    },
    {
        "name": "add_label",
        "description": "Add a label to a GitHub issue or pull request.",
        "schema": {
            "type": "object",
            "properties": {
                "issue_number": {"type": "integer"},
                "label": {"type": "string"},
            },
            "required": ["issue_number", "label"],
        },
        "risk": "low",
    },
]


class GitHubConnector:
    """ApplicationConnector for a single GitHub repository.

    - Registers a very small action set (comment_on_issue, add_label).
    - Polls the issues endpoint on an interval for new/updated issues
      instead of running a webhook server (simpler for a 4h MVP, no
      inbound networking needed).
    - Every real API call is skipped, with a clear failure message,
      when credentials are absent -- the connector never crashes core
      runtime for lack of config.
    """

    def __init__(
        self,
        application_id: "ApplicationId",
        settings: GitHubConnectorSettings,
        http_client_factory: Optional[Callable[[], GitHubHTTPClient]] = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.application_id = application_id
        self._settings = settings
        self._http_client_factory = http_client_factory or (lambda: HttpxGitHubClient())
        self._clock = clock

        self._gateway: Optional["ApplicationGateway"] = None
        self._http: Optional[GitHubHTTPClient] = None
        self._poll_task: Optional[asyncio.Task] = None
        self._seen_updated_at: dict[int, str] = {}
        self._stopped = asyncio.Event()

    async def start(self, gateway: "ApplicationGateway") -> None:
        self._gateway = gateway
        gateway.register_actions(self._build_action_definitions())

        if not self._settings.is_configured:
            logger.info(
                "GitHubConnector[{}]: no token/repo configured, actions will "
                "fail fast and polling is disabled.",
                self.application_id,
            )
            return

        self._http = self._http_client_factory()
        self._stopped.clear()
        self._poll_task = asyncio.create_task(self._poll_loop())

    async def stop(self) -> None:
        self._stopped.set()
        if self._poll_task is not None:
            self._poll_task.cancel()
            try:
                await self._poll_task
            except (asyncio.CancelledError, Exception):
                pass
            self._poll_task = None
        if self._http is not None:
            await self._http.aclose()
            self._http = None
        if self._gateway is not None:
            self._gateway.unregister_actions([a["name"] for a in _ACTIONS])

    async def dispatch_action(self, request: "ActionRequest") -> "ActionResult":
        if not self._settings.is_configured or self._http is None:
            return ActionResult(
                request_id=request.id,
                success=False,
                message="GitHubConnector is not configured (missing token/repo).",
                completed_at=_now(),
            )
        try:
            if request.action_name == "comment_on_issue":
                return await self._comment_on_issue(request)
            if request.action_name == "add_label":
                return await self._add_label(request)
            return ActionResult(
                request_id=request.id,
                success=False,
                message=f"Unknown GitHub action '{request.action_name}'.",
                completed_at=_now(),
            )
        except Exception as exc:  # never let a connector crash the runtime
            logger.warning("GitHubConnector dispatch failed: {}", exc)
            return ActionResult(
                request_id=request.id,
                success=False,
                message=f"GitHub API call failed: {exc}",
                completed_at=_now(),
            )

    # -- action implementations -------------------------------------------------

    async def _comment_on_issue(self, request: "ActionRequest") -> "ActionResult":
        issue_number = request.arguments["issue_number"]
        body = request.arguments["body"]
        url = f"{self._settings.base_url}/repos/{self._settings.repo}/issues/{issue_number}/comments"
        resp = await self._http.post(url, headers=self._headers(), json={"body": body})
        ok = 200 <= resp.status_code < 300
        return ActionResult(
            request_id=request.id,
            success=ok,
            message="Comment posted." if ok else f"GitHub API returned {resp.status_code}.",
            completed_at=_now(),
        )

    async def _add_label(self, request: "ActionRequest") -> "ActionResult":
        issue_number = request.arguments["issue_number"]
        label = request.arguments["label"]
        url = f"{self._settings.base_url}/repos/{self._settings.repo}/issues/{issue_number}/labels"
        resp = await self._http.post(url, headers=self._headers(), json={"labels": [label]})
        ok = 200 <= resp.status_code < 300
        return ActionResult(
            request_id=request.id,
            success=ok,
            message="Label added." if ok else f"GitHub API returned {resp.status_code}.",
            completed_at=_now(),
        )

    # -- polling ------------------------------------------------------------

    async def _poll_loop(self) -> None:
        assert self._http is not None
        url = f"{self._settings.base_url}/repos/{self._settings.repo}/issues"
        while not self._stopped.is_set():
            try:
                resp = await self._http.get(
                    url, headers=self._headers(), params={"state": "open", "sort": "updated"}
                )
                if resp.status_code == 200 and isinstance(resp.json_body, list):
                    for issue in resp.json_body:
                        self._maybe_publish(issue)
            except Exception as exc:
                logger.warning("GitHubConnector poll failed: {}", exc)

            try:
                await asyncio.wait_for(self._stopped.wait(), timeout=self._settings.poll_interval_seconds)
            except asyncio.TimeoutError:
                pass  # normal: just means it's time to poll again

    def _maybe_publish(self, issue: dict) -> None:
        number = issue.get("number")
        updated_at = issue.get("updated_at")
        if number is None or updated_at is None:
            return
        if self._seen_updated_at.get(number) == updated_at:
            return
        self._seen_updated_at[number] = updated_at
        title = issue.get("title", f"issue #{number}")
        if self._gateway is not None:
            self._gateway.publish_event(
                message=f"GitHub issue #{number} updated: {title}",
                silent=True,
                priority=Priority.LOW,
            )

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self._settings.token}",
            "Accept": "application/vnd.github+json",
        }

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
