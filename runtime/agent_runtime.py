"""AgentRuntime: composition-only orchestrator for the OpenNeuro agent loop.

Isolated-worker note (see 09_runtime.md / INDEX.md): this file must be
importable and testable in a sandbox that contains only the base
Open-LLM-VTuber repository plus this prompt. It never imports another
OpenNeuro worker's *business logic* module (registry/policy/attention/
executor/event-router implementations) -- those are expressed purely as
``typing.Protocol`` so any compatible implementation, real or fake, can be
injected.

Domain value types (ApplicationEvent, ActionRequest, ...) are a different
case: they are pure-data contracts owned by worker 01 and are needed here
to *construct* return values (e.g. ``AgentObservation``), not just to type
hints. We therefore try to import the real ``openneuro.domain.models``
module first, and only fall back to a local structural shim -- built
verbatim from the canonical contract in 09_runtime.md -- if that module is
absent from this sandbox. This is not a duplicate business-logic
implementation; it is inert data containers, and the real module always
wins when both are present. Flagged again in the integration report below.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any, Mapping, Optional, Protocol, runtime_checkable

try:  # pragma: no cover - exercised only once worker 01's output exists
    from openneuro.domain.models import (  # type: ignore
        ActionDefinition,
        ActionRequest,
        ActionRequestId,
        ActionResult,
        AgentObservation,
        ApplicationEvent,
        ApplicationId,
        ApplicationState,
        AttentionRequest,
        EventId,
        Priority,
        PolicyContext,
        PolicyDecision,
    )
except ImportError:  # pragma: no cover - fallback path used in isolation
    from dataclasses import dataclass, field
    from enum import Enum
    from typing import NewType

    ApplicationId = NewType("ApplicationId", str)
    EventId = NewType("EventId", str)
    ActionRequestId = NewType("ActionRequestId", str)

    class Priority(str, Enum):
        LOW = "low"
        MEDIUM = "medium"
        HIGH = "high"
        CRITICAL = "critical"

    @dataclass
    class ApplicationState:
        application_id: "ApplicationId"
        data: dict[str, Any]
        version: int
        updated_at: datetime

    @dataclass
    class ApplicationEvent:
        event_id: "EventId"
        application_id: "ApplicationId"
        message: str
        silent: bool
        priority: "Priority"
        created_at: datetime

    @dataclass
    class ActionDefinition:
        application_id: "ApplicationId"
        name: str
        description: str
        schema: dict[str, Any]
        risk: str  # Literal["low", "medium", "high"]
        registered_at: datetime

    @dataclass
    class ActionRequest:
        id: "ActionRequestId"
        application_id: "ApplicationId"
        action_name: str
        arguments: dict[str, Any]
        session_id: Any
        requested_at: datetime

    @dataclass
    class ActionResult:
        request_id: "ActionRequestId"
        success: bool
        message: str
        completed_at: datetime

    @dataclass
    class AttentionRequest:
        application_id: "ApplicationId"
        event_id: Optional["EventId"]
        priority: "Priority"
        state: str
        query: str
        ephemeral: bool
        candidate_actions: list[str]
        created_at: datetime
        expires_at: Optional[datetime]

    @dataclass
    class AgentObservation:
        state_snapshot: dict["ApplicationId", "ApplicationState"]
        pending_query: Optional[str]
        available_actions: list["ActionDefinition"]

    @dataclass
    class PolicyContext:
        """Assumption: 'registration/connection facts' == is_registered/is_connected.

        The canonical contract only says PolicyContext "must contain enough
        information for deterministic authorization, including ... and
        registration/connection facts as defined by the implementation." In
        the absence of worker 03's concrete definition, this shim spells that
        out as two booleans. The real PolicyContext (once integrated) simply
        replaces this shim; AgentRuntime only ever constructs it by keyword,
        so field additions on the real type are non-breaking.
        """

        application_id: "ApplicationId"
        action_name: str
        arguments: dict[str, Any]
        risk: str
        is_registered: bool
        is_connected: bool

    @dataclass
    class PolicyDecision:
        outcome: str  # Literal["allow", "deny", "require_confirmation"]
        reason: str
        decided_at: datetime


# ---------------------------------------------------------------------------
# Structural (Protocol) contracts for injected collaborators and for the
# application-side connector. These are intentionally narrow: only the
# methods AgentRuntime / ApplicationGatewayImpl actually call. Any concrete
# implementation from another worker -- or a test fake -- that has this
# shape is accepted; nothing here special-cases a particular application or
# imports another worker's implementation module.
# ---------------------------------------------------------------------------


@runtime_checkable
class ActionRegistryProtocol(Protocol):
    def list_available_actions(self) -> list["ActionDefinition"]: ...

    def get_action(
        self, application_id: "ApplicationId", action_name: str
    ) -> Optional["ActionDefinition"]: ...

    def register_actions(
        self, application_id: "ApplicationId", actions: list["ActionDefinition"]
    ) -> None: ...

    def unregister_actions(
        self, application_id: "ApplicationId", action_names: list[str]
    ) -> None: ...


@runtime_checkable
class AttentionManagerProtocol(Protocol):
    def request_attention(self, request: "AttentionRequest") -> None: ...

    def get_pending_query(self) -> Optional[str]: ...


@runtime_checkable
class PolicyEngineProtocol(Protocol):
    def evaluate(self, context: "PolicyContext") -> "PolicyDecision": ...


@runtime_checkable
class ApplicationConnectorProtocol(Protocol):
    application_id: "ApplicationId"

    async def start(self, gateway: "ApplicationGatewayProtocol") -> None: ...

    async def dispatch_action(self, request: "ActionRequest") -> "ActionResult": ...

    async def stop(self) -> None: ...


@runtime_checkable
class ApplicationGatewayProtocol(Protocol):
    def register_actions(self, actions: list["ActionDefinition"]) -> None: ...

    def unregister_actions(self, action_names: list[str]) -> None: ...

    def publish_state(self, state: "ApplicationState") -> None: ...

    def publish_event(
        self, message: str, silent: bool, priority: "Priority" = Priority.LOW
    ) -> "EventId": ...

    def request_attention(self, request: "AttentionRequest") -> None: ...

    def disconnect(self) -> None: ...


@runtime_checkable
class ActionExecutorProtocol(Protocol):
    async def execute(
        self, request: "ActionRequest", connector: "ApplicationConnectorProtocol"
    ) -> "ActionResult": ...


@runtime_checkable
class EventRouterProtocol(Protocol):
    def route_event(self, event: "ApplicationEvent") -> None: ...

    def publish_state(self, state: "ApplicationState") -> None: ...

    def get_state_snapshot(self) -> dict["ApplicationId", "ApplicationState"]: ...


@runtime_checkable
class TraceRecorderProtocol(Protocol):
    def record(self, kind: str, payload: dict[str, Any]) -> None: ...


ConnectorRepository = Mapping["ApplicationId", "ApplicationConnectorProtocol"]


class AgentRuntime:
    """Composition root for one agent loop tick.

    Holds no business logic of its own: policy, attention, action
    execution, and event/state handling all live in the injected
    collaborators. This class only wires calls together in the order
    required by 09_runtime.md and emits trace events. It never imports
    ``openneuro.applications.*`` and never branches on an application id.
    """

    def __init__(
        self,
        action_registry: ActionRegistryProtocol,
        attention_manager: AttentionManagerProtocol,
        policy_engine: PolicyEngineProtocol,
        action_executor: ActionExecutorProtocol,
        event_router: EventRouterProtocol,
        trace_recorder: TraceRecorderProtocol,
        connectors: ConnectorRepository,
    ) -> None:
        self._action_registry = action_registry
        self._attention_manager = attention_manager
        self._policy_engine = policy_engine
        self._action_executor = action_executor
        self._event_router = event_router
        self._trace_recorder = trace_recorder
        self._connectors = connectors

    # ------------------------------------------------------------- events

    def handle_application_event(self, evt: "ApplicationEvent") -> None:
        """Trace an inbound application event, then delegate routing."""
        self._trace(
            "event_received",
            {
                "application_id": str(evt.application_id),
                "event_id": str(evt.event_id),
                "priority": getattr(evt.priority, "value", evt.priority),
                "silent": evt.silent,
            },
        )
        self._event_router.route_event(evt)

    # -------------------------------------------------------- observation

    def build_observation(self) -> "AgentObservation":
        """Assemble the agent's current view of the world."""
        return AgentObservation(
            state_snapshot=self._event_router.get_state_snapshot(),
            pending_query=self._attention_manager.get_pending_query(),
            available_actions=self._action_registry.list_available_actions(),
        )

    # ------------------------------------------------- action turn (core)

    async def propose_and_execute(
        self,
        action_requests: list["ActionRequest"],
        connectors: Optional[ConnectorRepository] = None,
    ) -> list["ActionResult"]:
        """Evaluate every proposed action against policy, then dispatch.

        Denied (or not-yet-confirmed) actions are turned into a failed
        ActionResult and never reach ``ActionExecutor``/the connector.
        Allowed actions are handed to the injected ``ActionExecutor``,
        which owns the 20s default timeout; if it signals a timeout via
        ``TimeoutError`` this method synthesizes a failed result and traces
        ``action_timed_out`` instead of ``action_executed``.

        ``connectors`` may be supplied per-call (as named in the 09_runtime
        contract text); if omitted, the repository injected at construction
        time is used.
        """
        active_connectors = connectors if connectors is not None else self._connectors
        self._trace("turn_started", {"request_count": len(action_requests)})

        results: list["ActionResult"] = []
        for request in action_requests:
            self._trace(
                "action_proposed",
                {
                    "application_id": str(request.application_id),
                    "action_name": request.action_name,
                    "request_id": str(request.id),
                },
            )

            action_def = self._action_registry.get_action(
                request.application_id, request.action_name
            )
            connector = active_connectors.get(request.application_id)

            context = PolicyContext(
                application_id=request.application_id,
                action_name=request.action_name,
                arguments=request.arguments,
                risk=action_def.risk if action_def is not None else "high",
                is_registered=action_def is not None,
                is_connected=connector is not None,
            )
            decision = self._policy_engine.evaluate(context)
            self._trace(
                "policy_decision",
                {
                    "request_id": str(request.id),
                    "outcome": decision.outcome,
                    "reason": decision.reason,
                },
            )

            if decision.outcome != "allow":
                results.append(
                    ActionResult(
                        request_id=request.id,
                        success=False,
                        message=f"{decision.outcome}: {decision.reason}",
                        completed_at=datetime.now(timezone.utc),
                    )
                )
                continue

            if connector is None:
                results.append(
                    ActionResult(
                        request_id=request.id,
                        success=False,
                        message=f"no connector attached for application {request.application_id!r}",
                        completed_at=datetime.now(timezone.utc),
                    )
                )
                continue

            try:
                result = await self._action_executor.execute(request, connector)
                self._trace(
                    "action_executed",
                    {"request_id": str(request.id), "success": result.success},
                )
            except (asyncio.TimeoutError, TimeoutError):
                result = ActionResult(
                    request_id=request.id,
                    success=False,
                    message="action timed out",
                    completed_at=datetime.now(timezone.utc),
                )
                self._trace("action_timed_out", {"request_id": str(request.id)})

            results.append(result)

        self._trace("turn_completed", {"result_count": len(results)})
        return results

    # --------------------------------------------------------------- util

    def _trace(self, kind: str, payload: dict[str, Any]) -> None:
        self._trace_recorder.record(kind, payload)
