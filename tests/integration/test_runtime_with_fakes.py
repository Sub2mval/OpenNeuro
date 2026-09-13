"""Integration tests for AgentRuntime + ApplicationGatewayImpl.

These tests use only in-test fakes for every collaborator, matching the
09_runtime.md contract: no other OpenNeuro worker's implementation is
required for this module to be fully exercised. Domain value types are
imported from ``openneuro.runtime.agent_runtime``, which resolves to the
real ``openneuro.domain.models`` once the integrator has merged worker 01's
output, or to the local structural shim otherwise -- either way these tests
only rely on the field names given in the canonical contract.

Covers the two behaviors 09_runtime.md calls out explicitly:
  * an event reaches attention (via the gateway) and shows up in the next
    built observation;
  * allowed vs. denied action flow is correctly gated before dispatch.
Also covers a timeout, since AgentRuntime is responsible for turning an
ActionExecutor timeout into a reported failure + trace event.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Optional

from openneuro.runtime.agent_runtime import (
    ActionDefinition,
    ActionRequest,
    ActionRequestId,
    ActionResult,
    AgentRuntime,
    ApplicationEvent,
    ApplicationId,
    ApplicationState,
    AttentionRequest,
    EventId,
    PolicyDecision,
    Priority,
)
from openneuro.runtime.application_gateway_impl import ApplicationGatewayImpl


# --------------------------------------------------------------------------- 
# Fakes -- one small in-memory stand-in per collaborator protocol.
# ---------------------------------------------------------------------------


class FakeActionRegistry:
    def __init__(self) -> None:
        self._actions: dict[ApplicationId, dict[str, ActionDefinition]] = {}

    def register_actions(self, application_id, actions):
        bucket = self._actions.setdefault(application_id, {})
        for action in actions:
            bucket[action.name] = action  # registering an existing name replaces it

    def unregister_actions(self, application_id, action_names):
        bucket = self._actions.get(application_id, {})
        for name in action_names:
            bucket.pop(name, None)

    def list_available_actions(self):
        out: list[ActionDefinition] = []
        for bucket in self._actions.values():
            out.extend(bucket.values())
        return out

    def get_action(self, application_id, action_name):
        return self._actions.get(application_id, {}).get(action_name)


class FakeAttentionManager:
    """Only one unresolved attention request is in flight at a time."""

    def __init__(self) -> None:
        self._current: Optional[AttentionRequest] = None

    def request_attention(self, request):
        self._current = request

    def get_pending_query(self):
        return self._current.query if self._current is not None else None


class FakePolicyEngine:
    def __init__(self, outcome: str = "allow", reason: str = "ok") -> None:
        self.outcome = outcome
        self.reason = reason
        self.seen_contexts: list = []

    def evaluate(self, context):
        self.seen_contexts.append(context)
        return PolicyDecision(
            outcome=self.outcome, reason=self.reason, decided_at=datetime.now(timezone.utc)
        )


class FakeActionExecutor:
    def __init__(self, should_timeout: bool = False) -> None:
        self.should_timeout = should_timeout
        self.calls: list[ActionRequest] = []

    async def execute(self, request, connector):
        self.calls.append(request)
        if self.should_timeout:
            raise asyncio.TimeoutError()
        return await connector.dispatch_action(request)


class FakeEventRouter:
    def __init__(self) -> None:
        self.routed_events: list[ApplicationEvent] = []
        self._states: dict[ApplicationId, ApplicationState] = {}

    def route_event(self, event):
        self.routed_events.append(event)

    def publish_state(self, state):
        self._states[state.application_id] = state

    def get_state_snapshot(self):
        return dict(self._states)


class FakeTraceRecorder:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []

    def record(self, kind, payload):
        self.events.append((kind, payload))

    def kinds(self) -> list[str]:
        return [kind for kind, _ in self.events]


class FakeConnector:
    def __init__(self, application_id: ApplicationId) -> None:
        self.application_id = application_id
        self.dispatch_calls: list[ActionRequest] = []

    async def start(self, gateway):  # pragma: no cover - unused in these tests
        pass

    async def dispatch_action(self, request):
        self.dispatch_calls.append(request)
        return ActionResult(
            request_id=request.id,
            success=True,
            message="done",
            completed_at=datetime.now(timezone.utc),
        )

    async def stop(self):  # pragma: no cover - unused in these tests
        pass


# --------------------------------------------------------------------------- 
# Helpers
# ---------------------------------------------------------------------------


def _make_runtime(policy_outcome="allow", policy_reason="ok", executor_timeout=False, app_id=None, connector=None):
    app_id = app_id or ApplicationId("demo-app")
    registry = FakeActionRegistry()
    attention = FakeAttentionManager()
    policy = FakePolicyEngine(outcome=policy_outcome, reason=policy_reason)
    executor = FakeActionExecutor(should_timeout=executor_timeout)
    router = FakeEventRouter()
    tracer = FakeTraceRecorder()
    connector = connector or FakeConnector(app_id)

    runtime = AgentRuntime(
        action_registry=registry,
        attention_manager=attention,
        policy_engine=policy,
        action_executor=executor,
        event_router=router,
        trace_recorder=tracer,
        connectors={app_id: connector},
    )
    return runtime, dict(
        app_id=app_id,
        registry=registry,
        attention=attention,
        policy=policy,
        executor=executor,
        router=router,
        tracer=tracer,
        connector=connector,
    )


# --------------------------------------------------------------------------- 
# Tests
# ---------------------------------------------------------------------------


def test_event_reaches_attention_and_observation():
    runtime, parts = _make_runtime()
    app_id = parts["app_id"]

    gateway = ApplicationGatewayImpl(
        application_id=app_id,
        action_registry=parts["registry"],
        event_router=parts["router"],
        attention_manager=parts["attention"],
    )

    evt = ApplicationEvent(
        event_id=EventId("evt-1"),
        application_id=app_id,
        message="something happened",
        silent=False,
        priority=Priority.MEDIUM,
        created_at=datetime.now(timezone.utc),
    )
    runtime.handle_application_event(evt)

    assert parts["router"].routed_events == [evt]
    assert "event_received" in parts["tracer"].kinds()

    # The application (via its gateway) escalates to attention off the back
    # of that event.
    attn_request = AttentionRequest(
        application_id=app_id,
        event_id=evt.event_id,
        priority=Priority.MEDIUM,
        state="waiting",
        query="should I respond?",
        ephemeral=True,
        candidate_actions=[],
        created_at=datetime.now(timezone.utc),
        expires_at=None,
    )
    gateway.request_attention(attn_request)

    observation = runtime.build_observation()
    assert observation.pending_query == "should I respond?"
    assert observation.state_snapshot == {}
    assert observation.available_actions == []

    # A second attention request replaces the first (only one in flight).
    gateway.request_attention(
        AttentionRequest(
            application_id=app_id,
            event_id=None,
            priority=Priority.HIGH,
            state="waiting",
            query="new, more urgent question",
            ephemeral=True,
            candidate_actions=[],
            created_at=datetime.now(timezone.utc),
            expires_at=None,
        )
    )
    assert runtime.build_observation().pending_query == "new, more urgent question"


def test_allowed_action_flow_reaches_executor_and_connector():
    runtime, parts = _make_runtime(policy_outcome="allow")
    app_id = parts["app_id"]

    action_def = ActionDefinition(
        application_id=app_id,
        name="do_thing",
        description="does a thing",
        schema={},
        risk="low",
        registered_at=datetime.now(timezone.utc),
    )
    parts["registry"].register_actions(app_id, [action_def])

    request = ActionRequest(
        id=ActionRequestId("req-1"),
        application_id=app_id,
        action_name="do_thing",
        arguments={},
        session_id="session-1",
        requested_at=datetime.now(timezone.utc),
    )

    results = asyncio.run(runtime.propose_and_execute([request]))

    assert len(results) == 1
    assert results[0].success is True
    assert parts["executor"].calls == [request]
    assert parts["connector"].dispatch_calls == [request]
    assert parts["policy"].seen_contexts[0].risk == "low"
    assert parts["policy"].seen_contexts[0].is_registered is True
    assert parts["policy"].seen_contexts[0].is_connected is True

    kinds = parts["tracer"].kinds()
    for expected in (
        "turn_started",
        "action_proposed",
        "policy_decision",
        "action_executed",
        "turn_completed",
    ):
        assert expected in kinds
    assert "action_timed_out" not in kinds


def test_denied_action_never_reaches_executor_or_connector():
    runtime, parts = _make_runtime(policy_outcome="deny", policy_reason="risk too high")
    app_id = parts["app_id"]

    action_def = ActionDefinition(
        application_id=app_id,
        name="dangerous_thing",
        description="risky",
        schema={},
        risk="high",
        registered_at=datetime.now(timezone.utc),
    )
    parts["registry"].register_actions(app_id, [action_def])

    request = ActionRequest(
        id=ActionRequestId("req-2"),
        application_id=app_id,
        action_name="dangerous_thing",
        arguments={},
        session_id="session-1",
        requested_at=datetime.now(timezone.utc),
    )

    results = asyncio.run(runtime.propose_and_execute([request]))

    assert len(results) == 1
    assert results[0].success is False
    assert "risk too high" in results[0].message
    assert parts["executor"].calls == []
    assert parts["connector"].dispatch_calls == []

    kinds = parts["tracer"].kinds()
    assert "policy_decision" in kinds
    assert "action_executed" not in kinds
    assert "action_timed_out" not in kinds


def test_require_confirmation_also_blocks_dispatch():
    runtime, parts = _make_runtime(policy_outcome="require_confirmation", policy_reason="needs human ok")
    app_id = parts["app_id"]

    request = ActionRequest(
        id=ActionRequestId("req-3"),
        application_id=app_id,
        action_name="unregistered_action",
        arguments={},
        session_id="session-1",
        requested_at=datetime.now(timezone.utc),
    )

    results = asyncio.run(runtime.propose_and_execute([request]))

    assert results[0].success is False
    assert "require_confirmation" in results[0].message
    assert parts["executor"].calls == []
    assert parts["connector"].dispatch_calls == []
    # Unregistered actions are treated as unknown/high risk and reported as
    # such in the policy context, giving the policy engine a real signal.
    assert parts["policy"].seen_contexts[0].is_registered is False
    assert parts["policy"].seen_contexts[0].risk == "high"


def test_action_timeout_is_traced_and_reported_as_failure():
    runtime, parts = _make_runtime(policy_outcome="allow", executor_timeout=True)
    app_id = parts["app_id"]

    action_def = ActionDefinition(
        application_id=app_id,
        name="slow_thing",
        description="slow",
        schema={},
        risk="low",
        registered_at=datetime.now(timezone.utc),
    )
    parts["registry"].register_actions(app_id, [action_def])

    request = ActionRequest(
        id=ActionRequestId("req-4"),
        application_id=app_id,
        action_name="slow_thing",
        arguments={},
        session_id="session-1",
        requested_at=datetime.now(timezone.utc),
    )

    results = asyncio.run(runtime.propose_and_execute([request]))

    assert len(results) == 1
    assert results[0].success is False
    assert "timed out" in results[0].message
    assert "action_timed_out" in parts["tracer"].kinds()
    assert "action_executed" not in parts["tracer"].kinds()
    assert parts["connector"].dispatch_calls == []  # executor raised before calling connector


def test_no_connector_attached_is_reported_without_touching_executor():
    app_id = ApplicationId("orphan-app")
    runtime, parts = _make_runtime(policy_outcome="allow", app_id=app_id)
    # Deliberately construct a runtime whose connector map does not include
    # this application, simulating a disconnected/never-started app.
    empty_connectors: dict[ApplicationId, object] = {}

    request = ActionRequest(
        id=ActionRequestId("req-5"),
        application_id=app_id,
        action_name="whatever",
        arguments={},
        session_id="session-1",
        requested_at=datetime.now(timezone.utc),
    )

    results = asyncio.run(runtime.propose_and_execute([request], connectors=empty_connectors))

    assert results[0].success is False
    assert "no connector attached" in results[0].message
    assert parts["executor"].calls == []
