"""Unit tests for StateStore and EventRouter.

These tests use small local stubs for canonical domain types
(ApplicationState, ApplicationEvent, Priority, AttentionRequest) instead
of importing `openneuro.domain`, per the isolated-worker contract: other
OpenNeuro workers' modules are not assumed to exist in this sandbox.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

import pytest

from openneuro.state.event_router import EventRouter
from openneuro.state.state_store import StateStore


# ---------------------------------------------------------------------------
# Local stubs standing in for canonical domain types (owned by other
# OpenNeuro workers: 01_domain_models.md, 04_attention.md).
# ---------------------------------------------------------------------------


class Priority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass(frozen=True)
class ApplicationState:
    application_id: str
    data: Dict[str, Any]
    version: int
    updated_at: datetime


@dataclass(frozen=True)
class ApplicationEvent:
    event_id: str
    application_id: str
    message: str
    silent: bool
    priority: Priority
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass(frozen=True)
class AttentionRequest:
    application_id: str
    event_id: Optional[str]
    priority: Priority
    state: str
    query: str
    ephemeral: bool
    candidate_actions: List[str]
    created_at: datetime
    expires_at: Optional[datetime]


class FakeAttentionManager:
    """Records submitted AttentionRequests; does not implement replacement
    semantics, since that behavior belongs to the real AttentionManager."""

    def __init__(self) -> None:
        self.submitted: List[AttentionRequest] = []

    def submit(self, request: AttentionRequest) -> None:
        self.submitted.append(request)


def make_attention_request_factory():
    def factory(evt: ApplicationEvent) -> AttentionRequest:
        return AttentionRequest(
            application_id=evt.application_id,
            event_id=evt.event_id,
            priority=evt.priority,
            state="event_received",
            query=evt.message,
            ephemeral=False,
            candidate_actions=[],
            created_at=evt.created_at,
            expires_at=None,
        )

    return factory


def make_state(app_id="app-1", data=None, version=1) -> ApplicationState:
    return ApplicationState(
        application_id=app_id,
        data=data if data is not None else {"k": "v"},
        version=version,
        updated_at=datetime.now(timezone.utc),
    )


def make_event(app_id="app-1", silent=False, priority=Priority.LOW, message="hi", event_id="evt-1") -> ApplicationEvent:
    return ApplicationEvent(
        event_id=event_id,
        application_id=app_id,
        message=message,
        silent=silent,
        priority=priority,
    )


# ---------------------------------------------------------------------------
# StateStore tests
# ---------------------------------------------------------------------------


def test_get_missing_application_returns_none():
    store = StateStore()
    assert store.get("does-not-exist") is None


def test_upsert_then_get_returns_stored_state():
    store = StateStore()
    state = make_state()
    store.upsert(state)
    assert store.get("app-1") is state


def test_upsert_replaces_wholesale_not_merged():
    store = StateStore()
    store.upsert(make_state(data={"a": 1, "b": 2}, version=1))
    new_state = make_state(data={"c": 3}, version=2)
    store.upsert(new_state)

    result = store.get("app-1")
    assert result is new_state
    assert result.data == {"c": 3}  # old keys 'a'/'b' are gone, not merged
    assert result.version == 2


def test_all_returns_snapshot_copy_not_live_view():
    store = StateStore()
    store.upsert(make_state(app_id="app-1"))

    snapshot = store.all()
    assert snapshot == {"app-1": store.get("app-1")}

    # Mutating the returned mapping must not affect internal storage.
    snapshot["app-2"] = make_state(app_id="app-2")
    del snapshot["app-1"]

    assert store.get("app-1") is not None
    assert store.get("app-2") is None


def test_all_reflects_multiple_applications():
    store = StateStore()
    store.upsert(make_state(app_id="app-1"))
    store.upsert(make_state(app_id="app-2"))

    all_states = store.all()
    assert set(all_states.keys()) == {"app-1", "app-2"}


# ---------------------------------------------------------------------------
# EventRouter tests
# ---------------------------------------------------------------------------


def make_router(silent_buffer_size=20):
    store = StateStore()
    attention_manager = FakeAttentionManager()
    router = EventRouter(
        state_store=store,
        attention_manager=attention_manager,
        attention_request_factory=make_attention_request_factory(),
        silent_buffer_size=silent_buffer_size,
    )
    return router, store, attention_manager


def test_on_state_forwards_to_state_store():
    router, store, _ = make_router()
    state = make_state()

    router.on_state(state)

    assert store.get("app-1") is state


def test_silent_event_is_buffered_and_not_sent_to_attention():
    router, _, attention_manager = make_router()
    evt = make_event(silent=True, message="quietly logged")

    router.on_event(evt)

    assert attention_manager.submitted == []
    assert router.silent_context() == [evt]


def test_non_silent_event_submits_attention_request():
    router, _, attention_manager = make_router()
    evt = make_event(silent=False, priority=Priority.HIGH, message="need input")

    router.on_event(evt)

    assert len(attention_manager.submitted) == 1
    request = attention_manager.submitted[0]
    assert request.application_id == "app-1"
    assert request.priority == Priority.HIGH
    assert request.query == "need input"
    assert request.event_id == evt.event_id


def test_non_silent_event_is_not_buffered_as_silent_context():
    router, _, _ = make_router()
    router.on_event(make_event(silent=False))
    assert router.silent_context() == []


def test_silent_buffer_is_bounded_and_drops_oldest():
    router, _, _ = make_router(silent_buffer_size=2)

    e1 = make_event(silent=True, event_id="e1")
    e2 = make_event(silent=True, event_id="e2")
    e3 = make_event(silent=True, event_id="e3")

    router.on_event(e1)
    router.on_event(e2)
    router.on_event(e3)

    buffered = router.silent_context()
    assert len(buffered) == 2
    assert [e.event_id for e in buffered] == ["e2", "e3"]


def test_silent_context_returns_copy_not_live_buffer():
    router, _, _ = make_router()
    router.on_event(make_event(silent=True, event_id="e1"))

    ctx = router.silent_context()
    ctx.append(make_event(silent=True, event_id="fake"))

    assert [e.event_id for e in router.silent_context()] == ["e1"]


def test_event_router_does_not_import_applications():
    import openneuro.state.event_router as module

    source = open(module.__file__, encoding="utf-8").read()
    assert "import applications" not in source
    assert "from openneuro.applications" not in source


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
