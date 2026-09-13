"""Shared test doubles for OpenNeuro workers.

This package is deliberately self-contained per the zero-context worker
contract: it first tries to import the canonical OpenNeuro domain contracts
from ``openneuro.domain`` (owned by worker 01). If that module is not present
in this sandbox, a tiny local fallback that mirrors the documented contract
is used instead, so these fakes and their tests can run standalone. Once the
final integrator merges worker 01's real module, the ``try`` branch below
picks it up automatically and the fallback becomes dead code.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Literal, NewType, Optional

try:  # pragma: no cover - exercised only once the domain worker is merged in
    from openneuro.domain import (
        ActionDefinition,
        ActionRequest,
        ActionRequestId,
        ActionResult,
        ApplicationEvent,
        ApplicationId,
        ApplicationState,
        AttentionRequest,
        EventId,
        Priority,
        SessionId,
    )
except ImportError:  # pragma: no cover - fallback path used in isolated sandbox
    ApplicationId = NewType("ApplicationId", str)
    SessionId = NewType("SessionId", str)
    EventId = NewType("EventId", str)
    ActionRequestId = NewType("ActionRequestId", str)

    class Priority(str, Enum):
        LOW = "low"
        MEDIUM = "medium"
        HIGH = "high"
        CRITICAL = "critical"

    @dataclass
    class ApplicationState:
        application_id: ApplicationId
        data: Dict[str, Any]
        version: int
        updated_at: datetime

    @dataclass
    class ApplicationEvent:
        event_id: EventId
        application_id: ApplicationId
        message: str
        silent: bool
        priority: Priority
        created_at: datetime

    @dataclass
    class ActionDefinition:
        application_id: ApplicationId
        name: str
        description: str
        schema: Dict[str, Any]
        risk: Literal["low", "medium", "high"]
        registered_at: datetime

    @dataclass
    class ActionRequest:
        id: ActionRequestId
        application_id: ApplicationId
        action_name: str
        arguments: Dict[str, Any]
        session_id: SessionId
        requested_at: datetime

    @dataclass
    class ActionResult:
        request_id: ActionRequestId
        success: bool
        message: str
        completed_at: datetime

    @dataclass
    class AttentionRequest:
        application_id: ApplicationId
        event_id: Optional[EventId]
        priority: Priority
        state: str
        query: str
        ephemeral: bool
        candidate_actions: List[str]
        created_at: datetime
        expires_at: Optional[datetime] = None


__all__ = [
    "ApplicationId",
    "SessionId",
    "EventId",
    "ActionRequestId",
    "Priority",
    "ApplicationState",
    "ApplicationEvent",
    "ActionDefinition",
    "ActionRequest",
    "ActionResult",
    "AttentionRequest",
]
