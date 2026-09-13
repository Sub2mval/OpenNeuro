"""Domain value-object layer for OpenNeuro.

Everything in this subpackage is a plain data definition: IDs, enums, and
dataclasses. There is no business logic, no I/O, and no dependency on
Open-LLM-VTuber here. Re-export the public names so callers can write
``from openneuro.domain import ApplicationId, Priority, ...`` without
knowing which module each type lives in.
"""

from openneuro.domain.ids import (
    ActionRequestId,
    ApplicationId,
    EventId,
    SessionId,
)
from openneuro.domain.priority import Priority, priority_rank
from openneuro.domain.events import ApplicationEvent, ApplicationState
from openneuro.domain.actions import (
    ActionDefinition,
    ActionRequest,
    ActionResult,
)
from openneuro.domain.policy import PolicyContext, PolicyDecision
from openneuro.domain.attention import AttentionRequest
from openneuro.domain.agent import AgentDecision, AgentObservation, AgentStatus
from openneuro.domain.trace import TraceEvent, TraceEventKind

__all__ = [
    "ActionRequestId",
    "ApplicationId",
    "EventId",
    "SessionId",
    "Priority",
    "priority_rank",
    "ApplicationEvent",
    "ApplicationState",
    "ActionDefinition",
    "ActionRequest",
    "ActionResult",
    "PolicyContext",
    "PolicyDecision",
    "AttentionRequest",
    "AgentDecision",
    "AgentObservation",
    "AgentStatus",
    "TraceEvent",
    "TraceEventKind",
]
