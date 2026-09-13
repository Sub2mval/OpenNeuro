"""Trace events for observability across the OpenNeuro pipeline.

These are the append-only breadcrumbs emitted as a request/turn moves
through attention, policy, and execution. This module defines the shape
only; nothing here writes a trace anywhere.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class TraceEventKind(str, Enum):
    """The kind of milestone a ``TraceEvent`` records."""

    EVENT_RECEIVED = "event_received"
    ATTENTION_DECISION = "attention_decision"
    TURN_STARTED = "turn_started"
    ACTION_PROPOSED = "action_proposed"
    POLICY_DECISION = "policy_decision"
    ACTION_EXECUTED = "action_executed"
    ACTION_TIMED_OUT = "action_timed_out"
    INTERRUPT = "interrupt"
    TURN_COMPLETED = "turn_completed"


@dataclass(frozen=True)
class TraceEvent:
    """One append-only trace record.

    ``payload`` is intentionally an untyped dict: each ``kind`` defines
    its own informal payload shape, and this layer does not enforce one.
    """

    trace_id: str
    ts: datetime
    kind: TraceEventKind
    payload: dict[str, Any] = field(default_factory=dict)


__all__ = ["TraceEvent", "TraceEventKind"]
