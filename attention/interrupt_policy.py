"""Pure interruption-decision logic.

This module owns exactly one thing: the deterministic mapping from
(current agent status, incoming event priority) -> InterruptDecision.
It has no state and no side effects.

Canonical domain types (``Priority``, ``AgentStatus``) are defined by the
domain-models worker (01_domain_models.md) at ``openneuro.domain.models``.
That module is not guaranteed to exist in this isolated sandbox, so we try
to import the real thing first and fall back to a local, contract-identical
stub. When the real integration lands, the ``try`` branch succeeds and the
stub is simply unused.
"""

from __future__ import annotations

from enum import Enum

try:  # pragma: no cover - exercised only once real domain models exist
    from openneuro.domain.models import AgentStatus, Priority
except ImportError:  # pragma: no cover - expected in isolated worker sandbox

    class Priority(str, Enum):
        LOW = "low"
        MEDIUM = "medium"
        HIGH = "high"
        CRITICAL = "critical"

    class AgentStatus(str, Enum):
        IDLE = "idle"
        THINKING = "thinking"
        SPEAKING = "speaking"
        EXECUTING_ACTION = "executing_action"
        WAITING_FOR_RESULT = "waiting_for_result"


class InterruptDecision(str, Enum):
    """Outcome of evaluating an incoming priority against agent status."""

    QUEUE = "queue"
    INTERRUPT_SPEECH = "interrupt_speech"
    INTERRUPT_ACTION = "interrupt_action"
    REPLACE_PENDING = "replace_pending"


# Statuses in which the agent is actively producing output that a strong
# enough priority can preempt.
_SPEAKING_LIKE = frozenset({AgentStatus.SPEAKING})
_ACTING_LIKE = frozenset({AgentStatus.EXECUTING_ACTION})


def decide(current_status: AgentStatus, incoming_priority: Priority) -> InterruptDecision:
    """Return the deterministic InterruptDecision for this status/priority pair.

    Rules (from the contract):
      - LOW never interrupts; it always queues, regardless of status.
      - MEDIUM interrupts SPEAKING only; otherwise it queues.
      - HIGH interrupts SPEAKING or EXECUTING_ACTION; for any other status it
        forces itself into the single pending slot (REPLACE_PENDING), per the
        "one unresolved attention request" pending-force semantics.
      - CRITICAL interrupts immediately regardless of status: it preempts
        SPEAKING/EXECUTING_ACTION the same way HIGH does, and otherwise also
        forces the pending slot (it never merely queues).
    """
    if incoming_priority == Priority.LOW:
        return InterruptDecision.QUEUE

    if incoming_priority == Priority.MEDIUM:
        if current_status in _SPEAKING_LIKE:
            return InterruptDecision.INTERRUPT_SPEECH
        return InterruptDecision.QUEUE

    # HIGH and CRITICAL share the same status-based preemption shape; the
    # difference between them (CRITICAL always forcing itself, HIGH doing so
    # only for non-active statuses) collapses to the same mapping here since
    # neither priority is ever merely queued.
    if incoming_priority in (Priority.HIGH, Priority.CRITICAL):
        if current_status in _SPEAKING_LIKE:
            return InterruptDecision.INTERRUPT_SPEECH
        if current_status in _ACTING_LIKE:
            return InterruptDecision.INTERRUPT_ACTION
        return InterruptDecision.REPLACE_PENDING

    raise ValueError(f"Unknown priority: {incoming_priority!r}")
