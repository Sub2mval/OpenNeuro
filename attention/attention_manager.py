"""In-memory attention manager: pending slot + bounded FIFO-within-priority queue.

Owns exactly the attention/interruption state machine described in
04_attention.md. It does not know about applications, actions, or policy —
only about AgentStatus, Priority, and AttentionRequest.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Deque, Dict, Optional

from .interrupt_policy import AgentStatus, InterruptDecision, Priority, decide

try:  # pragma: no cover - exercised only once real domain models exist
    from openneuro.domain.models import ApplicationId, AttentionRequest, EventId
except ImportError:  # pragma: no cover - expected in isolated worker sandbox
    from typing import Any, List, NewType

    ApplicationId = NewType("ApplicationId", str)
    EventId = NewType("EventId", str)

    @dataclass(frozen=True)
    class AttentionRequest:  # type: ignore[no-redef]
        application_id: ApplicationId
        event_id: Optional[EventId]
        priority: Priority
        state: str
        query: str
        ephemeral: bool
        candidate_actions: List[str]
        created_at: datetime
        expires_at: Optional[datetime] = None


# Priorities in the order they should be drained from the queue (highest first).
_PRIORITY_ORDER = (Priority.CRITICAL, Priority.HIGH, Priority.MEDIUM, Priority.LOW)

# One step "up" the priority ladder, used by starvation aging. CRITICAL has no
# successor: aging must never promote past it.
_PROMOTION: Dict[Priority, Priority] = {
    Priority.LOW: Priority.MEDIUM,
    Priority.MEDIUM: Priority.HIGH,
    Priority.HIGH: Priority.CRITICAL,
    Priority.CRITICAL: Priority.CRITICAL,
}


@dataclass(frozen=True)
class AttentionManagerSettings:
    """Configuration injected into AttentionManager.

    Resolution of ``OPENNEURO_ATTENTION_QUEUE_MAX`` (or any other env-backed
    override) is the config layer's job, not this module's — this module
    only ever reads the already-resolved values below.
    """

    queue_max: int = 50
    starvation_age: timedelta = timedelta(seconds=30)


@dataclass
class _QueuedItem:
    request: AttentionRequest
    priority: Priority
    enqueued_at: datetime


class AttentionManager:
    """Tracks agent status, one pending attention request, and a bounded queue."""

    def __init__(
        self,
        settings: Optional[AttentionManagerSettings] = None,
        *,
        initial_status: AgentStatus = AgentStatus.IDLE,
    ) -> None:
        self._settings = settings or AttentionManagerSettings()
        self._status: AgentStatus = initial_status
        self._pending: Optional[AttentionRequest] = None
        self._queues: Dict[Priority, Deque[_QueuedItem]] = {
            p: deque() for p in _PRIORITY_ORDER
        }

    # -- status -----------------------------------------------------------

    @property
    def current_status(self) -> AgentStatus:
        return self._status

    def set_status(self, status: AgentStatus) -> None:
        """Update the agent's current status (e.g. on turn transitions)."""
        self._status = status

    # -- pending ------------------------------------------------------------

    @property
    def pending(self) -> Optional[AttentionRequest]:
        return self._pending

    # -- submission ---------------------------------------------------------

    def submit(self, request: AttentionRequest, *, now: Optional[datetime] = None) -> InterruptDecision:
        """Submit an incoming AttentionRequest and return the decision made.

        Forcing decisions (INTERRUPT_SPEECH, INTERRUPT_ACTION, REPLACE_PENDING)
        all mean the request becomes the single pending slot, unconditionally
        replacing whatever was pending before. QUEUE means the request is
        appended to the bounded FIFO-within-priority queue.
        """
        decision = decide(self._status, request.priority)
        if decision == InterruptDecision.QUEUE:
            self._enqueue(request, now=now or datetime.now())
        else:
            # A new force always replaces the old pending one.
            self._pending = request
        return decision

    # -- consumption --------------------------------------------------------

    def take_next(self) -> Optional[AttentionRequest]:
        """Return the next request to attend to, preferring the pending slot.

        If a pending (forced) request exists it is returned first and
        cleared. Otherwise the highest-priority, oldest queued request is
        popped off the FIFO-within-priority queue.
        """
        if self._pending is not None:
            request, self._pending = self._pending, None
            return request

        for priority in _PRIORITY_ORDER:
            queue = self._queues[priority]
            if queue:
                return queue.popleft().request
        return None

    def queue_depth(self, priority: Optional[Priority] = None) -> int:
        """Number of queued (non-pending) requests, optionally filtered by priority."""
        if priority is not None:
            return len(self._queues[priority])
        return sum(len(q) for q in self._queues.values())

    # -- queue internals ------------------------------------------------------

    def _enqueue(self, request: AttentionRequest, *, now: datetime) -> None:
        queue = self._queues[request.priority]
        if self.queue_depth() >= self._settings.queue_max:
            self._evict_one()
        queue.append(_QueuedItem(request=request, priority=request.priority, enqueued_at=now))

    def _evict_one(self) -> None:
        """Drop the oldest item from the lowest-priority non-empty queue."""
        for priority in reversed(_PRIORITY_ORDER):  # LOW first, CRITICAL last
            queue = self._queues[priority]
            if queue:
                queue.popleft()
                return

    # -- starvation aging -----------------------------------------------------

    def apply_aging(self, now: datetime) -> int:
        """Promote queued requests that have waited >= starvation_age by one
        priority level (never above CRITICAL). Returns the number promoted.

        Promoted items have their enqueue clock reset to ``now`` so a single
        aging pass cannot cascade an item through multiple levels at once.
        """
        promoted = 0
        threshold = self._settings.starvation_age

        # Walk from lowest to highest so a promoted item lands in a queue we
        # have not yet examined this pass (avoiding double-promotion).
        for priority in reversed(_PRIORITY_ORDER):  # LOW, MEDIUM, HIGH, CRITICAL
            if priority == Priority.CRITICAL:
                continue  # already at the ceiling, nothing to promote to
            queue = self._queues[priority]
            still_waiting: Deque[_QueuedItem] = deque()
            while queue:
                item = queue.popleft()
                if now - item.enqueued_at >= threshold:
                    new_priority = _PROMOTION[priority]
                    promoted_request = _with_priority(item.request, new_priority)
                    self._queues[new_priority].append(
                        _QueuedItem(request=promoted_request, priority=new_priority, enqueued_at=now)
                    )
                    promoted += 1
                else:
                    still_waiting.append(item)
            # Anything not promoted stays in this priority's queue, order preserved.
            queue.extend(still_waiting)
        return promoted


def _with_priority(request: AttentionRequest, new_priority: Priority) -> AttentionRequest:
    """Return a copy of ``request`` with its priority bumped.

    AttentionRequest is treated as an immutable value object (frozen
    dataclass per the canonical contract), so aging promotion produces a
    new instance rather than mutating the original.
    """
    try:
        from dataclasses import replace

        return replace(request, priority=new_priority)
    except TypeError:
        # Fallback for a non-dataclass AttentionRequest implementation: build
        # a shallow copy via its own constructor's field names.
        data = {
            "application_id": request.application_id,
            "event_id": request.event_id,
            "priority": new_priority,
            "state": request.state,
            "query": request.query,
            "ephemeral": request.ephemeral,
            "candidate_actions": request.candidate_actions,
            "created_at": request.created_at,
            "expires_at": request.expires_at,
        }
        return type(request)(**data)
