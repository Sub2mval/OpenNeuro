from datetime import datetime, timedelta

import pytest

from openneuro.attention import (
    AgentStatus,
    AttentionManager,
    AttentionManagerSettings,
    InterruptDecision,
    Priority,
    decide,
)
from openneuro.attention.attention_manager import AttentionRequest


def make_request(
    priority: Priority,
    *,
    application_id: str = "app-1",
    query: str = "need input",
    created_at: datetime | None = None,
) -> AttentionRequest:
    return AttentionRequest(
        application_id=application_id,  # type: ignore[arg-type]
        event_id=None,
        priority=priority,
        state="waiting",
        query=query,
        ephemeral=False,
        candidate_actions=[],
        created_at=created_at or datetime(2026, 1, 1, 12, 0, 0),
        expires_at=None,
    )


# ---------------------------------------------------------------------------
# decide() exhaustive status x priority matrix
# ---------------------------------------------------------------------------

EXPECTED = {
    (AgentStatus.IDLE, Priority.LOW): InterruptDecision.QUEUE,
    (AgentStatus.THINKING, Priority.LOW): InterruptDecision.QUEUE,
    (AgentStatus.SPEAKING, Priority.LOW): InterruptDecision.QUEUE,
    (AgentStatus.EXECUTING_ACTION, Priority.LOW): InterruptDecision.QUEUE,
    (AgentStatus.WAITING_FOR_RESULT, Priority.LOW): InterruptDecision.QUEUE,
    (AgentStatus.IDLE, Priority.MEDIUM): InterruptDecision.QUEUE,
    (AgentStatus.THINKING, Priority.MEDIUM): InterruptDecision.QUEUE,
    (AgentStatus.SPEAKING, Priority.MEDIUM): InterruptDecision.INTERRUPT_SPEECH,
    (AgentStatus.EXECUTING_ACTION, Priority.MEDIUM): InterruptDecision.QUEUE,
    (AgentStatus.WAITING_FOR_RESULT, Priority.MEDIUM): InterruptDecision.QUEUE,
    (AgentStatus.IDLE, Priority.HIGH): InterruptDecision.REPLACE_PENDING,
    (AgentStatus.THINKING, Priority.HIGH): InterruptDecision.REPLACE_PENDING,
    (AgentStatus.SPEAKING, Priority.HIGH): InterruptDecision.INTERRUPT_SPEECH,
    (AgentStatus.EXECUTING_ACTION, Priority.HIGH): InterruptDecision.INTERRUPT_ACTION,
    (AgentStatus.WAITING_FOR_RESULT, Priority.HIGH): InterruptDecision.REPLACE_PENDING,
    (AgentStatus.IDLE, Priority.CRITICAL): InterruptDecision.REPLACE_PENDING,
    (AgentStatus.THINKING, Priority.CRITICAL): InterruptDecision.REPLACE_PENDING,
    (AgentStatus.SPEAKING, Priority.CRITICAL): InterruptDecision.INTERRUPT_SPEECH,
    (AgentStatus.EXECUTING_ACTION, Priority.CRITICAL): InterruptDecision.INTERRUPT_ACTION,
    (AgentStatus.WAITING_FOR_RESULT, Priority.CRITICAL): InterruptDecision.REPLACE_PENDING,
}


def test_decide_matrix_is_exhaustive_over_all_statuses_and_priorities():
    all_pairs = {(s, p) for s in AgentStatus for p in Priority}
    assert all_pairs == set(EXPECTED.keys())


@pytest.mark.parametrize("status,priority", list(EXPECTED.keys()))
def test_decide_matches_contract(status, priority):
    assert decide(status, priority) == EXPECTED[(status, priority)]


def test_low_never_interrupts_any_status():
    for status in AgentStatus:
        assert decide(status, Priority.LOW) == InterruptDecision.QUEUE


def test_critical_never_merely_queues():
    for status in AgentStatus:
        assert decide(status, Priority.CRITICAL) != InterruptDecision.QUEUE


# ---------------------------------------------------------------------------
# AttentionManager: pending slot replacement
# ---------------------------------------------------------------------------


def test_high_priority_becomes_pending_and_replaces_previous_pending():
    mgr = AttentionManager(initial_status=AgentStatus.IDLE)
    first = make_request(Priority.HIGH)
    second = make_request(Priority.CRITICAL)

    decision1 = mgr.submit(first)
    assert decision1 == InterruptDecision.REPLACE_PENDING
    assert mgr.pending is first

    decision2 = mgr.submit(second)
    assert decision2 == InterruptDecision.REPLACE_PENDING
    assert mgr.pending is second  # new force replaced the old pending one


def test_take_next_returns_and_clears_pending_before_queue():
    mgr = AttentionManager(initial_status=AgentStatus.IDLE)
    queued = make_request(Priority.LOW)
    forced = make_request(Priority.HIGH)

    mgr.submit(queued)
    mgr.submit(forced)

    assert mgr.take_next() is forced
    assert mgr.pending is None
    # Now the queued LOW request should be next.
    assert mgr.take_next() is queued


def test_take_next_drains_queue_in_priority_then_fifo_order():
    mgr = AttentionManager(initial_status=AgentStatus.EXECUTING_ACTION)
    # EXECUTING_ACTION + MEDIUM -> QUEUE (per matrix); use two MEDIUMs + one LOW
    low = make_request(Priority.LOW, application_id="low-app")
    med1 = make_request(Priority.MEDIUM, application_id="med-1")
    med2 = make_request(Priority.MEDIUM, application_id="med-2")

    mgr.submit(low)
    mgr.submit(med1)
    mgr.submit(med2)

    assert mgr.take_next() is med1  # higher priority first
    assert mgr.take_next() is med2  # FIFO within same priority
    assert mgr.take_next() is low
    assert mgr.take_next() is None


# ---------------------------------------------------------------------------
# Bounded queue capacity
# ---------------------------------------------------------------------------


def test_queue_capacity_is_bounded_and_evicts_lowest_priority_oldest_first():
    settings = AttentionManagerSettings(queue_max=2)
    mgr = AttentionManager(settings=settings, initial_status=AgentStatus.EXECUTING_ACTION)

    low1 = make_request(Priority.LOW, application_id="low-1")
    low2 = make_request(Priority.LOW, application_id="low-2")
    med1 = make_request(Priority.MEDIUM, application_id="med-1")

    mgr.submit(low1)
    mgr.submit(low2)
    assert mgr.queue_depth() == 2

    # Queue is full: submitting another queued item must evict the oldest
    # lowest-priority entry (low1) to make room.
    mgr.submit(med1)
    assert mgr.queue_depth() == 2
    assert mgr.queue_depth(Priority.LOW) == 1  # low1 evicted, low2 survived
    # med1 outranks low2, so it still drains first despite arriving later.
    assert mgr.take_next() is med1
    assert mgr.take_next() is low2


def test_queue_max_is_read_from_injected_settings_not_environment(monkeypatch):
    monkeypatch.setenv("OPENNEURO_ATTENTION_QUEUE_MAX", "999")
    settings = AttentionManagerSettings(queue_max=1)
    mgr = AttentionManager(settings=settings, initial_status=AgentStatus.EXECUTING_ACTION)

    mgr.submit(make_request(Priority.LOW, application_id="a"))
    mgr.submit(make_request(Priority.LOW, application_id="b"))

    # Bound must come from the injected settings object (1), never os.environ (999).
    assert mgr.queue_depth() == 1


# ---------------------------------------------------------------------------
# Starvation aging
# ---------------------------------------------------------------------------


def test_aging_promotes_stale_low_priority_request_by_one_level():
    settings = AttentionManagerSettings(starvation_age=timedelta(seconds=10))
    mgr = AttentionManager(settings=settings, initial_status=AgentStatus.EXECUTING_ACTION)

    t0 = datetime(2026, 1, 1, 12, 0, 0)
    req = make_request(Priority.LOW, application_id="stale")
    mgr.submit(req, now=t0)

    # Not old enough yet.
    promoted = mgr.apply_aging(t0 + timedelta(seconds=5))
    assert promoted == 0
    assert mgr.queue_depth(Priority.LOW) == 1
    assert mgr.queue_depth(Priority.MEDIUM) == 0

    # Old enough now: promote LOW -> MEDIUM.
    promoted = mgr.apply_aging(t0 + timedelta(seconds=11))
    assert promoted == 1
    assert mgr.queue_depth(Priority.LOW) == 0
    assert mgr.queue_depth(Priority.MEDIUM) == 1

    next_request = mgr.take_next()
    assert next_request.priority == Priority.MEDIUM
    assert next_request.application_id == "stale"


def test_aging_never_promotes_above_critical():
    settings = AttentionManagerSettings(starvation_age=timedelta(seconds=1))
    mgr = AttentionManager(settings=settings, initial_status=AgentStatus.SPEAKING)
    # SPEAKING + LOW -> QUEUE, so this ends up in the LOW queue.
    t0 = datetime(2026, 1, 1, 12, 0, 0)
    mgr.submit(make_request(Priority.LOW, application_id="a"), now=t0)

    # Repeatedly age well past several thresholds; CRITICAL request in the
    # (hypothetical) CRITICAL queue must never be touched/promoted further,
    # and a request cannot cascade past CRITICAL in a single pass.
    later = t0 + timedelta(seconds=100)
    for _ in range(3):
        mgr.apply_aging(later)
        later += timedelta(seconds=100)

    depths = {p: mgr.queue_depth(p) for p in Priority}
    assert sum(depths.values()) == 1
    assert depths[Priority.CRITICAL] <= 1  # never more than the one item, never crashes


def test_aging_preserves_fifo_order_within_promoted_priority():
    settings = AttentionManagerSettings(starvation_age=timedelta(seconds=5))
    mgr = AttentionManager(settings=settings, initial_status=AgentStatus.EXECUTING_ACTION)

    t0 = datetime(2026, 1, 1, 12, 0, 0)
    stale_low = make_request(Priority.LOW, application_id="promoted-low")
    existing_medium = make_request(Priority.MEDIUM, application_id="already-medium")

    # stale_low has been waiting since t0; existing_medium arrived later and
    # is still fresh relative to the aging check below.
    mgr.submit(stale_low, now=t0)
    mgr.submit(existing_medium, now=t0 + timedelta(seconds=4))

    check_time = t0 + timedelta(seconds=6)
    promoted = mgr.apply_aging(check_time)

    # Only the stale LOW item (age 6s >= 5s) is old enough; the MEDIUM item
    # (age 2s) is untouched and does not get promoted to HIGH.
    assert promoted == 1
    assert mgr.queue_depth(Priority.MEDIUM) == 2
    assert mgr.queue_depth(Priority.HIGH) == 0

    # The already-queued MEDIUM item stays ahead of the newly promoted one.
    assert mgr.take_next().application_id == "already-medium"
    assert mgr.take_next().application_id == "promoted-low"


def test_status_helper_updates_current_status():
    mgr = AttentionManager(initial_status=AgentStatus.IDLE)
    assert mgr.current_status == AgentStatus.IDLE
    mgr.set_status(AgentStatus.SPEAKING)
    assert mgr.current_status == AgentStatus.SPEAKING
