"""Unit tests for openneuro.execution.action_executor.

Covers:
- immediate success / failure pass-through
- timeout behavior (returns a failed "timed out" ActionResult)
- late-result protection: a connector that doesn't honor cancellation and
  completes *after* the timeout must not have that late result applied a
  second time
- the underlying ResolvedRequestTracker's own bounded/TTL behavior
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

import pytest

from openneuro.execution.action_executor import (
    ActionResult,
    ResolvedRequestTracker,
    execute,
)
from tests.fakes.fake_connector import FakeConnector


@dataclass
class _StubActionRequest:
    """Minimal stand-in for the canonical ActionRequest contract. Only
    `id` is read by action_executor; the rest is here for realism."""

    id: str
    application_id: str = "app-1"
    action_name: str = "do_thing"
    arguments: Dict[str, Any] = field(default_factory=dict)
    session_id: str = "session-1"
    requested_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


def _request(request_id: str) -> _StubActionRequest:
    return _StubActionRequest(id=request_id)


@pytest.mark.asyncio
async def test_immediate_success_is_returned_as_is():
    ok = ActionResult(
        request_id="req-1", success=True, message="done", completed_at=datetime.now(timezone.utc)
    )
    connector = FakeConnector(result=ok)

    result = await execute(_request("req-1"), connector, timeout_seconds=1.0)

    assert result is ok
    assert connector.dispatched_request_ids == ["req-1"]


@pytest.mark.asyncio
async def test_immediate_failure_is_returned_as_is():
    failed = ActionResult(
        request_id="req-2",
        success=False,
        message="application rejected the action",
        completed_at=datetime.now(timezone.utc),
    )
    connector = FakeConnector(result=failed)

    result = await execute(_request("req-2"), connector, timeout_seconds=1.0)

    assert result is failed
    assert result.success is False


@pytest.mark.asyncio
async def test_timeout_returns_failed_timed_out_result():
    # Connector hangs well past the timeout and honors cancellation
    # (the common/cooperative case) -> no late result ever arrives.
    connector = FakeConnector(hang_seconds=10.0, ignore_cancellation=False)
    tracker = ResolvedRequestTracker()

    result = await execute(
        _request("req-3"), connector, timeout_seconds=0.02, tracker=tracker
    )

    assert result.success is False
    assert result.message == "timed out"
    assert result.request_id == "req-3"
    assert connector.dispatched_request_ids == ["req-3"]
    # the timeout itself already marked this request id resolved
    assert len(tracker) == 1


@pytest.mark.asyncio
async def test_late_result_after_timeout_is_discarded_not_applied_twice():
    # Connector ignores the first cancellation attempt and eventually
    # completes for real, *after* our timeout has already fired.
    late_success = ActionResult(
        request_id="req-4",
        success=True,
        message="actually finished eventually",
        completed_at=datetime.now(timezone.utc),
    )
    connector = FakeConnector(
        result=late_success, hang_seconds=0.02, ignore_cancellation=True
    )
    tracker = ResolvedRequestTracker()
    late_events: List[Tuple[Any, ActionResult]] = []

    result = await execute(
        _request("req-4"),
        connector,
        timeout_seconds=0.01,
        tracker=tracker,
        on_late_result=lambda request_id, late_result: late_events.append(
            (request_id, late_result)
        ),
    )

    # execute() itself only ever returns ONE result: the timeout result.
    assert result.success is False
    assert result.message == "timed out"

    # Give the connector's background sleep time to actually finish and
    # fire the late completion callback.
    await asyncio.sleep(0.08)

    # The real (late) result did arrive, but it was recognized as a
    # duplicate/late arrival and only surfaced via the observational
    # `on_late_result` hook -- never as a second return value from
    # execute(), and never re-marked as a fresh resolution.
    assert len(late_events) == 1
    late_request_id, late_result = late_events[0]
    assert late_request_id == "req-4"
    assert late_result is late_success
    assert len(tracker) == 1  # still just the one (timeout) resolution


@pytest.mark.asyncio
async def test_two_different_requests_do_not_interfere():
    # One connector times out; a second, unrelated connector for a
    # different request succeeds immediately. Neither should affect the
    # other (executor must not cancel/poison unrelated actions).
    tracker = ResolvedRequestTracker()

    hanging_connector = FakeConnector(hang_seconds=10.0)
    ok = ActionResult(
        request_id="req-6", success=True, message="fine", completed_at=datetime.now(timezone.utc)
    )
    fine_connector = FakeConnector(result=ok)

    timed_out_result, fine_result = await asyncio.gather(
        execute(_request("req-5"), hanging_connector, timeout_seconds=0.02, tracker=tracker),
        execute(_request("req-6"), fine_connector, timeout_seconds=1.0, tracker=tracker),
    )

    assert timed_out_result.message == "timed out"
    assert fine_result is ok
    assert len(tracker) == 2


# --- ResolvedRequestTracker behavior in isolation -------------------------


def test_tracker_first_mark_wins_second_is_rejected():
    tracker = ResolvedRequestTracker()

    assert tracker.mark_resolved("req-1") is True
    assert tracker.mark_resolved("req-1") is False  # late/duplicate
    assert tracker.mark_resolved("req-2") is True  # unrelated id, unaffected


def test_tracker_is_bounded_by_max_size(monkeypatch):
    tracker = ResolvedRequestTracker(ttl_seconds=3600, max_size=3)

    fake_time = {"t": 0.0}
    monkeypatch.setattr("time.monotonic", lambda: fake_time["t"])

    for i in range(5):
        fake_time["t"] += 1.0
        tracker.mark_resolved(f"req-{i}")

    assert len(tracker) == 3
    # the most recently resolved ids should be the ones retained
    assert tracker.mark_resolved("req-4") is False
    assert tracker.mark_resolved("req-3") is False
    # an early id should have been evicted and is treated as "new" again
    assert tracker.mark_resolved("req-0") is True


def test_tracker_prunes_by_ttl(monkeypatch):
    tracker = ResolvedRequestTracker(ttl_seconds=10, max_size=1000)

    fake_time = {"t": 0.0}
    monkeypatch.setattr("time.monotonic", lambda: fake_time["t"])

    tracker.mark_resolved("req-old")
    fake_time["t"] = 20.0  # past the 10s ttl

    # resolving anything now prunes req-old first
    assert tracker.mark_resolved("req-new") is True
    assert tracker.mark_resolved("req-old") is True  # expired, so "new" again
