"""Timed execution of a single ActionRequest against an ApplicationConnector.

Scope (per the 06_action_executor_and_fakes contract): ONLY action
execution. No policy decisions, no runtime orchestration, no registry
logic live here.

Zero-context sandbox note
--------------------------
This worker's sandbox does not contain the canonical `openneuro.domain`
and `openneuro.protocol` modules produced by other OpenNeuro workers
(01_domain_models.md, 02_protocol.md). Per the worker contract, this
module:

- imports `ApplicationConnector` and `ActionRequest` only under
  `TYPE_CHECKING` (they are used purely as type hints; this module never
  constructs them, so no runtime import is required), and
- attempts to import the canonical `ActionResult` at runtime, falling back
  to a minimal local stand-in with the same shape if it isn't present yet.
  The final integrator should confirm the canonical ActionResult ends up
  being the one actually used at runtime; this fallback exists only so
  this module and its tests are self-contained pre-integration.
"""
from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Callable, Optional

if TYPE_CHECKING:
    from openneuro.domain.models import ActionRequest, ActionRequestId
    from openneuro.protocol import ApplicationConnector

try:  # pragma: no cover - exercised only once import-time, either branch
    from openneuro.domain.models import ActionResult
except ModuleNotFoundError:  # pragma: no cover - isolated-sandbox fallback
    from dataclasses import dataclass

    @dataclass(frozen=True)
    class ActionResult:  # type: ignore[no-redef]
        """Local stand-in matching the canonical ActionResult contract
        (request_id, success, message, completed_at). Used only when
        openneuro.domain.models is not importable in this sandbox."""

        request_id: Any
        success: bool
        message: str
        completed_at: datetime


DEFAULT_TIMEOUT_SECONDS: float = 20.0  # matches "Action dispatch has a default timeout of 20 seconds" in the canonical contract
_DEFAULT_TTL_SECONDS: float = 300.0
_DEFAULT_MAX_TRACKED: int = 4096


class ResolvedRequestTracker:
    """Bounded, TTL-based memory of ActionRequestIds that have already
    produced a result.

    execute() uses this so that if a connector doesn't honor cancellation
    and its dispatch_action() eventually completes *after* we've already
    returned a timeout ActionResult for that request, the late completion
    is recognized and discarded rather than silently applied a second
    time. First resolution for a given id always wins.

    Bounded by both a TTL (entries older than ttl_seconds are pruned) and
    a hard max size (oldest entries are evicted beyond that), so this can
    run unattended in a long-lived process without growing forever.
    """

    def __init__(
        self,
        ttl_seconds: float = _DEFAULT_TTL_SECONDS,
        max_size: int = _DEFAULT_MAX_TRACKED,
    ) -> None:
        self._ttl_seconds = ttl_seconds
        self._max_size = max_size
        self._resolved_at: dict[str, float] = {}

    def _prune(self, now: float) -> None:
        if self._ttl_seconds > 0:
            expired = [
                rid
                for rid, ts in self._resolved_at.items()
                if now - ts > self._ttl_seconds
            ]
            for rid in expired:
                del self._resolved_at[rid]

        overflow = len(self._resolved_at) - self._max_size
        if overflow > 0:
            oldest = sorted(self._resolved_at.items(), key=lambda item: item[1])
            for rid, _ in oldest[:overflow]:
                del self._resolved_at[rid]

    def mark_resolved(self, request_id: str) -> bool:
        """Record request_id as resolved.

        Returns True the first time it's called for a given id, False on
        every later call for the same id (i.e. this is a duplicate/late
        arrival and must not be applied again).
        """
        now = time.monotonic()
        if request_id in self._resolved_at:
            self._prune(now)
            return False
        self._resolved_at[request_id] = now
        # Prune *after* inserting, so the size bound accounts for the
        # entry we just added (pruning first would let the dict grow one
        # past max_size on every insertion).
        self._prune(now)
        return True

    def __len__(self) -> int:
        return len(self._resolved_at)


# Module-level default tracker for production use (single process, shared
# across all execute() calls that don't inject their own tracker).
_default_tracker = ResolvedRequestTracker()


def _timeout_result(request: "ActionRequest") -> "ActionResult":
    return ActionResult(
        request_id=request.id,
        success=False,
        message="timed out",
        completed_at=datetime.now(timezone.utc),
    )


async def execute(
    request: "ActionRequest",
    connector: "ApplicationConnector",
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    *,
    tracker: Optional[ResolvedRequestTracker] = None,
    on_late_result: Optional[Callable[["ActionRequestId", "ActionResult"], None]] = None,
) -> "ActionResult":
    """Dispatch request to connector.dispatch_action with a bounded timeout.

    - On timeout, returns ActionResult(success=False, message="timed out").
    - The deadline is enforced with asyncio.wait(..., timeout=...) rather
      than asyncio.wait_for(). As of Python's documented wait_for
      semantics: "If the task suppresses the cancellation and returns a
      value instead, that value is returned" - i.e. wait_for alone does
      NOT guarantee a hard deadline against a connector that ignores
      cancellation; it can silently wait past the caller's timeout. Since
      this contract explicitly requires late results to be tracked and
      discarded (implying they can occur), the deadline here is enforced
      unconditionally: cancellation is requested as a courtesy, but the
      timeout result is returned the moment the deadline is reached
      regardless of whether the connector actually stops. Any eventual
      completion after that point is a "late result".
    - Late results (a connector completing after we've already returned
      the timeout result) are recognized via a done-callback and
      discarded through `tracker` rather than applied a second time.
    - `tracker` defaults to a module-level ResolvedRequestTracker shared
      across calls; inject your own for isolation in tests.
    - `on_late_result`, if given, is called (request_id, discarded_result)
      whenever a late completion is discarded. Purely observational — it
      never changes what execute() itself returns.
    - Never touches any task/action other than this request's own.
    """
    # NOTE: deliberately `is not None`, not `tracker or _default_tracker` -
    # ResolvedRequestTracker defines __len__, so a freshly-created (empty)
    # tracker is falsy and `or` would silently swap in the wrong one.
    active_tracker = tracker if tracker is not None else _default_tracker
    request_id = str(request.id)

    dispatch_task: "asyncio.Task[ActionResult]" = asyncio.ensure_future(
        connector.dispatch_action(request)
    )

    def _on_dispatch_done(task: "asyncio.Task[ActionResult]") -> None:
        if task.cancelled():
            return
        if task.exception() is not None:
            return
        late_result = task.result()
        if not active_tracker.mark_resolved(request_id):
            if on_late_result is not None:
                on_late_result(request.id, late_result)

    dispatch_task.add_done_callback(_on_dispatch_done)

    done, _pending = await asyncio.wait({dispatch_task}, timeout=timeout_seconds)

    if dispatch_task in done:
        active_tracker.mark_resolved(request_id)
        return dispatch_task.result()

    # Deadline reached and the connector hasn't finished. Request
    # cancellation as a courtesy (well-behaved connectors will stop), but
    # don't wait for it - the done-callback above handles whatever happens
    # to this task from here on out, including a late/uncooperative result.
    dispatch_task.cancel()
    active_tracker.mark_resolved(request_id)
    return _timeout_result(request)
