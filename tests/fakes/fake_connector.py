"""In-process test double for the ApplicationConnector protocol.

Structural (duck-typed) implementation of:

    class ApplicationConnector(Protocol):
        application_id: ApplicationId
        async def start(self, gateway: ApplicationGateway) -> None: ...
        async def dispatch_action(self, request: ActionRequest) -> ActionResult: ...
        async def stop(self) -> None: ...

Not imported from openneuro.protocol directly (that worker's module may be
absent in this sandbox) -- this class simply satisfies the shape.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set

from . import ActionRequest, ActionResult, ApplicationId


@dataclass
class RecordedCall:
    """A single call made against a FakeConnector, for test assertions."""

    name: str
    args: Dict[str, Any] = field(default_factory=dict)


class FakeConnector:
    """Scriptable double for ApplicationConnector.

    - Records every call (``start``/``dispatch_action``/``stop``) in order.
    - Lets tests queue one or more ``ActionResult``s per action name; each
      dispatch consumes the next queued result (or falls back to a generic
      success result if none was queued).
    - Lets tests mark an action name as "hanging" so ``dispatch_action``
      never resolves, to exercise executor-level timeout/cancellation logic
      elsewhere in the system.
    """

    def __init__(self, application_id: ApplicationId) -> None:
        self.application_id = application_id
        self.calls: List[RecordedCall] = []
        self.started_with_gateway: Optional[Any] = None
        self.stopped: bool = False
        self._queued_results: Dict[str, List[ActionResult]] = {}
        self._hang_actions: Set[str] = set()

    # -- scripting API ------------------------------------------------------

    def queue_result(self, action_name: str, result: ActionResult) -> None:
        self._queued_results.setdefault(action_name, []).append(result)

    def hang_on(self, action_name: str) -> None:
        self._hang_actions.add(action_name)

    def unhang(self, action_name: str) -> None:
        self._hang_actions.discard(action_name)

    # -- ApplicationConnector shape ------------------------------------------

    async def start(self, gateway: Any) -> None:
        self.calls.append(RecordedCall("start", {"gateway": gateway}))
        self.started_with_gateway = gateway
        self.stopped = False

    async def dispatch_action(self, request: ActionRequest) -> ActionResult:
        self.calls.append(RecordedCall("dispatch_action", {"request": request}))
        if request.action_name in self._hang_actions:
            # Never resolves; the caller (executor) is responsible for
            # timing out and/or cancelling this coroutine.
            await asyncio.Event().wait()

        queue = self._queued_results.get(request.action_name)
        if queue:
            return queue.pop(0)

        return ActionResult(
            request_id=request.id,
            success=True,
            message=f"fake-default-result:{request.action_name}",
            completed_at=datetime.now(timezone.utc),
        )

    async def stop(self) -> None:
        self.calls.append(RecordedCall("stop"))
        self.stopped = True

    # -- assertion helpers ----------------------------------------------------

    def call_names(self) -> List[str]:
        return [call.name for call in self.calls]
