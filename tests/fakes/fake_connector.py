"""Test fake for ApplicationConnector, scoped to the execution layer's own
unit tests.

Provided because, at the time this worker ran, its isolated sandbox did not
contain a canonical ApplicationConnector test fake (that would normally
come from 02_protocol.md or 08_test_llm_and_simulation.md). If the
integrated repository already provides a compatible one under tests/fakes/,
the integrator should keep only one canonical implementation and delete
this duplicate.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, List

if TYPE_CHECKING:
    from openneuro.domain.models import ActionRequest, ActionResult
    from openneuro.protocol import ApplicationGateway


@dataclass
class FakeConnector:
    """Minimal in-process ApplicationConnector double.

    Configure exactly one behavior per instance:

    - Immediate success/failure: set `result` to an ActionResult-like
      object (with success=True or success=False) and leave `hang_seconds`
      as None. dispatch_action returns `result` right away.

    - Hang past the caller's timeout: set `hang_seconds` to longer than the
      caller's timeout. dispatch_action sleeps; when the caller's
      asyncio.wait_for times out, this task is cancelled during that
      sleep and dispatch_action never produces a result.

    - Hang, then produce a late result anyway (simulates a connector that
      does not honor cancellation): same as above but with
      `ignore_cancellation=True`. The first CancelledError raised into the
      sleep is swallowed once, then it sleeps again and returns `result` -
      i.e. a real result arrives *after* the caller already timed out.

    Every call records request.id in `dispatched_request_ids`, in order,
    including calls that are later cancelled.
    """

    application_id: str = "fake-app"
    result: "ActionResult | None" = None
    hang_seconds: "float | None" = None
    ignore_cancellation: bool = False
    dispatched_request_ids: List[Any] = field(default_factory=list)

    async def start(self, gateway: "ApplicationGateway") -> None:  # pragma: no cover
        return None

    async def stop(self) -> None:  # pragma: no cover
        return None

    async def dispatch_action(self, request: "ActionRequest") -> "ActionResult":
        self.dispatched_request_ids.append(request.id)

        if self.hang_seconds is not None:
            try:
                await asyncio.sleep(self.hang_seconds)
            except asyncio.CancelledError:
                if not self.ignore_cancellation:
                    raise
                # Simulate a connector that doesn't honor cancellation:
                # keep running and eventually produce a real result.
                await asyncio.sleep(self.hang_seconds)

        if self.result is None:
            raise AssertionError(
                "FakeConnector.result must be set for a non-hanging dispatch, "
                "or hang_seconds must lead to a raised CancelledError"
            )
        return self.result
