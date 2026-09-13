"""``ApplicationConnector``: the structural contract each application implements.

No concrete application, runtime, or agent code is imported here. Domain
types are only needed for type-checking, so they are imported under
``TYPE_CHECKING`` and referenced as forward-reference strings — this module
has zero runtime dependency on the (possibly absent) domain package.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from openneuro.domain.ids import ApplicationId
    from openneuro.domain.actions import ActionRequest, ActionResult
    from openneuro.protocol.gateway import ApplicationGateway


@runtime_checkable
class ApplicationConnector(Protocol):
    """Structural contract implemented by each application.

    Implementations are plain objects (in-process for demo/test connectors;
    remote transport is out of scope here) satisfied purely by shape —
    no subclassing of this Protocol is required.
    """

    application_id: "ApplicationId"

    async def start(self, gateway: "ApplicationGateway") -> None:
        """Start the connector, wiring it to the runtime via ``gateway``.

        Per core semantics, the runtime treats a call to ``start()`` as
        clearing any actions previously registered for this
        ``application_id`` before the connector registers its current set.
        """
        ...

    async def dispatch_action(self, request: "ActionRequest") -> "ActionResult":
        """Execute a single action request and return its result.

        The runtime applies a default 20 second timeout to dispatch; a
        result returned after the runtime has already timed out the
        matching ``request.id`` is discarded by the runtime, not by the
        connector.
        """
        ...

    async def stop(self) -> None:
        """Stop the connector and release any resources it holds."""
        ...
