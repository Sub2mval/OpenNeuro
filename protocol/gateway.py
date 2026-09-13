"""``ApplicationGateway``: the structural contract implemented by the runtime.

An instance satisfying this Protocol is handed to each
``ApplicationConnector.start()`` call. As with ``connector.py``, domain
types are only needed for type-checking and are imported under
``TYPE_CHECKING`` so this module has no runtime dependency on the
(possibly absent) domain package, on the runtime, on applications, or on
the agent adapter / embodiment layers. It performs no I/O and contains no
application-name literals.

The one exception is ``Priority``, whose ``LOW`` member is the contractual
default for ``publish_event``'s ``priority`` argument and therefore must
exist as a real object at import time, not just a type. If the domain
package is present in this sandbox its real ``Priority`` is used; if not
(this worker is meant to be runnable in isolation), a minimal local
stand-in with the same member is used instead. Either way this module
never branches on application identity and the fallback is structural
only — it does not implement any domain behavior.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from openneuro.domain.actions import ActionDefinition
    from openneuro.domain.attention import AttentionRequest
    from openneuro.domain.events import Priority
    from openneuro.domain.ids import EventId
    from openneuro.domain.state import ApplicationState

try:  # pragma: no cover - exercised only when domain package is absent
    from openneuro.domain.events import Priority as _Priority
except ImportError:  # pragma: no cover
    import enum

    class _Priority(str, enum.Enum):
        """Local stand-in used only when the domain package isn't present.

        Structurally identical (str Enum, same member names) to the
        canonical ``Priority``. The final integrator collapses this to the
        one real ``Priority`` from the domain module.
        """

        LOW = "low"
        MEDIUM = "medium"
        HIGH = "high"
        CRITICAL = "critical"


@runtime_checkable
class ApplicationGateway(Protocol):
    """Structural contract implemented by the runtime for applications."""

    def register_actions(self, actions: "list[ActionDefinition]") -> None:
        """Register (or replace, by name) this application's actions."""
        ...

    def unregister_actions(self, action_names: "list[str]") -> None:
        """Remove the named actions previously registered by this application."""
        ...

    def publish_state(self, state: "ApplicationState") -> None:
        """Publish a new ``ApplicationState`` snapshot for this application."""
        ...

    def publish_event(
        self,
        message: str,
        silent: bool,
        priority: "Priority" = _Priority.LOW,
    ) -> "EventId":
        """Publish an application event and return its assigned ``EventId``."""
        ...

    def request_attention(self, request: "AttentionRequest") -> None:
        """Request agent attention, replacing any unresolved prior request."""
        ...

    def disconnect(self) -> None:
        """Disconnect this application from the runtime."""
        ...
