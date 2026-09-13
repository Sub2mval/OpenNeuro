"""Value objects describing actions: their definitions, requests, and results."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

from openneuro.domain.ids import ActionRequestId, ApplicationId, SessionId

ActionRisk = Literal["low", "medium", "high"]


@dataclass(frozen=True)
class ActionDefinition:
    """A single action an application has registered as callable.

    ``schema`` is a plain JSON-schema-like dict describing ``arguments``
    accepted by the action; OpenNeuro does not validate against it here,
    that is the concern of whatever component executes the action.
    """

    application_id: ApplicationId
    name: str
    description: str
    schema: dict[str, Any] = field(default_factory=dict)
    risk: ActionRisk = "low"
    registered_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass(frozen=True)
class ActionRequest:
    """A single request to invoke a registered action.

    ``id`` is the correlation key used to discard late/timed-out results:
    an ``ActionResult`` is only meaningful if its ``request_id`` matches
    an ``ActionRequest.id`` that is still awaiting a result.
    """

    id: ActionRequestId
    application_id: ApplicationId
    action_name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    session_id: SessionId | None = None
    requested_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass(frozen=True)
class ActionResult:
    """The outcome of dispatching an ``ActionRequest``."""

    request_id: ActionRequestId
    success: bool
    message: str
    completed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


__all__ = ["ActionDefinition", "ActionRequest", "ActionResult", "ActionRisk"]
