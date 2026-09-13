"""Generic inputs and outputs of the policy engine.

Nothing here encodes application-specific concepts (no per-app special
cases, no application-name branching material). ``PolicyContext`` carries
only the generic facts a policy engine needs to make a deterministic
authorization decision about one action request.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

from openneuro.domain.actions import ActionRisk
from openneuro.domain.ids import ApplicationId, SessionId

PolicyOutcome = Literal["allow", "deny", "require_confirmation"]


@dataclass(frozen=True)
class PolicyContext:
    """Everything a policy rule needs to evaluate one action request.

    - ``application_id`` / ``action_name`` / ``arguments``: identify what
      is being requested.
    - ``risk``: the declared risk tier of the action being invoked.
    - ``is_registered``: whether the action is currently a live
      registration (as opposed to a stale/unregistered name).
    - ``is_connected``: whether the owning application's connector is
      currently started/connected.
    - ``session_id``: the session the request originated from, if any.
    """

    application_id: ApplicationId
    action_name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    risk: ActionRisk = "low"
    is_registered: bool = True
    is_connected: bool = True
    session_id: SessionId | None = None


@dataclass(frozen=True)
class PolicyDecision:
    """The result of evaluating a ``PolicyContext``."""

    outcome: PolicyOutcome
    reason: str
    decided_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


__all__ = ["PolicyContext", "PolicyDecision", "PolicyOutcome"]
