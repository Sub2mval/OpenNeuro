"""The value object used to ask the runtime for the user's attention.

Only one unresolved ``AttentionRequest`` is ever in flight at a time; a
new request replaces an old one. That behavior is enforced by whichever
component manages attention state, not by this dataclass itself.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone

from openneuro.domain.ids import ApplicationId, EventId
from openneuro.domain.priority import Priority


@dataclass(frozen=True)
class AttentionRequest:
    """A request that the agent/user attend to something.

    - ``event_id``: the triggering event, if this request originated from
      one (as opposed to, e.g., a proactive check-in).
    - ``state``: a free-form label for the requesting application's
      current interaction state (e.g. "awaiting_confirmation").
    - ``query``: the question or prompt to surface.
    - ``ephemeral``: whether this request should be dropped rather than
      queued if it cannot be served immediately.
    - ``candidate_actions``: action names relevant to resolving this
      request.
    - ``expires_at``: optional deadline after which the request is stale.
    """

    application_id: ApplicationId
    event_id: EventId | None
    priority: Priority
    state: str
    query: str
    ephemeral: bool
    candidate_actions: list[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: datetime | None = None


__all__ = ["AttentionRequest"]
