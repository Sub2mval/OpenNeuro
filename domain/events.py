"""State snapshots and events published by applications."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from openneuro.domain.ids import ApplicationId, EventId
from openneuro.domain.priority import Priority


@dataclass(frozen=True)
class ApplicationState:
    """A versioned snapshot of one application's data.

    Published wholesale via ``ApplicationGateway.publish_state``; a new
    publication is a new ``ApplicationState`` instance rather than a
    mutation of an existing one, hence frozen.
    """

    application_id: ApplicationId
    data: dict[str, Any] = field(default_factory=dict)
    version: int = 0
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass(frozen=True)
class ApplicationEvent:
    """A discrete occurrence published by an application.

    ``silent`` indicates the event should not itself trigger spoken
    output; ``priority`` feeds into attention/interruption decisions
    elsewhere in the system.
    """

    event_id: EventId
    application_id: ApplicationId
    message: str
    silent: bool
    priority: Priority
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


__all__ = ["ApplicationState", "ApplicationEvent"]
