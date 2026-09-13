"""Value objects at the boundary between the runtime and the agent."""

from dataclasses import dataclass, field
from enum import Enum

from openneuro.domain.actions import ActionDefinition, ActionRequest
from openneuro.domain.events import ApplicationState
from openneuro.domain.ids import ApplicationId


@dataclass(frozen=True)
class AgentObservation:
    """Everything the agent is given to make one decision.

    ``state_snapshot`` maps each known application to its latest
    published state; ``pending_query`` is the resolved attention query
    (if any) the agent should address; ``available_actions`` is the
    current registry contents visible to the agent.
    """

    state_snapshot: dict[ApplicationId, ApplicationState] = field(
        default_factory=dict
    )
    pending_query: str | None = None
    available_actions: list[ActionDefinition] = field(default_factory=list)


@dataclass(frozen=True)
class AgentDecision:
    """What the agent produced from one ``AgentObservation``.

    ``message_to_say`` is optional spoken output; ``action_requests`` are
    zero or more actions the agent wants dispatched as a result.
    """

    message_to_say: str | None = None
    action_requests: list[ActionRequest] = field(default_factory=list)


class AgentStatus(str, Enum):
    """The agent's current high-level activity, for UI/state reporting."""

    IDLE = "idle"
    THINKING = "thinking"
    SPEAKING = "speaking"
    EXECUTING_ACTION = "executing_action"
    WAITING_FOR_RESULT = "waiting_for_result"


__all__ = ["AgentObservation", "AgentDecision", "AgentStatus"]
