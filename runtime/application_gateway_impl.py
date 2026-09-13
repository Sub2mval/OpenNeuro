"""ApplicationGatewayImpl: the runtime-side ApplicationGateway from
09_runtime.md.

One instance is constructed per connected application and handed to that
application's ``ApplicationConnector.start(gateway)``. It carries only the
``application_id`` it was bound to and delegates every call to injected
collaborators -- it never branches on *which* application it is, so core
code stays decoupled from GitHub/Discord/Browser/Simulation by name.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Callable, Optional

from openneuro.runtime.agent_runtime import (
    ActionDefinition,
    ActionRegistryProtocol,
    ApplicationEvent,
    ApplicationId,
    ApplicationState,
    AttentionManagerProtocol,
    AttentionRequest,
    EventId,
    EventRouterProtocol,
    Priority,
)


class ApplicationGatewayImpl:
    """Concrete ``ApplicationGateway`` bound to a single application_id."""

    def __init__(
        self,
        application_id: ApplicationId,
        action_registry: ActionRegistryProtocol,
        event_router: EventRouterProtocol,
        attention_manager: AttentionManagerProtocol,
        disconnect_callback: Optional[Callable[[ApplicationId], None]] = None,
    ) -> None:
        self._application_id = application_id
        self._action_registry = action_registry
        self._event_router = event_router
        self._attention_manager = attention_manager
        self._disconnect_callback = disconnect_callback

    def register_actions(self, actions: list[ActionDefinition]) -> None:
        self._action_registry.register_actions(self._application_id, actions)

    def unregister_actions(self, action_names: list[str]) -> None:
        self._action_registry.unregister_actions(self._application_id, action_names)

    def publish_state(self, state: ApplicationState) -> None:
        self._event_router.publish_state(state)

    def publish_event(
        self,
        message: str,
        silent: bool,
        priority: Priority = Priority.LOW,
    ) -> EventId:
        """Create an ApplicationEvent for this application and route it.

        Note (integration hazard): per the 09_runtime contract, GatewayImpl
        routes directly through the EventRouter rather than through
        ``AgentRuntime.handle_application_event``, so events originated via
        this method are not individually wrapped in an "event_received"
        trace the way externally-injected events are. If uniform tracing of
        every event (regardless of origin) turns out to be required, the
        integrator should either have this method call back into
        AgentRuntime, or move "event_received" tracing into the EventRouter
        itself.
        """
        event_id = EventId(str(uuid.uuid4()))
        event = ApplicationEvent(
            event_id=event_id,
            application_id=self._application_id,
            message=message,
            silent=silent,
            priority=priority,
            created_at=datetime.now(timezone.utc),
        )
        self._event_router.route_event(event)
        return event_id

    def request_attention(self, request: AttentionRequest) -> None:
        self._attention_manager.request_attention(request)

    def disconnect(self) -> None:
        if self._disconnect_callback is not None:
            self._disconnect_callback(self._application_id)
