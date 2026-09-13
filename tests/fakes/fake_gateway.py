"""In-process test double for the ApplicationGateway protocol.

Structural (duck-typed) implementation of:

    class ApplicationGateway(Protocol):
        def register_actions(self, actions: list[ActionDefinition]) -> None: ...
        def unregister_actions(self, action_names: list[str]) -> None: ...
        def publish_state(self, state: ApplicationState) -> None: ...
        def publish_event(self, message: str, silent: bool,
                           priority: Priority = Priority.LOW) -> EventId: ...
        def request_attention(self, request: AttentionRequest) -> None: ...
        def disconnect(self) -> None: ...

Not imported from openneuro.protocol directly (that worker's module may be
absent in this sandbox) -- this class simply satisfies the shape.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from . import ActionDefinition, ApplicationState, AttentionRequest, EventId, Priority


@dataclass
class RecordedCall:
    """A single call made against a FakeGateway, for test assertions."""

    name: str
    args: Dict[str, Any] = field(default_factory=dict)


class FakeGateway:
    """Records every ApplicationGateway call so tests can assert on them
    without a real runtime/policy/attention stack.
    """

    def __init__(self) -> None:
        self.calls: List[RecordedCall] = []
        self.registered_actions: Dict[str, ActionDefinition] = {}
        self.published_states: List[ApplicationState] = []
        self.published_events: List[Dict[str, Any]] = []
        self.attention_requests: List[AttentionRequest] = []
        self.disconnected: bool = False
        self._event_ids = itertools.count(1)

    def register_actions(self, actions: List[ActionDefinition]) -> None:
        self.calls.append(RecordedCall("register_actions", {"actions": list(actions)}))
        for action in actions:
            # Registering an existing action name replaces it (core semantics).
            self.registered_actions[action.name] = action

    def unregister_actions(self, action_names: List[str]) -> None:
        self.calls.append(RecordedCall("unregister_actions", {"action_names": list(action_names)}))
        for name in action_names:
            self.registered_actions.pop(name, None)

    def publish_state(self, state: ApplicationState) -> None:
        self.calls.append(RecordedCall("publish_state", {"state": state}))
        self.published_states.append(state)

    def publish_event(self, message: str, silent: bool, priority: Priority = Priority.LOW) -> EventId:
        event_id = EventId(f"evt-{next(self._event_ids)}")
        self.calls.append(
            RecordedCall(
                "publish_event",
                {"message": message, "silent": silent, "priority": priority, "event_id": event_id},
            )
        )
        self.published_events.append(
            {"event_id": event_id, "message": message, "silent": silent, "priority": priority}
        )
        return event_id

    def request_attention(self, request: AttentionRequest) -> None:
        self.calls.append(RecordedCall("request_attention", {"request": request}))
        # Contract: only one unresolved attention request is in flight; a new
        # one replaces the old one. Enforcing the replacement is the real
        # attention/runtime worker's job -- this fake just records history
        # and exposes the latest one for assertions.
        self.attention_requests.append(request)

    def disconnect(self) -> None:
        self.calls.append(RecordedCall("disconnect"))
        self.disconnected = True

    # -- assertion helpers ----------------------------------------------------

    def call_names(self) -> List[str]:
        return [call.name for call in self.calls]

    @property
    def latest_attention_request(self) -> Optional[AttentionRequest]:
        return self.attention_requests[-1] if self.attention_requests else None
