"""Routes ApplicationEvent/ApplicationState into state storage and attention.

EventRouter owns no policy decisions, performs no external dispatch, and
never imports or branches on application-specific code. It depends on the
canonical domain/protocol types only for static typing (`TYPE_CHECKING`),
and on two small injected collaborators at runtime:

- a state store exposing `upsert(state) -> None`
  (the concrete `StateStore` in this package satisfies this)
- an `AttentionManagerProtocol` exposing `submit(request) -> None`
  (the real AttentionManager from 04_attention.md is expected to satisfy
  this; see ASSUMPTIONS in the worker report for the exact method name)

Because this is an isolated worker sandbox, EventRouter never imports the
canonical domain module at runtime to build an AttentionRequest. Instead,
the caller injects an `attention_request_factory` callable that knows how
to build one from an ApplicationEvent. This keeps EventRouter fully
testable with local stubs and avoids a hard runtime dependency on a
module (`openneuro.domain`) that another worker owns.
"""

from __future__ import annotations

from collections import deque
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Deque,
    List,
    Protocol,
    runtime_checkable,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from openneuro.domain import ApplicationEvent, ApplicationState, AttentionRequest


@runtime_checkable
class AttentionManagerProtocol(Protocol):
    """Minimal contract EventRouter depends on.

    ASSUMPTION: per 04_attention.md, only one unresolved attention request
    is in flight at a time and a new one replaces the old one. That
    replacement behavior belongs to the real AttentionManager
    implementation, not to EventRouter, which only ever calls `submit`.
    """

    def submit(self, request: "AttentionRequest") -> None: ...


@runtime_checkable
class StateStoreProtocol(Protocol):
    """Minimal contract EventRouter depends on for state persistence."""

    def upsert(self, state: "ApplicationState") -> None: ...


def _default_attention_request_factory(evt: "ApplicationEvent") -> "AttentionRequest":
    """Build a canonical AttentionRequest from a non-silent event.

    Imported lazily so that constructing an EventRouter (or running its
    unit tests) never requires `openneuro.domain` to exist. This default
    is only exercised once the real domain module is available, e.g. at
    integration time.
    """
    from openneuro.domain import AttentionRequest  # local import by design

    return AttentionRequest(
        application_id=evt.application_id,
        event_id=evt.event_id,
        priority=evt.priority,
        state="event_received",
        query=evt.message,
        ephemeral=False,
        candidate_actions=[],
        created_at=evt.created_at,
        expires_at=None,
    )


class EventRouter:
    """Routes application events/state to storage and attention.

    - `on_state` forwards to the injected state store's `upsert`.
    - Silent events (`silent=True`) are never sent to attention; they are
      kept in a small bounded FIFO buffer for the next AgentObservation.
    - Non-silent events are turned into an AttentionRequest (via the
      injected factory) and submitted to the injected AttentionManager.
    """

    def __init__(
        self,
        state_store: StateStoreProtocol,
        attention_manager: AttentionManagerProtocol,
        attention_request_factory: Callable[
            ["ApplicationEvent"], "AttentionRequest"
        ] = _default_attention_request_factory,
        silent_buffer_size: int = 20,
    ) -> None:
        self._state_store = state_store
        self._attention_manager = attention_manager
        self._attention_request_factory = attention_request_factory
        self._silent_events: "Deque[ApplicationEvent]" = deque(maxlen=silent_buffer_size)

    def on_state(self, state: "ApplicationState") -> None:
        """Forward published application state to the state store."""
        self._state_store.upsert(state)

    def on_event(self, evt: "ApplicationEvent") -> None:
        """Route an application event to buffering or attention."""
        if evt.silent:
            self._silent_events.append(evt)
            return
        request = self._attention_request_factory(evt)
        self._attention_manager.submit(request)

    def silent_context(self) -> "List[Any]":
        """Return a copy of buffered silent events for AgentObservation.

        A list copy is returned so callers cannot mutate the internal
        buffer via the returned value.
        """
        return list(self._silent_events)
