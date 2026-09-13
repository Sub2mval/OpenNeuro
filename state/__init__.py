"""Application state store and event routing.

Public surface for this OpenNeuro slice: `StateStore` (holds the latest
ApplicationState per application) and `EventRouter` (routes
ApplicationEvent/ApplicationState into storage and attention).
"""

from .event_router import AttentionManagerProtocol, EventRouter
from .state_store import StateStore

__all__ = ["StateStore", "EventRouter", "AttentionManagerProtocol"]
