"""Opaque identifier types shared across OpenNeuro.

Each ID is a ``typing.NewType`` over ``str``. NewType gives static type
checkers a way to catch accidental mixing of, say, an ApplicationId where
a SessionId was expected, while remaining a plain string at runtime (so
these values serialize, hash, and compare exactly like ordinary strings).

No business logic or validation lives here by design.
"""

from typing import NewType

ApplicationId = NewType("ApplicationId", str)
SessionId = NewType("SessionId", str)
EventId = NewType("EventId", str)
ActionRequestId = NewType("ActionRequestId", str)

__all__ = [
    "ApplicationId",
    "SessionId",
    "EventId",
    "ActionRequestId",
]
