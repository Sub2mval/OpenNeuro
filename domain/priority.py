"""Priority levels used for events and attention requests."""

from enum import Enum


class Priority(str, Enum):
    """Ordinal urgency level.

    Inherits from ``str`` so instances compare equal to their plain-string
    value (``Priority.LOW == "low"``) and serialize cleanly with
    ``json.dumps`` without a custom encoder.
    """

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


_RANK: dict[Priority, int] = {
    Priority.LOW: 0,
    Priority.MEDIUM: 1,
    Priority.HIGH: 2,
    Priority.CRITICAL: 3,
}


def priority_rank(priority: Priority) -> int:
    """Return an integer rank for ``priority`` such that a higher rank
    means a more urgent priority.

    Pure function, no I/O: ``priority_rank(Priority.LOW) < priority_rank(Priority.CRITICAL)``.
    """

    return _RANK[priority]


__all__ = ["Priority", "priority_rank"]
