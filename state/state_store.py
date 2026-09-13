"""In-memory store for the latest ApplicationState per application.

This module deliberately has no import-time dependency on the canonical
OpenNeuro domain module (see 01_domain_models.md), since that worker's
output is not guaranteed to exist in this sandbox. Domain types are only
referenced for static typing, under `TYPE_CHECKING`. At runtime the store
treats `state` as any object exposing an `application_id` attribute
(duck typing), which is compatible with the canonical ApplicationState
dataclass once it exists.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Optional

if TYPE_CHECKING:  # pragma: no cover - typing only
    from openneuro.domain import ApplicationId, ApplicationState


class StateStore:
    """Holds the most recently published ApplicationState per application.

    Semantics:
    - `upsert` replaces any existing state for `state.application_id`
      wholesale; it never merges into the previous `state.data`.
    - `all()` returns a snapshot copy, so mutating the returned mapping
      (or reassigning entries) cannot corrupt internal storage.
    """

    def __init__(self) -> None:
        self._states: "Dict[ApplicationId, ApplicationState]" = {}

    def upsert(self, state: "ApplicationState") -> None:
        """Replace whatever is stored for `state.application_id`."""
        self._states[state.application_id] = state

    def get(self, application_id: "ApplicationId") -> "Optional[ApplicationState]":
        """Return the latest known state for `application_id`, or None."""
        return self._states.get(application_id)

    def all(self) -> "Dict[ApplicationId, ApplicationState]":
        """Return a safe snapshot copy of all known application states.

        The returned dict is a new mapping; the ApplicationState values
        inside it are expected to be immutable per the canonical contract,
        so a shallow copy is sufficient to protect internal storage from
        accidental mutation via the returned mapping's keys/values.
        """
        return dict(self._states)
