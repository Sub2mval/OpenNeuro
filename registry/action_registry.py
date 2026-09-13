"""Dynamic, in-memory action registry.

This module has no dependency on any concrete application connector and
does not branch on application identity. It only manipulates
`ActionDefinition`-shaped objects (application_id, name, ...) supplied by
callers.

The canonical `ActionDefinition` / `ApplicationId` types are defined by the
`01_domain_models` worker (expected at `openneuro.domain.models`). That
module is not assumed to exist in this isolated sandbox, so it is only
imported for static type checking. At runtime this module relies purely on
duck typing: any object with `.application_id` and `.name` attributes is
accepted.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional

if TYPE_CHECKING:  # pragma: no cover - import only used for type hints
    from openneuro.domain.models import ActionDefinition, ApplicationId


class ActionRegistry:
    """In-memory registry of dynamically (un)registered actions.

    Actions are scoped per `application_id`. Registering an action with the
    same `application_id` and `name` as an existing entry replaces it rather
    than duplicating it.
    """

    def __init__(self) -> None:
        # application_id -> {action_name -> ActionDefinition}
        self._actions_by_app: "Dict[ApplicationId, Dict[str, ActionDefinition]]" = {}

    def register(self, actions: "List[ActionDefinition]") -> None:
        """Register (or replace) a batch of actions.

        Each action is scoped to its own `application_id`. If an action with
        the same `application_id` and `name` is already registered, it is
        replaced in place.
        """
        for action in actions:
            per_app = self._actions_by_app.setdefault(action.application_id, {})
            per_app[action.name] = action

    def unregister(
        self, application_id: "ApplicationId", action_names: "List[str]"
    ) -> None:
        """Remove the named actions for a given application.

        Unregistering a name that isn't currently registered (for this
        application, or because the application has no registered actions
        at all) is a silent no-op.
        """
        per_app = self._actions_by_app.get(application_id)
        if per_app is None:
            return

        for name in action_names:
            per_app.pop(name, None)

        if not per_app:
            # Keep the registry tidy; an empty per-app map is equivalent to
            # the application never having registered anything.
            self._actions_by_app.pop(application_id, None)

    def available_actions(
        self, application_ids: "Optional[List[ApplicationId]]" = None
    ) -> "List[ActionDefinition]":
        """Return currently registered actions.

        With `application_ids=None`, returns all registered actions across
        all applications. Otherwise, returns only the registered actions
        belonging to the given applications (unknown application ids simply
        contribute no actions).
        """
        if application_ids is None:
            result: "List[ActionDefinition]" = []
            for per_app in self._actions_by_app.values():
                result.extend(per_app.values())
            return result

        result = []
        for application_id in application_ids:
            per_app = self._actions_by_app.get(application_id)
            if per_app:
                result.extend(per_app.values())
        return result
