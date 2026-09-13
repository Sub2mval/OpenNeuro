"""Generic policy engine.

Evaluates a ``PolicyContext`` against an ordered list of ``PolicyRule``
callables and returns the first non-``None`` ``PolicyDecision``, falling back
to a configured default outcome when no rule matches.

This module intentionally contains no application-specific logic: rules are
plain functions/data, and the engine never branches on an application id or
name. The policy boundary is meant to run BEFORE any dispatch to an
ApplicationConnector.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, List, Literal, Optional

try:
    # Canonical domain contracts (see 01_domain_models.md in the OpenNeuro
    # architecture plan). Imported this way so that once the domain-models
    # worker's output is merged into the final integration, this module
    # picks up the real, shared types automatically with no code changes.
    from openneuro.domain.models import PolicyContext, PolicyDecision  # type: ignore
except ImportError:  # pragma: no cover - exercised only in isolated sandboxes
    @dataclass(frozen=True)
    class PolicyContext:  # type: ignore[no-redef]
        """Standalone stand-in for the canonical ``PolicyContext``.

        Mirrors the fields described in the architecture contract: the
        application id, action name, arguments, the action's declared risk,
        and registration/connection facts. This class is only used when the
        real domain-models module is not present in the current sandbox; the
        real one takes precedence automatically once available on the import
        path.
        """

        application_id: str
        action_name: str
        arguments: dict = field(default_factory=dict)
        risk: Literal["low", "medium", "high"] = "low"
        is_registered: bool = True
        is_connected: bool = True
        extra: dict = field(default_factory=dict)

    @dataclass(frozen=True)
    class PolicyDecision:  # type: ignore[no-redef]
        """Standalone stand-in for the canonical ``PolicyDecision``."""

        outcome: Literal["allow", "deny", "require_confirmation"]
        reason: str
        decided_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


PolicyRule = Callable[[PolicyContext], Optional[PolicyDecision]]

_VALID_OUTCOMES = {"allow", "deny", "require_confirmation"}


class PolicyEngine:
    """Evaluates rules in order; the first non-None decision wins.

    If no rule produces a decision, ``default_outcome`` is used. This class
    holds no application-specific logic and never branches on application
    identity.
    """

    def __init__(self, rules: List[PolicyRule], default_outcome: str = "deny") -> None:
        if default_outcome not in _VALID_OUTCOMES:
            raise ValueError(
                f"default_outcome must be one of {_VALID_OUTCOMES}, got {default_outcome!r}"
            )
        self._rules = list(rules)
        self._default_outcome = default_outcome

    def evaluate(self, ctx: PolicyContext) -> PolicyDecision:
        for rule in self._rules:
            decision = rule(ctx)
            if decision is not None:
                return decision
        return PolicyDecision(
            outcome=self._default_outcome,
            reason="no policy rule matched; applying default outcome",
            decided_at=datetime.now(timezone.utc),
        )
