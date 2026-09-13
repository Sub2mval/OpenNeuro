"""Generic, reusable policy rules.

None of these rules branch on application identity or action-name string
matching against a specific app; they only look at generic ``PolicyContext``
fields such as registration status and declared risk.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from openneuro.policy.policy_engine import PolicyContext, PolicyDecision


def deny_unregistered_action(ctx: PolicyContext) -> Optional[PolicyDecision]:
    """Deny any action that is not currently a registered action.

    Relies on ``ctx.is_registered`` (a generic registration fact populated by
    the runtime/action registry), never on application id or action-name
    string matching.
    """
    if not getattr(ctx, "is_registered", True):
        return PolicyDecision(
            outcome="deny",
            reason="action is not a registered action",
            decided_at=datetime.now(timezone.utc),
        )
    return None


def confirm_high_risk(ctx: PolicyContext) -> Optional[PolicyDecision]:
    """Require confirmation for actions whose declared risk is "high".

    Only fires when no earlier rule has already decided the context, per the
    engine's first-match semantics.
    """
    if getattr(ctx, "risk", "low") == "high":
        return PolicyDecision(
            outcome="require_confirmation",
            reason="action is high risk and requires confirmation",
            decided_at=datetime.now(timezone.utc),
        )
    return None
