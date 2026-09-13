"""Tiny YAML-backed policy rule loader.

Config format: a YAML list of records, each with:

    application: str      # application id, or "*" for any application
    action_pattern: str    # fnmatch-style action-name pattern, or "*"
    decision: str          # one of allow / deny / require_confirmation

This loader is intentionally generic: ``application`` and ``action_pattern``
are just data fields matched against ``ctx.application_id`` /
``ctx.action_name`` at rule-evaluation time. The loader itself never
special-cases any particular application name.
"""
from __future__ import annotations

import fnmatch
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import yaml

from openneuro.policy.policy_engine import PolicyContext, PolicyDecision, PolicyRule

_VALID_OUTCOMES = {"allow", "deny", "require_confirmation"}
_REQUIRED_FIELDS = ("application", "action_pattern", "decision")


class PolicyConfigError(ValueError):
    """Raised when a policy config file is missing or malformed."""


def _make_rule(application: str, action_pattern: str, decision: str, record_index: int) -> PolicyRule:
    def rule(ctx: PolicyContext) -> Optional[PolicyDecision]:
        app_ok = application == "*" or application == ctx.application_id
        action_ok = fnmatch.fnmatch(ctx.action_name, action_pattern)
        if app_ok and action_ok:
            return PolicyDecision(
                outcome=decision,
                reason=f"matched config rule #{record_index} ({application}/{action_pattern})",
                decided_at=datetime.now(timezone.utc),
            )
        return None

    return rule


def load_rules_from_config(path: str) -> List[PolicyRule]:
    """Load an ordered list of ``PolicyRule`` callables from a YAML file.

    Fails clearly with ``PolicyConfigError`` on a missing file, invalid YAML,
    a top-level structure that isn't a list of records, missing required
    fields, or an unrecognized decision value.
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise PolicyConfigError(f"policy config not found: {path}")

    try:
        raw = yaml.safe_load(file_path.read_text())
    except yaml.YAMLError as exc:
        raise PolicyConfigError(f"invalid YAML in policy config {path}: {exc}") from exc

    if raw is None:
        return []

    if not isinstance(raw, list):
        raise PolicyConfigError(
            f"policy config {path} must be a YAML list of records, got {type(raw).__name__}"
        )

    rules: List[PolicyRule] = []
    for index, record in enumerate(raw):
        if not isinstance(record, dict):
            raise PolicyConfigError(
                f"record #{index} in {path} must be a mapping, got {type(record).__name__}"
            )

        missing = [f for f in _REQUIRED_FIELDS if f not in record]
        if missing:
            raise PolicyConfigError(
                f"record #{index} in {path} missing required field(s): {missing}"
            )

        application = record["application"]
        action_pattern = record["action_pattern"]
        decision = record["decision"]

        if decision not in _VALID_OUTCOMES:
            raise PolicyConfigError(
                f"record #{index} in {path} has invalid decision {decision!r}; "
                f"must be one of {_VALID_OUTCOMES}"
            )

        rules.append(_make_rule(application, action_pattern, decision, index))

    return rules
