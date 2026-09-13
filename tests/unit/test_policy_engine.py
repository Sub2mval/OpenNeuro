"""Unit tests for the OpenNeuro policy engine.

Standalone: does not depend on any other OpenNeuro worker's output. Uses the
canonical ``PolicyContext``/``PolicyDecision`` (or their local fallback
stand-ins) as exported from ``openneuro.policy.policy_engine``.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from openneuro.policy.policy_engine import PolicyContext, PolicyDecision, PolicyEngine
from openneuro.policy.rule_loader import PolicyConfigError, load_rules_from_config
from openneuro.policy.rules import confirm_high_risk, deny_unregistered_action


def make_ctx(**overrides):
    fields = dict(
        application_id="app-1",
        action_name="do_thing",
        arguments={},
        risk="low",
        is_registered=True,
        is_connected=True,
    )
    fields.update(overrides)
    return PolicyContext(**fields)


def test_first_match_wins():
    def rule_a(ctx):
        return PolicyDecision(outcome="allow", reason="rule_a", decided_at=datetime.now(timezone.utc))

    def rule_b(ctx):
        return PolicyDecision(outcome="deny", reason="rule_b", decided_at=datetime.now(timezone.utc))

    engine = PolicyEngine([rule_a, rule_b], default_outcome="deny")
    decision = engine.evaluate(make_ctx())
    assert decision.outcome == "allow"
    assert decision.reason == "rule_a"


def test_default_outcome_when_no_rule_matches():
    engine = PolicyEngine([lambda ctx: None], default_outcome="deny")
    decision = engine.evaluate(make_ctx())
    assert decision.outcome == "deny"

    engine_allow = PolicyEngine([], default_outcome="allow")
    assert engine_allow.evaluate(make_ctx()).outcome == "allow"


def test_invalid_default_outcome_rejected():
    with pytest.raises(ValueError):
        PolicyEngine([], default_outcome="not-a-real-outcome")


def test_deny_unregistered_action():
    engine = PolicyEngine([deny_unregistered_action], default_outcome="allow")

    decision = engine.evaluate(make_ctx(is_registered=False))
    assert decision.outcome == "deny"

    decision2 = engine.evaluate(make_ctx(is_registered=True))
    assert decision2.outcome == "allow"  # falls through to default


def test_confirm_high_risk():
    engine = PolicyEngine([deny_unregistered_action, confirm_high_risk], default_outcome="allow")

    decision = engine.evaluate(make_ctx(risk="high"))
    assert decision.outcome == "require_confirmation"

    decision2 = engine.evaluate(make_ctx(risk="low"))
    assert decision2.outcome == "allow"


def test_unregistered_denial_precedes_high_risk_confirmation():
    # First-match semantics: an unregistered, high-risk action is denied,
    # not sent to confirmation, because deny_unregistered_action runs first.
    engine = PolicyEngine([deny_unregistered_action, confirm_high_risk], default_outcome="allow")
    decision = engine.evaluate(make_ctx(is_registered=False, risk="high"))
    assert decision.outcome == "deny"


def test_no_hardcoded_application_names_in_policy_source():
    forbidden = {"github", "discord", "browser", "simulation"}
    policy_dir = Path(__file__).resolve().parents[2] / "openneuro" / "policy"
    for py_file in policy_dir.glob("*.py"):
        source = py_file.read_text().lower()
        for name in forbidden:
            assert name not in source, f"found hard-coded application name {name!r} in {py_file}"


def test_load_rules_from_config(tmp_path):
    config = tmp_path / "policy.yaml"
    config.write_text(
        """
- application: "*"
  action_pattern: "delete_*"
  decision: deny
- application: app-1
  action_pattern: "*"
  decision: allow
"""
    )
    rules = load_rules_from_config(str(config))
    engine = PolicyEngine(rules, default_outcome="deny")

    denied = engine.evaluate(make_ctx(action_name="delete_everything"))
    assert denied.outcome == "deny"

    allowed = engine.evaluate(make_ctx(application_id="app-1", action_name="send_message"))
    assert allowed.outcome == "allow"

    other_app = engine.evaluate(make_ctx(application_id="app-2", action_name="send_message"))
    assert other_app.outcome == "deny"  # no matching record -> default


def test_load_rules_empty_config(tmp_path):
    config = tmp_path / "empty.yaml"
    config.write_text("")
    assert load_rules_from_config(str(config)) == []


def test_load_rules_missing_file():
    with pytest.raises(PolicyConfigError):
        load_rules_from_config("/nonexistent/path/policy.yaml")


def test_load_rules_not_a_list(tmp_path):
    config = tmp_path / "bad_shape.yaml"
    config.write_text("application: app-1\ndecision: allow\n")
    with pytest.raises(PolicyConfigError):
        load_rules_from_config(str(config))


def test_load_rules_malformed_config_missing_field(tmp_path):
    config = tmp_path / "bad.yaml"
    config.write_text("- application: app-1\n  decision: allow\n")  # missing action_pattern
    with pytest.raises(PolicyConfigError):
        load_rules_from_config(str(config))


def test_load_rules_invalid_decision_value(tmp_path):
    config = tmp_path / "bad2.yaml"
    config.write_text("- application: app-1\n  action_pattern: '*'\n  decision: maybe\n")
    with pytest.raises(PolicyConfigError):
        load_rules_from_config(str(config))
