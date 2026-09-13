"""Unit tests for OpenNeuroSettings and TraceRecorder.

This worker is isolated: `openneuro.domain.models` (TraceEvent, TraceEventKind)
is owned by the 01_domain_models worker and is not present in this sandbox.
We install a minimal in-test stub module before importing the trace recorder,
per this worker's contract ("use small in-test stubs/TYPE_CHECKING imports
where needed"). This is test-only scaffolding, not a second production
implementation of the domain models.
"""
from __future__ import annotations

import sys
import types
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any

import pytest


def _install_domain_models_stub() -> None:
    if "openneuro.domain.models" in sys.modules:
        return

    if "openneuro" not in sys.modules:
        top = types.ModuleType("openneuro")
        top.__path__ = []  # mark as (namespace) package
        sys.modules["openneuro"] = top

    if "openneuro.domain" not in sys.modules:
        domain_pkg = types.ModuleType("openneuro.domain")
        domain_pkg.__path__ = []
        sys.modules["openneuro.domain"] = domain_pkg

    models = types.ModuleType("openneuro.domain.models")

    class TraceEventKind(str, Enum):
        event_received = "event_received"
        attention_decision = "attention_decision"
        turn_started = "turn_started"
        action_proposed = "action_proposed"
        policy_decision = "policy_decision"
        action_executed = "action_executed"
        action_timed_out = "action_timed_out"
        interrupt = "interrupt"
        turn_completed = "turn_completed"

    @dataclass
    class TraceEvent:
        trace_id: str
        ts: datetime
        kind: Any
        payload: dict

    models.TraceEvent = TraceEvent
    models.TraceEventKind = TraceEventKind
    sys.modules["openneuro.domain.models"] = models


_install_domain_models_stub()

from openneuro.config.settings import OpenNeuroSettings  # noqa: E402
from openneuro.observability.trace_recorder import TraceRecorder, redact  # noqa: E402


# ---------------------------------------------------------------------------
# OpenNeuroSettings
# ---------------------------------------------------------------------------

_ALL_ENV_VARS = [
    "OPENNEURO_MODEL",
    "OPENNEURO_LOG_LEVEL",
    "OPENNEURO_ACTION_TIMEOUT",
    "OPENNEURO_MAX_TOOL_TURNS",
    "OPENNEURO_ATTENTION_QUEUE_MAX",
    "OPENNEURO_STARVATION_AGE_SECONDS",
    "OPENNEURO_POLICY_MODE",
    "OPENNEURO_POLICY_CONFIG_PATH",
    "OPENNEURO_TRACE_PATH",
    "OPENNEURO_SIM_TICK_SECONDS",
    "OPENNEURO_GITHUB_TOKEN",
]


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for var in _ALL_ENV_VARS:
        monkeypatch.delenv(var, raising=False)


def test_settings_defaults():
    settings = OpenNeuroSettings(_env_file=None)
    assert settings.model == ""
    assert settings.log_level == "INFO"
    assert settings.action_timeout == 20
    assert settings.max_tool_turns == 4
    assert settings.attention_queue_max == 50
    assert settings.starvation_age_seconds == 30
    assert settings.trace_path == "./openneuro_trace.jsonl"
    assert settings.sim_tick_seconds == 2
    assert settings.github_token is None


def test_settings_reads_environment(monkeypatch):
    monkeypatch.setenv("OPENNEURO_MODEL", "claude-sonnet-4-6")
    monkeypatch.setenv("OPENNEURO_ACTION_TIMEOUT", "5")
    monkeypatch.setenv("OPENNEURO_MAX_TOOL_TURNS", "2")
    monkeypatch.setenv("OPENNEURO_TRACE_PATH", "/tmp/trace.jsonl")
    monkeypatch.setenv("OPENNEURO_GITHUB_TOKEN", "ghp_supersecretvalue")

    settings = OpenNeuroSettings(_env_file=None)

    assert settings.model == "claude-sonnet-4-6"
    assert settings.action_timeout == 5
    assert settings.max_tool_turns == 2
    assert settings.trace_path == "/tmp/trace.jsonl"
    assert settings.github_token is not None
    assert settings.github_token.get_secret_value() == "ghp_supersecretvalue"
    # Secret must never render in repr/str.
    assert "ghp_supersecretvalue" not in repr(settings.github_token)
    assert "ghp_supersecretvalue" not in str(settings)


# ---------------------------------------------------------------------------
# redact()
# ---------------------------------------------------------------------------


def test_redact_flat_and_nested_secret_keys():
    payload = {
        "token": "abc123",
        "nested": {"api_key": "sk-xyz", "safe": "ok"},
        "list_of_dicts": [{"password": "hunter2"}, {"safe": "fine"}],
        "plain": "unchanged",
    }
    out = redact(payload)
    assert out["token"] == "***REDACTED***"
    assert out["nested"]["api_key"] == "***REDACTED***"
    assert out["nested"]["safe"] == "ok"
    assert out["list_of_dicts"][0]["password"] == "***REDACTED***"
    assert out["list_of_dicts"][1]["safe"] == "fine"
    assert out["plain"] == "unchanged"


# ---------------------------------------------------------------------------
# TraceRecorder
# ---------------------------------------------------------------------------


def test_trace_recorder_round_trip_and_isolation_by_trace_id(tmp_path):
    trace_path = tmp_path / "nested" / "trace.jsonl"
    recorder = TraceRecorder(str(trace_path))

    recorder.record("turn_started", {"foo": "bar"}, trace_id="t1")
    recorder.record("action_proposed", {"api_key": "should-not-appear"}, trace_id="t1")
    recorder.record("turn_started", {"foo": "other"}, trace_id="t2")

    t1_events = recorder.events_for("t1")
    assert [e.kind for e in t1_events] == ["turn_started", "action_proposed"]
    assert t1_events[0].payload == {"foo": "bar"}
    assert t1_events[1].payload == {"api_key": "***REDACTED***"}
    assert all(e.trace_id == "t1" for e in t1_events)

    t2_events = recorder.events_for("t2")
    assert len(t2_events) == 1
    assert t2_events[0].payload == {"foo": "other"}

    assert recorder.events_for("does-not-exist") == []


def test_trace_recorder_never_writes_secret_values_to_disk(tmp_path):
    trace_path = tmp_path / "trace.jsonl"
    recorder = TraceRecorder(str(trace_path))

    recorder.record(
        "policy_decision",
        {"github_token": "ghp_leak_me_not", "nested": {"secret": "also-hidden"}},
        trace_id="t1",
    )

    raw = trace_path.read_text(encoding="utf-8")
    assert "ghp_leak_me_not" not in raw
    assert "also-hidden" not in raw
    assert "***REDACTED***" in raw


def test_trace_recorder_appends_across_instances(tmp_path):
    trace_path = tmp_path / "trace.jsonl"
    TraceRecorder(str(trace_path)).record("turn_started", {}, trace_id="t1")
    TraceRecorder(str(trace_path)).record("turn_completed", {}, trace_id="t1")

    events = TraceRecorder(str(trace_path)).events_for("t1")
    assert [e.kind for e in events] == ["turn_started", "turn_completed"]
