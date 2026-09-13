"""Integration-style test for EmbodimentAdapter.

This worker is isolated: the real Open-LLM-VTuber package
(`open_llm_vtuber.agent.input_types`, `open_llm_vtuber.service_context`) and
the OpenNeuro domain models (`openneuro.domain.models.AttentionRequest`) are
owned elsewhere and not present in this sandbox. We install minimal in-test
stubs so this worker's own logic (metadata construction, empty-query
rejection, delegation to the existing proactive-turn hook) can be verified
without those packages. The final integrator should re-run this test against
the real `open_llm_vtuber.agent.input_types` module as a compatibility check.
"""
from __future__ import annotations

import asyncio
import sys
import types
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import pytest


def _install_ollv_input_types_stub() -> None:
    if "open_llm_vtuber.agent.input_types" in sys.modules:
        return

    if "open_llm_vtuber" not in sys.modules:
        top = types.ModuleType("open_llm_vtuber")
        top.__path__ = []
        sys.modules["open_llm_vtuber"] = top

    if "open_llm_vtuber.agent" not in sys.modules:
        agent_pkg = types.ModuleType("open_llm_vtuber.agent")
        agent_pkg.__path__ = []
        sys.modules["open_llm_vtuber.agent"] = agent_pkg

    module = types.ModuleType("open_llm_vtuber.agent.input_types")

    class TextSource(str, Enum):
        INPUT = "input"
        NON_HUMAN = "non_human"

    @dataclass
    class TextData:
        source: TextSource
        content: str
        from_name: str = ""

    @dataclass
    class BatchInput:
        texts: list
        images: list = field(default_factory=list)
        metadata: dict = field(default_factory=dict)

    module.TextSource = TextSource
    module.TextData = TextData
    module.BatchInput = BatchInput
    sys.modules["open_llm_vtuber.agent.input_types"] = module


_install_ollv_input_types_stub()

from openneuro.embodiment.embodiment_adapter import (  # noqa: E402
    EmbodimentAdapter,
    EmbodimentAdapterError,
)


class _FakeAttentionRequest:
    """Stand-in for openneuro.domain.models.AttentionRequest (duck-typed)."""

    def __init__(
        self,
        query: str,
        application_id: str = "github",
        ephemeral: bool = False,
        candidate_actions: list[str] | None = None,
        priority: Any = None,
        event_id: Any = None,
    ) -> None:
        self.query = query
        self.application_id = application_id
        self.ephemeral = ephemeral
        self.candidate_actions = candidate_actions or []
        self.priority = priority
        self.event_id = event_id


class _FakeServiceContext:
    """Stand-in for open_llm_vtuber.service_context.ServiceContext."""


def test_trigger_agent_turn_delegates_to_existing_proactive_path():
    captured: dict[str, Any] = {}

    async def fake_run_proactive_turn(context, batch_input):
        captured["context"] = context
        captured["batch_input"] = batch_input

    adapter = EmbodimentAdapter(run_proactive_turn=fake_run_proactive_turn)
    context = _FakeServiceContext()
    request = _FakeAttentionRequest(
        query="Should I merge the open pull request?",
        application_id="github",
        ephemeral=True,
        candidate_actions=["merge_pr", "comment_pr"],
    )

    asyncio.run(adapter.trigger_agent_turn(context, request))

    assert captured["context"] is context
    batch_input = captured["batch_input"]
    assert batch_input.texts[0].content == "Should I merge the open pull request?"
    assert batch_input.texts[0].from_name == "openneuro"

    metadata = batch_input.metadata
    assert metadata["from_openneuro"] is True
    assert metadata["source"] == "openneuro"
    assert metadata["application_id"] == "github"
    assert metadata["ephemeral"] is True
    assert metadata["candidate_actions"] == ["merge_pr", "comment_pr"]
    assert metadata["skip_human_history_entry"] is True


def test_trigger_agent_turn_rejects_empty_query():
    async def fake_run_proactive_turn(context, batch_input):  # pragma: no cover
        raise AssertionError("run_proactive_turn should not be called for an empty query")

    adapter = EmbodimentAdapter(run_proactive_turn=fake_run_proactive_turn)
    request = _FakeAttentionRequest(query="")

    with pytest.raises(EmbodimentAdapterError):
        asyncio.run(adapter.trigger_agent_turn(_FakeServiceContext(), request))


def test_no_batch_input_constructed_when_ollv_input_types_missing(monkeypatch):
    """If open_llm_vtuber isn't importable, we should fail loudly and clearly
    rather than silently no-op or crash with an unrelated traceback."""
    monkeypatch.delitem(sys.modules, "open_llm_vtuber.agent.input_types", raising=False)
    monkeypatch.delitem(sys.modules, "open_llm_vtuber.agent", raising=False)
    monkeypatch.delitem(sys.modules, "open_llm_vtuber", raising=False)

    async def fake_run_proactive_turn(context, batch_input):  # pragma: no cover
        raise AssertionError("should not be reached")

    adapter = EmbodimentAdapter(run_proactive_turn=fake_run_proactive_turn)
    request = _FakeAttentionRequest(query="hello")

    with pytest.raises(EmbodimentAdapterError):
        asyncio.run(adapter.trigger_agent_turn(_FakeServiceContext(), request))

    # restore for any later tests in the same process
    _install_ollv_input_types_stub()
