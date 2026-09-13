"""Integration test: OpenNeuroAgentAdapter driven by a FakeLLM.

Proves, in a single turn:
  * an allowed action is dispatched to the (fake) connector and its real
    result is fed back to the model as a tool result,
  * a denied action is represented to the model as a tool result but is
    NEVER sent to the connector,
  * the loop terminates cleanly once the model stops requesting tools.

All domain/runtime types used here are local stubs (duck-typed), since the
canonical openneuro.domain.models / runtime modules belong to other workers
and are not assumed to exist in this sandbox.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any, Dict, List

import pytest

from openneuro.agent_adapter.openneuro_agent_adapter import OpenNeuroAgentAdapter


# --- local stub domain objects -------------------------------------------------


@dataclass
class StubActionDefinition:
    application_id: str
    name: str
    description: str
    schema: Dict[str, Any] = field(default_factory=dict)
    risk: str = "low"


@dataclass
class StubActionResult:
    request_id: Any
    success: bool
    message: str
    completed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class StubAgentObservation:
    state_snapshot: Dict[str, Any]
    pending_query: Any
    available_actions: List[StubActionDefinition]


# --- fakes ----------------------------------------------------------------------


class FakeConnector:
    """Records every dispatch it is asked to perform. A denied action must
    never show up here."""

    def __init__(self):
        self.dispatch_calls: List[Any] = []

    async def dispatch_action(self, request):
        self.dispatch_calls.append(request)
        return StubActionResult(
            request_id=request.id, success=True, message="did the thing"
        )


class FakeRuntime:
    """Stands in for worker 09's AgentRuntime: owns policy evaluation and
    only forwards to the connector when allowed."""

    def __init__(self, connector: FakeConnector, actions: List[StubActionDefinition]):
        self._connector = connector
        self._actions = actions
        self.evaluated: List[str] = []

    async def get_observation(self):
        return StubAgentObservation(
            state_snapshot={}, pending_query=None, available_actions=self._actions
        )

    async def evaluate_and_execute(self, request):
        self.evaluated.append(request.action_name)
        if request.action_name == "denied_action":
            # Policy denial: represented to the model, but the connector is
            # never touched.
            return StubActionResult(
                request_id=request.id,
                success=False,
                message="denied: insufficient risk clearance",
            )
        return await self._connector.dispatch_action(request)


class FakeLLM:
    """Yields a scripted sequence of turns. Turn 1 requests two tool calls;
    turn 2 (after tool results are fed back) just finishes with text."""

    def __init__(self):
        self.calls: List[List[Dict[str, Any]]] = []

    async def chat_completion(self, messages, tools=None):
        self.calls.append(messages)
        turn = len(self.calls)
        if turn == 1:
            async for event in self._turn_one_events():
                yield event
        else:
            async for event in self._final_turn_events():
                yield event

    async def _turn_one_events(self):
        for e in [
            {"type": "text_delta", "text": "Let me check that."},
            {
                "type": "tool_use_complete",
                "id": "call-1",
                "name": "allowed_action",
                "input": {"target": "widget"},
            },
            {
                "type": "tool_use_complete",
                "id": "call-2",
                "name": "denied_action",
                "input": {"target": "secret"},
            },
            {"type": "message_stop"},
        ]:
            yield e

    async def _final_turn_events(self):
        for e in [
            {"type": "text_delta", "text": "Done: one worked, one was blocked."},
            {"type": "message_stop"},
        ]:
            yield e


# --- test -------------------------------------------------------------------


@pytest.mark.asyncio
async def test_allowed_and_denied_action_in_single_turn():
    connector = FakeConnector()
    actions = [
        StubActionDefinition(
            application_id="app-1",
            name="allowed_action",
            description="An action the policy permits",
            schema={"type": "object", "properties": {}},
        ),
        StubActionDefinition(
            application_id="app-1",
            name="denied_action",
            description="An action the policy forbids",
            schema={"type": "object", "properties": {}},
        ),
    ]
    runtime = FakeRuntime(connector, actions)
    llm = FakeLLM()
    adapter = OpenNeuroAgentAdapter(runtime, llm, session_id="sess-1")

    input_data = SimpleNamespace(text="please check the widget and the secret")

    outputs = []
    async for sentence in adapter.chat(input_data):
        outputs.append(sentence.display_text)

    # Both actions were evaluated by the runtime (policy boundary sits there).
    assert runtime.evaluated == ["allowed_action", "denied_action"]

    # Only the allowed action ever reached the connector.
    assert len(connector.dispatch_calls) == 1
    assert connector.dispatch_calls[0].action_name == "allowed_action"

    # The denied action's name never appears anywhere the connector saw.
    dispatched_names = [r.action_name for r in connector.dispatch_calls]
    assert "denied_action" not in dispatched_names

    # The adapter still produced text output across both turns.
    assert any("Let me check" in o for o in outputs)
    assert any("Done" in o for o in outputs)

    # Two LLM turns: first requests tools, second is the wrap-up after
    # tool results were fed back in-context.
    assert len(llm.calls) == 2
    second_turn_messages = llm.calls[1]
    tool_messages = [m for m in second_turn_messages if m.get("role") == "tool"]
    assert len(tool_messages) == 2
    denied_tool_msg = next(
        m
        for m in tool_messages
        if m["tool_call_id"] == "call-2"
    )
    assert "denied" in denied_tool_msg["content"]


@pytest.mark.asyncio
async def test_handle_interrupt_stops_the_loop_early():
    connector = FakeConnector()
    actions = []
    runtime = FakeRuntime(connector, actions)

    class InterruptingLLM:
        def __init__(self):
            self.calls = 0

        async def chat_completion(self, messages, tools=None):
            self.calls += 1
            yield {"type": "text_delta", "text": "starting..."}
            yield {
                "type": "tool_use_complete",
                "id": "call-x",
                "name": "irrelevant",
                "input": {},
            }
            yield {"type": "message_stop"}

    llm = InterruptingLLM()
    adapter = OpenNeuroAgentAdapter(runtime, llm, session_id="sess-1")

    outputs = []
    gen = adapter.chat(SimpleNamespace(text="hello"))
    async for sentence in gen:
        outputs.append(sentence)
        # Simulate the human interrupting mid-turn, right after the first
        # piece of speech is produced.
        adapter.handle_interrupt("user spoke over the agent")

    # Once interrupted, the adapter must not act on any further tool calls
    # or advance to another turn, even if the LLM already emitted them.
    assert len(outputs) == 1
    assert runtime.evaluated == []
    assert connector.dispatch_calls == []


@pytest.mark.asyncio
async def test_set_memory_from_history_stores_ids():
    runtime = FakeRuntime(FakeConnector(), [])
    adapter = OpenNeuroAgentAdapter(runtime, FakeLLM())
    await adapter.set_memory_from_history("conf-123", "hist-456")
    assert adapter._conf_uid == "conf-123"
    assert adapter._history_uid == "hist-456"
