"""OpenNeuroAgentAdapter -- the only module allowed to call the LLM.

This adapter implements Open-LLM-VTuber's existing ``AgentInterface``
*directly* (not via ``BasicMemoryAgent``), because BasicMemoryAgent snapshots
its tool set at construction time and OpenNeuro needs dynamic, per-turn
action sets sourced from the injected ``AgentRuntime``.

Zero-context sandbox note
--------------------------
This worker's sandbox does not contain the base Open-LLM-VTuber repository,
nor the sibling OpenNeuro modules (domain models, runtime, protocol). Per
the worker contract:

* Base-repo types (``AgentInterface``, ``BatchInput``, ``SentenceOutput``,
  ``StatelessLLMInterface``) are imported for real when available, and fall
  back to minimal structural stand-ins otherwise so this module still
  imports and is unit-testable in isolation. At integration time the real
  imports will simply take over -- no code change should be required as
  long as the real classes expose the same shape used here.
* Canonical OpenNeuro domain/runtime types (``ActionDefinition``,
  ``ActionRequest``, ``ActionResult``, ``AgentObservation``, ``Priority``,
  ``AgentRuntime``) are referenced under ``TYPE_CHECKING`` only. At runtime
  the adapter interacts with them purely via duck-typing / the local
  ``AgentRuntime`` Protocol declared below, so no import of another
  worker's module is required for this module to run.

Integration hazard: the exact method names/signatures of the real
``AgentRuntime`` (worker 09_runtime.md) must match the ``AgentRuntime``
Protocol declared here (``get_observation`` / ``evaluate_and_execute``).
If the real runtime uses different names, the integrator must adapt one
side.
"""

from __future__ import annotations

from typing import (
    Any,
    AsyncIterator,
    Dict,
    List,
    Optional,
    Protocol,
    TYPE_CHECKING,
    runtime_checkable,
)

from .tool_schema import to_claude_tools, to_openai_tools

# --- Base-repo types: real import if present, structural fallback otherwise ---
try:  # pragma: no cover - exercised only when base repo is absent
    from open_llm_vtuber.agent.agents.agent_interface import AgentInterface
except ImportError:  # pragma: no cover
    class AgentInterface:  # type: ignore[no-redef]
        """Fallback stand-in used only when the base repo is not present."""

try:  # pragma: no cover
    from open_llm_vtuber.agent.input_types import BatchInput  # noqa: F401
except ImportError:  # pragma: no cover
    BatchInput = Any  # type: ignore[misc,assignment]

try:  # pragma: no cover
    from open_llm_vtuber.agent.output_types import SentenceOutput
except ImportError:  # pragma: no cover
    class SentenceOutput:  # type: ignore[no-redef]
        """Fallback stand-in used only when the base repo is not present.

        Integration hazard: replace with the real dataclass fields at
        integration time (this fallback only carries display/tts text).
        """

        def __init__(self, display_text: str, tts_text: Optional[str] = None):
            self.display_text = display_text
            self.tts_text = tts_text if tts_text is not None else display_text

if TYPE_CHECKING:  # pragma: no cover - typing only
    from open_llm_vtuber.agent.stateless_llm.stateless_llm_interface import (
        StatelessLLMInterface,
    )
    from openneuro.domain.models import (
        ActionDefinition,
        ActionRequest,
        ActionResult,
        AgentObservation,
        Priority,
    )

OPENNEURO_MAX_TOOL_TURNS = 6


@runtime_checkable
class AgentRuntime(Protocol):
    """Minimal interface this adapter needs from the injected AgentRuntime.

    Integration hazard: names/signatures must line up with worker
    09_runtime.md's actual runtime coordinator.
    """

    async def get_observation(self) -> "AgentObservation": ...

    async def evaluate_and_execute(
        self, request: "ActionRequest"
    ) -> "ActionResult": ...


def _build_request(
    application_id: Any,
    action_name: str,
    arguments: Dict[str, Any],
    session_id: Any,
    request_id: Any,
    now: Any,
) -> Any:
    """Construct an ActionRequest-shaped object without importing the
    canonical dataclass (kept duck-typed so this module has no hard runtime
    dependency on worker 01's module).
    """
    try:  # pragma: no cover - preferred path once domain models exist
        from openneuro.domain.models import ActionRequest as _ActionRequest

        return _ActionRequest(
            id=request_id,
            application_id=application_id,
            action_name=action_name,
            arguments=arguments,
            session_id=session_id,
            requested_at=now,
        )
    except ImportError:  # pragma: no cover
        # Fallback: a simple namespace with the same attribute contract.
        from types import SimpleNamespace

        return SimpleNamespace(
            id=request_id,
            application_id=application_id,
            action_name=action_name,
            arguments=arguments,
            session_id=session_id,
            requested_at=now,
        )


class OpenNeuroAgentAdapter(AgentInterface):
    """Implements AgentInterface directly; drives the dynamic tool loop.

    Not the policy engine, attention manager, or executor -- those live in
    the injected ``runtime``. This class only: builds an observation-derived
    tool list, talks to the LLM, and turns tool calls into ActionRequests
    that it hands to ``runtime.evaluate_and_execute``.
    """

    def __init__(
        self,
        runtime: AgentRuntime,
        llm: "StatelessLLMInterface",
        *,
        session_id: Any = None,
        provider: str = "openai",
        max_tool_turns: int = OPENNEURO_MAX_TOOL_TURNS,
    ) -> None:
        if provider not in ("openai", "claude"):
            raise ValueError(f"unsupported provider: {provider!r}")
        self._runtime = runtime
        self._llm = llm
        self._session_id = session_id
        self._provider = provider
        self._max_tool_turns = max_tool_turns
        self._interrupted = False
        self._conf_uid: Optional[str] = None
        self._history_uid: Optional[str] = None
        self._request_seq = 0

    async def chat(self, input_data: "BatchInput") -> AsyncIterator[SentenceOutput]:
        self._interrupted = False
        observation = await self._runtime.get_observation()
        tools = self._tools_for(observation.available_actions)
        messages = self._seed_messages(input_data, observation)

        for _ in range(self._max_tool_turns):
            if self._interrupted:
                return

            text_chunks: List[str] = []
            tool_calls: List[Dict[str, Any]] = []

            async for event in self._llm.chat_completion(messages, tools=tools):
                if self._interrupted:
                    return
                etype = event.get("type")
                if etype == "text_delta":
                    text = event.get("text", "")
                    text_chunks.append(text)
                    if text:
                        yield SentenceOutput(display_text=text)
                elif etype == "tool_use_complete":
                    tool_calls.append(event)
                elif etype == "error":
                    yield SentenceOutput(
                        display_text=f"[agent error] {event.get('message', '')}"
                    )
                    return
                elif etype == "message_stop":
                    break

            messages.append(
                {
                    "role": "assistant",
                    "content": "".join(text_chunks),
                    "tool_calls": tool_calls,
                }
            )

            if not tool_calls:
                return

            for call in tool_calls:
                if self._interrupted:
                    return
                result = await self._execute_tool_call(call, observation)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.get("id"),
                        "content": result.message,
                    }
                )
        # max_tool_turns exhausted; loop simply ends without a hard error.

    async def _execute_tool_call(
        self, call: Dict[str, Any], observation: "AgentObservation"
    ) -> "ActionResult":
        """Build an ActionRequest and delegate to the runtime for
        policy evaluation + (possible) dispatch. The runtime is solely
        responsible for deciding whether the connector ever sees this
        action; a denial never reaches a connector because this adapter
        never talks to connectors directly.
        """
        from datetime import datetime, timezone

        self._request_seq += 1
        request = _build_request(
            application_id=call.get("application_id"),
            action_name=call["name"],
            arguments=call.get("input", {}),
            session_id=self._session_id,
            request_id=f"req-{self._request_seq}",
            now=datetime.now(timezone.utc),
        )
        return await self._runtime.evaluate_and_execute(request)

    def _tools_for(self, defs: List["ActionDefinition"]) -> List[Dict[str, Any]]:
        if self._provider == "claude":
            return to_claude_tools(defs)
        return to_openai_tools(defs)

    def _seed_messages(
        self, input_data: "BatchInput", observation: "AgentObservation"
    ) -> List[Dict[str, Any]]:
        user_text = getattr(input_data, "text", None) or str(input_data)
        messages: List[Dict[str, Any]] = [{"role": "user", "content": user_text}]
        if observation.pending_query:
            messages.append(
                {"role": "system", "content": observation.pending_query}
            )
        return messages

    def handle_interrupt(self, heard_response: str) -> None:
        self._interrupted = True

    async def set_memory_from_history(self, conf_uid: Any, history_uid: Any) -> None:
        self._conf_uid = conf_uid
        self._history_uid = history_uid
