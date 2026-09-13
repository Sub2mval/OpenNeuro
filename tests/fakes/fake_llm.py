"""Deterministic, network-free test double for OLLV's StatelessLLMInterface.

INTEGRATION HAZARD: this worker's sandbox does not contain the base
Open-LLM-VTuber repository, so the real
``agent/stateless_llm/stateless_llm_interface.py`` could not be inspected as
instructed. The fallback shape below is this worker's best-effort
reconstruction from the documented OLLV architecture:

    class StatelessLLMInterface:
        async def chat_completion(
            self,
            messages: list[dict],
            system: str | None = None,
            tools: list[dict] | None = None,
        ) -> AsyncIterator[str | ToolCallEvent]:
            ...

i.e. an async method returning an async iterator that yields plain string
text-deltas, and yields a discrete ``ToolCallEvent`` once a tool call is
complete (rather than streaming partial tool-call JSON).

The final integrator MUST diff this file against the real
``stateless_llm_interface.py`` and adjust the method name/signature and the
tool-call event shape if they differ -- the ``try`` import below is written
so that once the real interface is importable, it is used automatically and
this reconstruction becomes dead code.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, List, Optional, Sequence, Union

try:  # pragma: no cover - only present once the base repo is available
    from agent.stateless_llm.stateless_llm_interface import StatelessLLMInterface
except ImportError:  # pragma: no cover - base repo not present in this sandbox

    class StatelessLLMInterface:  # type: ignore[no-redef]
        """Reconstructed stand-in -- see module docstring for caveats."""

        async def chat_completion(
            self,
            messages: List[Dict[str, Any]],
            system: Optional[str] = None,
            tools: Optional[List[Dict[str, Any]]] = None,
        ) -> AsyncIterator[Any]:
            raise NotImplementedError
            yield  # pragma: no cover - keeps this an async generator for typing


@dataclass
class ToolCallEvent:
    """A single completed tool call surfaced mid-stream."""

    name: str
    arguments: Dict[str, Any] = field(default_factory=dict)


ScriptedChunk = Union[str, ToolCallEvent]


class FakeLLM(StatelessLLMInterface):
    """Scriptable, deterministic StatelessLLMInterface double.

    Never calls a real provider. Tests script one or more "turns" up front
    (or incrementally via ``script``); each call to ``chat_completion``
    consumes the next scripted turn and yields its chunks in order -- plain
    strings as text deltas, ``ToolCallEvent`` for a completed tool call.
    """

    def __init__(self, turns: Optional[Sequence[Sequence[ScriptedChunk]]] = None) -> None:
        self._turns: List[List[ScriptedChunk]] = [list(t) for t in (turns or [])]
        self.calls: List[Dict[str, Any]] = []

    def script(self, *chunks: ScriptedChunk) -> None:
        """Append one more scripted turn (consumed on the next call)."""
        self._turns.append(list(chunks))

    async def chat_completion(
        self,
        messages: List[Dict[str, Any]],
        system: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> AsyncIterator[ScriptedChunk]:
        self.calls.append({"messages": messages, "system": system, "tools": tools})
        turn = self._turns.pop(0) if self._turns else []
        for chunk in turn:
            yield chunk
