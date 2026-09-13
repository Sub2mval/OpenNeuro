"""Bridge between OpenNeuro attention requests and the existing Open-LLM-VTuber
(OLLV) embodiment stack (ASR, VAD, TTS, Live2D, sentence streaming, human
websocket transport, chat history).

This module intentionally does NOT reimplement any of those systems. It only
adapts an OpenNeuro `AttentionRequest` into the same kind of proactive-speech
turn OLLV already knows how to run (the path exercised today by the existing
"ai-speak-signal" trigger), so that:

  * ordinary human voice/text input keeps following its original, unmodified
    path through the conversation handler, and
  * an OpenNeuro-triggered turn looks like one more agent turn to everything
    downstream of the agent call (history manager, TTS, Live2D, websocket
    broadcast), tagged only via turn metadata - never via an application-id
    string check in core code.

INTEGRATION NOTE FOR THE FINAL INTEGRATOR: this worker was built in an
isolated sandbox with no access to the actual Open-LLM-VTuber source tree
(per its zero-context contract). The shapes of `ServiceContext`, `BatchInput`,
`TextData`/`TextSource`, and the exact call site of the existing
proactive-speech / "ai-speak-signal" path are reproduced here as best-effort
from the contract in 11_embodiment_and_config.md. Please diff this file's
`_build_batch_input` against the real `open_llm_vtuber/agent/input_types.py`
and adjust field/enum names if they differ; the public surface
(`EmbodimentAdapter.trigger_agent_turn`) should not need to change.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Awaitable, Callable

if TYPE_CHECKING:  # pragma: no cover - typing only, no runtime dependency
    from open_llm_vtuber.service_context import ServiceContext
    from openneuro.domain.models import AttentionRequest

logger = logging.getLogger(__name__)

# Signature of the existing upstream hook this adapter calls back into. See
# the additive branch added to open_llm_vtuber/conversations/conversation_handler.py
# (documented in this worker's final report), which mirrors the existing
# ai-speak-signal path and simply forwards to whatever already runs a
# proactive agent turn given a BatchInput.
RunProactiveTurn = Callable[["ServiceContext", Any], Awaitable[None]]


class EmbodimentAdapterError(RuntimeError):
    """Raised when a turn cannot be synthesized or dispatched."""


def _build_metadata(request: "AttentionRequest") -> dict[str, Any]:
    """Build turn metadata that avoids double-counting this turn as a normal
    human utterance, and lets downstream code recognize it *without* any
    application-id string branching: only the `from_openneuro` flag matters
    to shared/core code; `application_id` is carried for logging/telemetry
    only.
    """
    priority = getattr(request, "priority", None)
    event_id = getattr(request, "event_id", None)
    return {
        "source": "openneuro",
        "from_openneuro": True,
        "application_id": str(getattr(request, "application_id", "")),
        "event_id": str(event_id) if event_id is not None else None,
        "priority": getattr(priority, "value", priority),
        "ephemeral": bool(getattr(request, "ephemeral", False)),
        "candidate_actions": list(getattr(request, "candidate_actions", None) or []),
        # Tells the chat-history writer this turn's input should not be
        # persisted as if a human typed/said it (still logged via TraceRecorder).
        "skip_human_history_entry": True,
    }


class EmbodimentAdapter:
    """Bridges an OpenNeuro `AttentionRequest` into one existing OLLV agent
    turn, without owning any ASR/VAD/TTS/Live2D/websocket logic itself.

    `run_proactive_turn` is injected (dependency injection, not a global) and
    is expected to be the existing OLLV function that already runs a
    proactive/non-human-triggered agent turn given a `ServiceContext` and a
    `BatchInput` - i.e. the same function the "ai-speak-signal" path calls.
    Injecting it keeps this module import-safe and unit-testable even where
    the real OLLV package is not present.
    """

    def __init__(self, run_proactive_turn: RunProactiveTurn) -> None:
        self._run_proactive_turn = run_proactive_turn

    async def trigger_agent_turn(self, context: "ServiceContext", request: "AttentionRequest") -> None:
        """Synthesize a `BatchInput` from `request` and run one agent turn.

        Uses `request.query` as the turn's textual content. Raises
        `EmbodimentAdapterError` if the query is empty (nothing to say/ask)
        or if the existing OLLV input types cannot be imported (i.e. this
        adapter is being run outside the full application).
        """
        query = getattr(request, "query", None)
        if not query:
            raise EmbodimentAdapterError(
                "AttentionRequest.query is empty; EmbodimentAdapter has nothing to say/ask."
            )

        metadata = _build_metadata(request)
        batch_input = self._build_batch_input(query=query, metadata=metadata)

        logger.info(
            "openneuro.embodiment.trigger_agent_turn application_id=%s priority=%s ephemeral=%s",
            metadata["application_id"],
            metadata["priority"],
            metadata["ephemeral"],
        )

        await self._run_proactive_turn(context, batch_input)

    @staticmethod
    def _build_batch_input(*, query: str, metadata: dict[str, Any]) -> Any:
        """Construct the existing OLLV `BatchInput` used for proactive turns.

        Imported lazily (only when a turn actually fires), not at module
        import time, so `openneuro.embodiment.embodiment_adapter` stays
        importable in isolated sandboxes/tests that don't have the base repo.
        """
        try:
            from open_llm_vtuber.agent.input_types import BatchInput, TextData, TextSource
        except ImportError as exc:  # pragma: no cover - only hit outside the full app
            raise EmbodimentAdapterError(
                "open_llm_vtuber.agent.input_types is unavailable; EmbodimentAdapter "
                "must run inside the full Open-LLM-VTuber application."
            ) from exc

        # Prefer an explicit "not spoken by the human" source if the real
        # enum exposes one; fall back to whatever default input source exists
        # so this still works if the exact member name differs upstream.
        source = getattr(TextSource, "NON_HUMAN", None) or getattr(TextSource, "SYSTEM", None) or getattr(TextSource, "INPUT")

        return BatchInput(
            texts=[TextData(source=source, content=query, from_name="openneuro")],
            images=[],
            metadata=metadata,
        )
