"""Provider-neutral conversion from ActionDefinition to LLM tool schemas.

This module has no runtime dependency on the canonical
``openneuro.domain.models`` module (owned by worker 01_domain_models.md and
not guaranteed to exist in this sandbox). At runtime it only relies on
duck-typing: any object exposing ``.name``, ``.description``, and ``.schema``
attributes works. Type hints reference the canonical ``ActionDefinition``
type under ``TYPE_CHECKING`` only, so this module imports cleanly whether or
not that module is present.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List

if TYPE_CHECKING:  # pragma: no cover - typing only, no runtime import
    from openneuro.domain.models import ActionDefinition


def to_openai_tools(defs: "List[ActionDefinition]") -> List[Dict[str, Any]]:
    """Convert ActionDefinition objects to the OpenAI ``tools`` array shape.

    Each entry becomes:
        {"type": "function", "function": {"name", "description", "parameters"}}
    """
    tools: List[Dict[str, Any]] = []
    for d in defs:
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": d.name,
                    "description": d.description,
                    "parameters": d.schema,
                },
            }
        )
    return tools


def to_claude_tools(defs: "List[ActionDefinition]") -> List[Dict[str, Any]]:
    """Convert ActionDefinition objects to the Claude ``tools`` array shape.

    Each entry becomes:
        {"name", "description", "input_schema"}
    """
    tools: List[Dict[str, Any]] = []
    for d in defs:
        tools.append(
            {
                "name": d.name,
                "description": d.description,
                "input_schema": d.schema,
            }
        )
    return tools
