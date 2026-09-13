"""OpenNeuro agent adapter package.

Public surface:
    OpenNeuroAgentAdapter -- implements Open-LLM-VTuber's AgentInterface directly.
    to_openai_tools / to_claude_tools -- provider-neutral tool schema conversion.
"""

from .openneuro_agent_adapter import OpenNeuroAgentAdapter, OPENNEURO_MAX_TOOL_TURNS
from .tool_schema import to_openai_tools, to_claude_tools

__all__ = [
    "OpenNeuroAgentAdapter",
    "OPENNEURO_MAX_TOOL_TURNS",
    "to_openai_tools",
    "to_claude_tools",
]
