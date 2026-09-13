"""Unit tests for openneuro.agent_adapter.tool_schema.

Uses a local stub ActionDefinition (duck-typed: .name/.description/.schema)
since the canonical openneuro.domain.models module is owned by a different
worker and is not assumed to exist in this sandbox.
"""

from dataclasses import dataclass, field
from typing import Any, Dict

import pytest

from openneuro.agent_adapter.tool_schema import to_claude_tools, to_openai_tools


@dataclass
class StubActionDefinition:
    application_id: str
    name: str
    description: str
    schema: Dict[str, Any] = field(default_factory=dict)
    risk: str = "low"


@pytest.fixture
def defs():
    return [
        StubActionDefinition(
            application_id="app-1",
            name="open_issue",
            description="Open a GitHub issue",
            schema={
                "type": "object",
                "properties": {"title": {"type": "string"}},
                "required": ["title"],
            },
        ),
        StubActionDefinition(
            application_id="app-1",
            name="close_issue",
            description="Close a GitHub issue",
            schema={
                "type": "object",
                "properties": {"issue_id": {"type": "string"}},
                "required": ["issue_id"],
            },
        ),
    ]


def test_to_openai_tools_shape(defs):
    tools = to_openai_tools(defs)
    assert len(tools) == 2
    for tool, d in zip(tools, defs):
        assert tool["type"] == "function"
        assert tool["function"]["name"] == d.name
        assert tool["function"]["description"] == d.description
        assert tool["function"]["parameters"] == d.schema


def test_to_claude_tools_shape(defs):
    tools = to_claude_tools(defs)
    assert len(tools) == 2
    for tool, d in zip(tools, defs):
        assert tool["name"] == d.name
        assert tool["description"] == d.description
        assert tool["input_schema"] == d.schema
        assert "type" not in tool  # claude shape has no OpenAI 'type' wrapper


def test_empty_list_returns_empty():
    assert to_openai_tools([]) == []
    assert to_claude_tools([]) == []


def test_conversions_are_independent_of_each_other(defs):
    """Converting to one schema must not mutate the source defs, so the
    same ActionDefinition list can be converted to both provider shapes."""
    to_openai_tools(defs)
    tools_claude = to_claude_tools(defs)
    assert tools_claude[0]["input_schema"] is defs[0].schema
