"""OpenNeuro runtime package.

Exposes ``AgentRuntime`` (composition root for the agent loop) and
``ApplicationGatewayImpl`` (the concrete ApplicationGateway handed to each
connected application). See 09_runtime.md for the full contract.
"""

from openneuro.runtime.agent_runtime import AgentRuntime
from openneuro.runtime.application_gateway_impl import ApplicationGatewayImpl

__all__ = ["AgentRuntime", "ApplicationGatewayImpl"]
