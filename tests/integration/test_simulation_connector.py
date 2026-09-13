"""Integration tests for SimulationConnector against the fake gateway.

Written as plain ``asyncio.run``-driven functions (not ``@pytest.mark.asyncio``)
so they don't depend on pytest-asyncio being installed/configured in the base
repo, which this isolated worker's sandbox has no way to verify.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from openneuro.applications.simulation_connector import SimulationConnector
from tests.fakes import ActionRequest
from tests.fakes.fake_gateway import FakeGateway

APP_ID = "sim-demo"


def _request(action_name: str, arguments: dict, req_id: str = "req-1") -> ActionRequest:
    return ActionRequest(
        id=req_id,
        application_id=APP_ID,
        action_name=action_name,
        arguments=arguments,
        session_id="sess-1",
        requested_at=datetime.now(timezone.utc),
    )


def test_start_registers_expected_actions_and_publishes_initial_state() -> None:
    async def body() -> None:
        gateway = FakeGateway()
        connector = SimulationConnector(APP_ID, tick_interval_seconds=100)
        await connector.start(gateway)
        try:
            assert set(gateway.registered_actions) == {"gather", "build", "attack", "wait"}
            assert len(gateway.published_states) == 1
            assert gateway.published_states[0].data["turn"] == 0
        finally:
            await connector.stop()

    asyncio.run(body())


def test_gather_mutates_state_deterministically_given_seed() -> None:
    async def body() -> None:
        gateway_a = FakeGateway()
        connector_a = SimulationConnector(APP_ID, seed=42, tick_interval_seconds=100)
        await connector_a.start(gateway_a)

        gateway_b = FakeGateway()
        connector_b = SimulationConnector(APP_ID, seed=42, tick_interval_seconds=100)
        await connector_b.start(gateway_b)

        try:
            result_a = await connector_a.dispatch_action(_request("gather", {"resource": "wood"}))
            result_b = await connector_b.dispatch_action(_request("gather", {"resource": "wood"}))

            assert result_a.success and result_b.success
            wood_a = gateway_a.published_states[-1].data["resources"]["wood"]
            wood_b = gateway_b.published_states[-1].data["resources"]["wood"]
            assert wood_a == wood_b
        finally:
            await connector_a.stop()
            await connector_b.stop()

    asyncio.run(body())


def test_build_requires_sufficient_resources() -> None:
    async def body() -> None:
        gateway = FakeGateway()
        connector = SimulationConnector(APP_ID, tick_interval_seconds=100)
        await connector.start(gateway)
        try:
            result = await connector.dispatch_action(_request("build", {"building": "house"}))
            assert result.success is False
            assert "need" in result.message
        finally:
            await connector.stop()

    asyncio.run(body())


def test_build_succeeds_once_resources_gathered() -> None:
    async def body() -> None:
        gateway = FakeGateway()
        connector = SimulationConnector(APP_ID, seed=7, tick_interval_seconds=100)
        await connector.start(gateway)
        try:
            for _ in range(15):
                await connector.dispatch_action(_request("gather", {"resource": "wood"}))
                await connector.dispatch_action(_request("gather", {"resource": "stone"}))
            result = await connector.dispatch_action(_request("build", {"building": "house"}))
            assert result.success is True
            assert gateway.published_states[-1].data["buildings"]["house"] == 1
        finally:
            await connector.stop()

    asyncio.run(body())


def test_invalid_action_arguments_are_rejected() -> None:
    async def body() -> None:
        gateway = FakeGateway()
        connector = SimulationConnector(APP_ID, tick_interval_seconds=100)
        await connector.start(gateway)
        try:
            result = await connector.dispatch_action(_request("gather", {"resource": "gold"}))
            assert result.success is False
        finally:
            await connector.stop()

    asyncio.run(body())


def test_unknown_action_name_is_rejected() -> None:
    async def body() -> None:
        gateway = FakeGateway()
        connector = SimulationConnector(APP_ID, tick_interval_seconds=100)
        await connector.start(gateway)
        try:
            result = await connector.dispatch_action(_request("dance", {}))
            assert result.success is False
        finally:
            await connector.stop()

    asyncio.run(body())


def test_tick_loop_publishes_state_and_can_emit_events() -> None:
    async def body() -> None:
        gateway = FakeGateway()
        connector = SimulationConnector(APP_ID, seed=3, tick_interval_seconds=0.01)
        await connector.start(gateway)
        try:
            await asyncio.sleep(0.2)
            assert len(gateway.published_states) > 1
            assert gateway.published_states[-1].data["turn"] > 0
        finally:
            await connector.stop()

    asyncio.run(body())


def test_stop_cancels_tick_loop_and_disconnects_gateway() -> None:
    async def body() -> None:
        gateway = FakeGateway()
        connector = SimulationConnector(APP_ID, tick_interval_seconds=0.01)
        await connector.start(gateway)

        await connector.stop()

        assert gateway.disconnected is True
        turn_at_stop = gateway.published_states[-1].data["turn"]
        await asyncio.sleep(0.1)
        assert gateway.published_states[-1].data["turn"] == turn_at_stop

    asyncio.run(body())


def test_low_hp_triggers_exactly_one_attention_request() -> None:
    async def body() -> None:
        gateway = FakeGateway()
        # Seed/duration chosen so the settlement deterministically takes
        # enough monster damage to cross the low-hp threshold and stay there.
        connector = SimulationConnector(APP_ID, seed=99, tick_interval_seconds=0.005)
        await connector.start(gateway)
        try:
            await asyncio.sleep(0.4)
            assert gateway.published_states[-1].data["hp"] <= 30
            # The connector's own in-flight guard should not re-fire once
            # already open; de-duplicating across ticks/runtime restarts is
            # the runtime/attention worker's job, not this connector's.
            assert len(gateway.attention_requests) == 1
            assert gateway.attention_requests[0].state == "settlement_in_danger"
        finally:
            await connector.stop()

    asyncio.run(body())
