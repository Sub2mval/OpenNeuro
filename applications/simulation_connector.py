"""Deterministic in-process demo application: a tiny resource/combat game.

Implements ApplicationConnector directly (see the protocol contract at the
top of this worker's prompt). Requires no external service or credential and
is fully deterministic given a fixed ``seed`` -- this is the "always works"
guaranteed demo app.

Deliberately does NOT import runtime, policy, or agent_adapter: this module
must stay usable standalone by anything that only has the protocol contract.
"""
from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from . import (
    ActionDefinition,
    ActionRequest,
    ActionResult,
    ApplicationId,
    ApplicationState,
    AttentionRequest,
    Priority,
)

GATHER_RESOURCES = ("wood", "stone", "food")
BUILDINGS = ("house", "farm")

_ACTION_SCHEMAS: Dict[str, Dict[str, Any]] = {
    "gather": {
        "type": "object",
        "properties": {"resource": {"type": "string", "enum": list(GATHER_RESOURCES)}},
        "required": ["resource"],
    },
    "build": {
        "type": "object",
        "properties": {"building": {"type": "string", "enum": list(BUILDINGS)}},
        "required": ["building"],
    },
    "attack": {"type": "object", "properties": {}, "required": []},
    "wait": {"type": "object", "properties": {}, "required": []},
}

_BUILD_COST: Dict[str, Dict[str, int]] = {
    "house": {"wood": 5, "stone": 2},
    "farm": {"wood": 3, "stone": 1},
}


class SchemaValidationError(ValueError):
    """Raised when ActionRequest.arguments don't match the action's own schema."""


def _validate_arguments(action_name: str, arguments: Dict[str, Any]) -> None:
    schema = _ACTION_SCHEMAS.get(action_name)
    if schema is None:
        raise SchemaValidationError(f"unknown action: {action_name}")

    for required_field in schema.get("required", []):
        if required_field not in arguments:
            raise SchemaValidationError(f"{action_name}: missing required field '{required_field}'")

    properties = schema.get("properties", {})
    for key, value in arguments.items():
        prop = properties.get(key)
        if prop is None:
            raise SchemaValidationError(f"{action_name}: unexpected field '{key}'")
        if prop.get("type") == "string" and not isinstance(value, str):
            raise SchemaValidationError(f"{action_name}.{key}: expected string")
        allowed = prop.get("enum")
        if allowed is not None and value not in allowed:
            raise SchemaValidationError(f"{action_name}.{key}: '{value}' not in {allowed}")


def _now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class _GameState:
    turn: int = 0
    wood: int = 0
    stone: int = 0
    food: int = 0
    hp: int = 100
    monster_hp: int = 30
    houses: int = 0
    farms: int = 0
    log: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "turn": self.turn,
            "resources": {"wood": self.wood, "stone": self.stone, "food": self.food},
            "hp": self.hp,
            "monster_hp": self.monster_hp,
            "buildings": {"house": self.houses, "farm": self.farms},
            "log": list(self.log[-5:]),
        }


class SimulationConnector:
    """Deterministic local demo app: gather resources, build, fight a monster.

    A fixed ``seed`` makes every tick (resource yields, monster attacks, and
    the scripted event/attention requests they trigger) reproducible, which
    keeps demos and tests stable.
    """

    def __init__(
        self,
        application_id: ApplicationId,
        *,
        seed: int = 1337,
        tick_interval_seconds: float = 1.0,
    ) -> None:
        self.application_id = application_id
        self._seed = seed
        self._rng = random.Random(seed)
        self._tick_interval = tick_interval_seconds
        self._state = _GameState()
        self._version = 0
        self._gateway: Optional[Any] = None
        self._tick_task: Optional["asyncio.Task[None]"] = None
        self._attention_open = False

    # -- ApplicationConnector protocol --------------------------------------

    async def start(self, gateway: Any) -> None:
        # start() clears/reinitializes previously registered state for this
        # application (core semantics: a fresh start means a fresh game).
        self._gateway = gateway
        self._state = _GameState()
        self._version = 0
        self._rng = random.Random(self._seed)
        self._attention_open = False

        gateway.register_actions(self._action_definitions())
        self._publish_state()
        self._tick_task = asyncio.create_task(self._tick_loop())

    async def dispatch_action(self, request: ActionRequest) -> ActionResult:
        try:
            _validate_arguments(request.action_name, request.arguments)
            message = self._apply_action(request.action_name, request.arguments)
            success = True
        except SchemaValidationError as exc:
            message = str(exc)
            success = False

        self._version += 1
        self._publish_state()
        return ActionResult(
            request_id=request.id,
            success=success,
            message=message,
            completed_at=_now(),
        )

    async def stop(self) -> None:
        if self._tick_task is not None:
            self._tick_task.cancel()
            try:
                await self._tick_task
            except asyncio.CancelledError:
                pass
            self._tick_task = None

        if self._gateway is not None:
            self._gateway.unregister_actions(list(_ACTION_SCHEMAS.keys()))
            self._gateway.disconnect()

    # -- action catalogue -----------------------------------------------------

    def _action_definitions(self) -> List[ActionDefinition]:
        descriptions = {
            "gather": "Gather one batch of a resource (wood, stone, or food).",
            "build": "Spend resources to build a house or a farm.",
            "attack": "Attack the monster threatening the settlement.",
            "wait": "Do nothing this turn.",
        }
        risks = {"gather": "low", "build": "low", "attack": "medium", "wait": "low"}
        now = _now()
        return [
            ActionDefinition(
                application_id=self.application_id,
                name=name,
                description=descriptions[name],
                schema=_ACTION_SCHEMAS[name],
                risk=risks[name],
                registered_at=now,
            )
            for name in _ACTION_SCHEMAS
        ]

    # -- internal game logic ----------------------------------------------

    def _apply_action(self, action_name: str, arguments: Dict[str, Any]) -> str:
        state = self._state

        if action_name == "gather":
            resource = arguments["resource"]
            amount = self._rng.randint(1, 3)
            setattr(state, resource, getattr(state, resource) + amount)
            msg = f"gathered {amount} {resource}"

        elif action_name == "build":
            building = arguments["building"]
            cost = _BUILD_COST[building]
            if state.wood < cost["wood"] or state.stone < cost["stone"]:
                raise SchemaValidationError(
                    f"build {building}: need {cost['wood']} wood and {cost['stone']} stone"
                )
            state.wood -= cost["wood"]
            state.stone -= cost["stone"]
            if building == "house":
                state.houses += 1
            else:
                state.farms += 1
            msg = f"built a {building}"

        elif action_name == "attack":
            damage = self._rng.randint(4, 9)
            state.monster_hp = max(0, state.monster_hp - damage)
            msg = f"attacked for {damage} (monster hp={state.monster_hp})"

        else:  # wait
            msg = "waited"

        state.log.append(msg)
        return msg

    def _publish_state(self) -> None:
        if self._gateway is None:
            return
        self._gateway.publish_state(
            ApplicationState(
                application_id=self.application_id,
                data=self._state.as_dict(),
                version=self._version,
                updated_at=_now(),
            )
        )

    async def _tick_loop(self) -> None:
        while True:
            await asyncio.sleep(self._tick_interval)
            self._advance_turn()

    def _advance_turn(self) -> None:
        state = self._state
        state.turn += 1
        self._version += 1

        # Deterministic scripted monster pressure: attacks every 3rd turn
        # while it's still alive.
        if state.monster_hp > 0 and state.turn % 3 == 0:
            damage = self._rng.randint(3, 7)
            state.hp = max(0, state.hp - damage)
            state.log.append(f"monster hit the settlement for {damage} (hp={state.hp})")
            if self._gateway is not None:
                self._gateway.publish_event(
                    f"The monster attacks! Settlement hp is now {state.hp}.",
                    silent=False,
                    priority=Priority.MEDIUM,
                )

        if state.hp <= 30 and not self._attention_open and self._gateway is not None:
            self._attention_open = True
            self._gateway.request_attention(
                AttentionRequest(
                    application_id=self.application_id,
                    event_id=None,
                    priority=Priority.HIGH,
                    state="settlement_in_danger",
                    query="Settlement HP is low. Attack the monster or keep building defenses?",
                    ephemeral=True,
                    candidate_actions=["attack", "build", "wait"],
                    created_at=_now(),
                    expires_at=None,
                )
            )
        if state.hp > 60:
            self._attention_open = False

        self._publish_state()
