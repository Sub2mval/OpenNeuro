"""OpenNeuro multi-app demo driver.

Calls each connector's real dispatch_action() directly through a FakeGateway
(the same mechanism already proven by tests/integration/test_simulation_connector.py),
bypassing the runtime/registry/event-router wiring, which has method-name
mismatches between its pieces in this snapshot and is not safe to route a
live demo through under time pressure. Every connector below still makes a
REAL network call to a REAL external app when its credentials are set.

Needs only the Python standard library PLUS ``loguru`` and ``httpx`` (used by
the GitHub/Discord/Notion connectors' real HTTP path). If those two aren't
installed, this script exits with a clear message rather than a traceback.

Setup: see DEMO_TODO.md for the openneuro import-path symlink. Then set
whichever of these env vars you have credentials for; any unset connector
still runs and reports "not configured" instead of crashing:

    GITHUB_TOKEN, GITHUB_REPO           (owner/name)
    DISCORD_WEBHOOK_URL
    NOTION_TOKEN, NOTION_PAGE_ID

Run:
    python3 demo_multi_app.py
"""
from __future__ import annotations

import asyncio
import os
import sys

try:
    import loguru  # noqa: F401
    import httpx  # noqa: F401
except ImportError as exc:  # pragma: no cover
    print(f"Missing dependency ({exc}). Install with: pip install loguru httpx")
    sys.exit(1)

from tests.fakes.fake_gateway import FakeGateway
from tests.fakes import ActionRequest

from applications.github_connector import GitHubConnector, GitHubConnectorSettings
from applications.discord_connector import DiscordConnector, DiscordConnectorSettings
from applications.notion_connector import NotionConnector, NotionConnectorSettings
from applications.simulation_connector import SimulationConnector

from datetime import datetime, timezone


def _request(app_id: str, action_name: str, arguments: dict) -> ActionRequest:
    return ActionRequest(
        id=f"{app_id}-{action_name}",
        application_id=app_id,
        action_name=action_name,
        arguments=arguments,
        session_id="demo-session",
        requested_at=datetime.now(timezone.utc),
    )


async def _run_connector(label: str, app_id: str, connector, request: ActionRequest) -> None:
    gateway = FakeGateway()
    await connector.start(gateway)
    print(f"\n=== {label} ===")
    print(f"registered actions: {sorted(gateway.registered_actions.keys())}")
    result = await connector.dispatch_action(request)
    status = "SUCCESS" if result.success else "FAILED"
    print(f"dispatch_action({request.action_name!r}) -> {status}: {result.message}")
    await connector.stop()


async def main() -> None:
    # --- Simulation: no credentials, always works, ties the story together.
    sim = SimulationConnector("simulation", tick_interval_seconds=100)
    await _run_connector(
        "Simulation (internal, no credentials needed)",
        "simulation",
        sim,
        _request("simulation", "gather", {"resource": "wood"}),
    )

    # --- GitHub: real REST API call if GITHUB_TOKEN/GITHUB_REPO are set.
    gh_settings = GitHubConnectorSettings(
        token=os.environ.get("GITHUB_TOKEN"),
        repo=os.environ.get("GITHUB_REPO"),
    )
    gh = GitHubConnector("github", gh_settings)
    issue_number = int(os.environ.get("GITHUB_ISSUE_NUMBER", "1"))
    await _run_connector(
        "GitHub",
        "github",
        gh,
        _request(
            "github",
            "comment_on_issue",
            {"issue_number": issue_number, "body": "Hello from OpenNeuro's demo agent!"},
        ),
    )

    # --- Discord: real webhook POST if DISCORD_WEBHOOK_URL is set.
    discord_settings = DiscordConnectorSettings(webhook_url=os.environ.get("DISCORD_WEBHOOK_URL"))
    discord = DiscordConnector("discord", discord_settings)
    await _run_connector(
        "Discord",
        "discord",
        discord,
        _request("discord", "post_message", {"content": "Hello from OpenNeuro's demo agent!"}),
    )

    # --- Notion: real API call if NOTION_TOKEN/NOTION_PAGE_ID are set.
    notion_settings = NotionConnectorSettings(
        token=os.environ.get("NOTION_TOKEN"),
        page_id=os.environ.get("NOTION_PAGE_ID"),
    )
    notion = NotionConnector("notion", notion_settings)
    await _run_connector(
        "Notion",
        "notion",
        notion,
        _request("notion", "append_note", {"text": "Hello from OpenNeuro's demo agent!"}),
    )

    print("\n--- summary ---")
    print("3 external apps exercised: GitHub, Discord, Notion (plus the internal simulation).")
    print("Set GITHUB_TOKEN/GITHUB_REPO, DISCORD_WEBHOOK_URL, NOTION_TOKEN/NOTION_PAGE_ID")
    print("for real actions; any unset connector reports 'not configured' instead of crashing.")


if __name__ == "__main__":
    asyncio.run(main())
