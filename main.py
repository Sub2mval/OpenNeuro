"""OpenNeuro bootstrap / composition root.

Isolated-worker note
---------------------
This sandbox does not contain the other OpenNeuro workers' production
modules (Settings, the runtime coordinator, SimulationConnector, the agent
adapter, OLLV's config/integration hook, etc.) -- only this prompt's
contract. This script is therefore written defensively: every collaborator
is imported lazily inside ``build_runtime``/``main`` with a clear error if
it is missing, so that:

  * running this file inside an isolated worker sandbox fails fast with a
    readable message instead of a confusing traceback, and
  * once the final integrator merges all workers, this file runs as-is
    with no changes needed.

This file is composition/bootstrap code only -- it does not implement a new
framework, and it never modifies OLLV upstream files (that is worker 11's
responsibility).
"""

from __future__ import annotations

import asyncio
import sys
from dataclasses import dataclass, field
from typing import Any, Optional

from loguru import logger


def _import_or_none(module_path: str, attr: str) -> Optional[Any]:
    try:
        module = __import__(module_path, fromlist=[attr])
        return getattr(module, attr)
    except (ImportError, AttributeError):
        return None


@dataclass
class BootstrapSettings:
    """Fallback settings shape, used only if openneuro.config.Settings is absent.

    The real Settings type (worker 11) is expected to supply at least these
    fields plus whatever OLLV/config-file loading it does; this fallback
    exists purely so main.py can be read/exercised in isolation.
    """

    github_token: Optional[str] = None
    github_repo: Optional[str] = None
    github_poll_interval_seconds: float = 30.0
    enable_browser: bool = False


def load_settings() -> Any:
    """Load Settings via the canonical config module if present, else fall back."""
    settings_cls = _import_or_none("openneuro.config", "Settings")
    if settings_cls is not None:
        try:
            return settings_cls.load()  # type: ignore[attr-defined]
        except AttributeError:
            return settings_cls()  # type: ignore[call-arg]
    logger.warning(
        "openneuro.config.Settings not found in this sandbox; using a "
        "minimal BootstrapSettings() with no external connectors configured."
    )
    return BootstrapSettings()


def build_simulation_connector(application_id: Any) -> Any:
    """SimulationConnector is the mandatory, no-credential fallback application."""
    simulation_cls = _import_or_none("openneuro.applications.simulation", "SimulationConnector")
    if simulation_cls is None:
        raise RuntimeError(
            "openneuro.applications.simulation.SimulationConnector is required "
            "(it is the no-credential fallback application) but is not present "
            "in this sandbox. This module belongs to worker 08 and must be "
            "merged in before main.py can actually run."
        )
    return simulation_cls(application_id=application_id)


def build_optional_connectors(settings: Any) -> list:
    """Best-effort construction of GitHub/Browser connectors when configured."""
    from openneuro.applications.github_connector import GitHubConnector, GitHubConnectorSettings
    from openneuro.applications.browser_connector import BrowserConnector

    connectors: list = []

    github_token = getattr(settings, "github_token", None)
    github_repo = getattr(settings, "github_repo", None)
    if github_token and github_repo:
        gh_settings = GitHubConnectorSettings(
            token=github_token,
            repo=github_repo,
            poll_interval_seconds=getattr(settings, "github_poll_interval_seconds", 30.0),
        )
        connectors.append(GitHubConnector(application_id="github", settings=gh_settings))
        logger.info("GitHubConnector configured for repo {}", github_repo)
    else:
        logger.info("GitHubConnector not configured (missing token/repo); skipping.")

    if getattr(settings, "enable_browser", False):
        browser = BrowserConnector(application_id="browser")
        if browser.enabled:
            connectors.append(browser)
            logger.info("BrowserConnector enabled.")
        else:
            logger.info("BrowserConnector requested but no browser automation available; skipping.")
    else:
        logger.info("BrowserConnector not requested; skipping.")

    return connectors


def build_runtime(settings: Any) -> Any:
    """Construct the runtime coordinator + gateway and register applications.

    Requires worker 09's runtime module. Raises a clear error if absent so
    this fails fast rather than partially wiring things up.
    """
    runtime_cls = _import_or_none("openneuro.runtime", "Runtime")
    if runtime_cls is None:
        raise RuntimeError(
            "openneuro.runtime.Runtime is not present in this sandbox. main.py "
            "is composition glue for the merged system; it is expected to be "
            "exercised by the final integrator, not inside an isolated worker "
            "sandbox."
        )

    runtime = runtime_cls(settings=settings)
    runtime.register_application(build_simulation_connector(application_id="simulation"))
    for connector in build_optional_connectors(settings):
        runtime.register_application(connector)
    return runtime


def wire_agent_into_ollv(runtime: Any, settings: Any) -> None:
    """Hook the OpenNeuro agent adapter into OLLV's existing agent selection.

    Worker 11 owns the actual OLLV integration point (the two "tiny OLLV
    hooks" mentioned in the architecture). This function assumes that point
    exposes a registration callable importable as
    ``open_llm_vtuber.agent.agent_factory.register_agent`` accepting an
    agent-type key and a factory callable, and that ``openneuro_agent`` is
    the key end users select in their OLLV character config to opt into
    OpenNeuro. If that hook is not present, we log and continue -- a
    missing OLLV hook should not prevent the OpenNeuro runtime itself
    (e.g. for the smoke test or the GitHub/Browser connectors) from working.
    """
    register_agent = _import_or_none("open_llm_vtuber.agent.agent_factory", "register_agent")
    adapter_cls = _import_or_none("openneuro.agent_adapter", "OpenNeuroAgent")

    if register_agent is None or adapter_cls is None:
        logger.warning(
            "OLLV agent registration hook or openneuro.agent_adapter.OpenNeuroAgent "
            "not found in this sandbox; skipping OLLV wiring. This is expected "
            "when running main.py outside of the fully-integrated build."
        )
        return

    register_agent("openneuro_agent", lambda **kwargs: adapter_cls(runtime=runtime, **kwargs))
    logger.info("Registered 'openneuro_agent' with OLLV's agent factory.")


async def async_main() -> None:
    settings = load_settings()
    runtime = build_runtime(settings)
    wire_agent_into_ollv(runtime, settings)

    start = getattr(runtime, "start", None)
    if start is None:
        raise RuntimeError("openneuro.runtime.Runtime has no start() method.")

    logger.info("Starting OpenNeuro runtime...")
    await start()


def main() -> None:
    try:
        asyncio.run(async_main())
    except RuntimeError as exc:
        logger.error(str(exc))
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Shutting down.")


if __name__ == "__main__":
    main()
