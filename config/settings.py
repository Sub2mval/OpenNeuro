"""Cross-cutting OpenNeuro configuration.

A single `OpenNeuroSettings` instance is meant to be constructed once (at
process start, or once per test) and then *injected* into every OpenNeuro
component that needs configuration (policy engine, action executor, attention
queue, trace recorder, agent adapter, ...). No OpenNeuro module should read
`os.environ` directly; they should all receive an `OpenNeuroSettings` (or the
specific values they need) through their constructor.

Environment variable names (see 11_embodiment_and_config.md):
    OPENNEURO_MODEL
    OPENNEURO_LOG_LEVEL
    OPENNEURO_ACTION_TIMEOUT            (default 20)
    OPENNEURO_MAX_TOOL_TURNS            (default 4)
    OPENNEURO_ATTENTION_QUEUE_MAX       (default 50)
    OPENNEURO_STARVATION_AGE_SECONDS    (default 30)
    OPENNEURO_POLICY_MODE
    OPENNEURO_POLICY_CONFIG_PATH
    OPENNEURO_TRACE_PATH
    OPENNEURO_SIM_TICK_SECONDS          (default 2)
    OPENNEURO_GITHUB_TOKEN              (secret)

Additional per-application secrets (e.g. a future OPENNEURO_DISCORD_TOKEN)
should follow the same pattern: a `SecretStr` field with no default, added
here as the project grows. Nothing in this file ever logs a secret's value;
`SecretStr.__repr__` already redacts it, and callers must use
`.get_secret_value()` explicitly to obtain the raw string.
"""
from __future__ import annotations

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class OpenNeuroSettings(BaseSettings):
    """Typed, environment-backed configuration for OpenNeuro.

    Field names deliberately match their env var name with the
    ``OPENNEURO_`` prefix stripped and lower-cased, e.g. ``trace_path`` <-
    ``OPENNEURO_TRACE_PATH``, so callers can write ``settings.trace_path``
    directly as required by the trace recorder contract.
    """

    model_config = SettingsConfigDict(
        env_prefix="OPENNEURO_",
        case_sensitive=False,
        extra="ignore",
    )

    # Agent / model selection.
    model: str = Field(default="", description="Model identifier used by the OpenNeuro agent adapter.")
    log_level: str = Field(default="INFO")

    # Action execution.
    action_timeout: float = Field(default=20.0, description="Seconds before an in-flight action dispatch is considered timed out.")
    max_tool_turns: int = Field(default=4, description="Max dynamic tool-call turns per agent decision loop.")

    # Attention / interruption.
    attention_queue_max: int = Field(default=50, description="Bound on queued attention requests before oldest/lowest priority is dropped.")
    starvation_age_seconds: float = Field(default=30.0, description="Age after which a queued attention request is boosted to avoid starvation.")

    # Policy engine.
    policy_mode: str = Field(default="", description="Selects which policy rule set/mode the policy engine loads.")
    policy_config_path: str | None = Field(default=None)

    # Observability.
    trace_path: str = Field(default="./openneuro_trace.jsonl", description="JSONL file the TraceRecorder appends events to.")

    # Simulation.
    sim_tick_seconds: float = Field(default=2.0, description="Tick interval for the deterministic simulation application.")

    # Secrets - never logged; obtain raw values via `.get_secret_value()` only
    # at the point of use (e.g. constructing an external connector client).
    github_token: SecretStr | None = Field(default=None)
