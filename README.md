# OpenNeuro

**A protocol-first runtime for persistent embodied agents interacting with multiple applications.**

## Project Overview

Most agent systems look like `user → LLM → tool`: the model only acts when a human
prompts it, and the available tools are frozen at startup. OpenNeuro treats each
connected application as an environment that continuously publishes **state**,
**events**, and **attention requests**, and that can register or unregister
**actions** while the agent is already running.

> The LLM decides what it wants to do.
> The environment decides what is relevant.
> The runtime decides what is permitted.

The demo agent connects to and takes real action across **three external
apps** — GitHub, Discord, and Notion — via small `ApplicationConnector`
implementations, plus an internal deterministic simulation used to show the
attention/priority story without needing any of the three apps to cooperate
on cue.

### Architecture

```
Applications (GitHub / Discord / Notion / Simulation)
        │  register actions, publish state/events, request attention
        ▼
ApplicationGateway
        ▼
AgentRuntime
   ├── ActionRegistry     — the live, dynamic action set the LLM sees
   ├── AttentionManager   — priority-aware interruption / queueing
   ├── PolicyEngine       — allow / deny / require_confirmation, BEFORE dispatch
   ├── ActionExecutor     — timeout + late-result handling
   └── TraceRecorder      — structured, append-only log of every step
        ▼
OpenNeuroAgentAdapter  →  Open-LLM-VTuber embodiment (ASR/TTS/Live2D, unchanged)
```

**Known limitation, found while wiring this demo:** the boxes under
`AgentRuntime` above don't actually call each other correctly in this
snapshot — e.g. `ApplicationGatewayImpl` calls
`action_registry.register_actions(app_id, actions)` and
`event_router.publish_state(state)` / `.get_state_snapshot()`, but the real
`ActionRegistry` only exposes `.register(actions)` (no `app_id` arg) and the
real `EventRouter` only exposes `.on_state()` / `.on_event()` — different
method names entirely. Only the hand-written test fakes match what
`AgentRuntime` expects. **The demo below intentionally bypasses this broken
wiring** and drives each connector's real `dispatch_action()` directly, which
is enough to prove real, external, multi-app actions and is the same pattern
the repo's own `SimulationConnector` integration test already uses.

### Core abstractions

- **ApplicationConnector** — what an application implements: `start()`,
  `dispatch_action()`, `stop()`.
- **Attention** — applications compete for the agent's turn using a `Priority`
  (`low/medium/high/critical`).
- **Dynamic actions** — an `ActionRegistry` holds whatever actions are
  currently registered by connected applications.
- **Policy boundary** — a proposed `ActionRequest` is meant to be evaluated by
  a `PolicyEngine` before dispatch (see the limitation above for the current
  wiring gap).
- **Action execution** — dispatch runs under a timeout; a late result is
  discarded rather than double-applied.
- **Embodiment** — OpenNeuro is designed to implement Open-LLM-VTuber's
  `AgentInterface`; that wiring is not present in this snapshot.

### The three external apps

| App | Action(s) | What it needs |
|---|---|---|
| **GitHub** | `comment_on_issue`, `add_label` | `GITHUB_TOKEN` (a personal access token), `GITHUB_REPO` (`owner/name`) |
| **Discord** | `post_message` (via incoming webhook) | `DISCORD_WEBHOOK_URL` |
| **Notion** | `append_note` (appends a paragraph to a page) | `NOTION_TOKEN` (integration token), `NOTION_PAGE_ID` |

Each connector (`applications/github_connector.py`,
`applications/discord_connector.py`, `applications/notion_connector.py`) makes
a real HTTP call to that app's real API when configured, and returns a clear
"not configured" `ActionResult` — never a crash — when it isn't. `httpx` does
the actual request; none of the connectors call `.raise_for_status()`, so a
non-2xx response is reported as a failed `ActionResult` with the status code
and body, not an unhandled exception. A genuine network failure (no
connectivity, DNS, etc.) is *not* caught and will raise — see Demo hazards in
`DEMO_TODO.md`.

### Demo application: the simulation

`applications/simulation_connector.py` is a small, deterministic resource/
combat game with zero external dependency. It exposes `gather`, `build`,
`attack`, `wait`, and on its own tick loop raises a `high`-priority attention
request once the settlement's HP drops to 30 or below.

## External Apps / Tools Used

- **GitHub** (real, via `GitHubConnector`) — posts a comment / adds a label on
  a configured issue using a personal access token.
- **Discord** (real, via `DiscordConnector`) — posts a message via an
  incoming webhook URL.
- **Notion** (real, via `NotionConnector`) — appends a note to a configured
  page using an integration token.
- **Open-LLM-VTuber** — the intended embodiment target; **not wired up in
  this snapshot**, no avatar UI is shown.
- **Neuro SDK** — design inspiration only (a public protocol spec), not a
  runtime dependency.

## Setup Instructions

Needs Python 3.9+, plus `loguru` and `httpx`:

```bash
pip install loguru httpx
```

The code imports itself as `openneuro.*` everywhere, but this folder isn't
named `openneuro`. Fix with a symlink, once, from the parent directory of
this repo:

```bash
ln -s /absolute/path/to/this-repo /absolute/path/to/this-repo/../openneuro
export PYTHONPATH="/absolute/path/to/this-repo:/absolute/path/to/this-repo/.."
```

Set whichever credentials you have (any that are unset make that one
connector report "not configured" instead of crashing):

```bash
export GITHUB_TOKEN=...         GITHUB_REPO=owner/name   GITHUB_ISSUE_NUMBER=1
export DISCORD_WEBHOOK_URL=...
export NOTION_TOKEN=...         NOTION_PAGE_ID=...
```

Run the demo:

```bash
python3 demo_multi_app.py
```

**`main.py` does not run** in this snapshot — it imports names (`Runtime`,
`Settings`, `simulation.SimulationConnector`, `OpenNeuroAgent`) that don't
match what's implemented, and `AgentRuntime` itself has no `start()` method.
Don't try to run it; use `demo_multi_app.py` instead.

## How We Tested Reliability

- **Each external connector was exercised through `demo_multi_app.py`**,
  which calls its real `start()` and `dispatch_action()` directly (bypassing
  the broken runtime wiring described above). Verified: with no credentials
  set, all three connectors register their action(s) and return a clean
  "not configured" failure — no crash. With credentials set, each reaches its
  real HTTP call (confirmed by tracing the call path; see `DEMO_TODO.md` for
  how to verify this end-to-end against real accounts before recording).
- **`tests/integration/test_simulation_connector.py`** (9 tests, real
  `SimulationConnector`, zero dependencies) — dynamic action registration,
  deterministic resource yields given a fixed seed, build-cost enforcement,
  invalid/unknown actions rejected, and exactly one attention request firing
  when settlement HP goes low.
- **`tests/integration/test_runtime_with_fakes.py`** (6 tests) — proves the
  *intended* policy boundary (allow/deny/require-confirmation/timeout) against
  the real `AgentRuntime`, using scripted fakes for the collaborators, since
  the real collaborators don't interoperate yet (see limitation above).
- **What this does not cover:** no test exercises `main.py`, a real LLM call,
  the Open-LLM-VTuber path, or a full run of `AgentRuntime` wired to the real
  `ActionRegistry`/`EventRouter`/`ApplicationGatewayImpl` together — that
  wiring has the method-name mismatches described above and needs a follow-up
  fix, not a documentation workaround.

## Demo Video

https://www.youtube.com/watch?v=dPDJ1-Uy3uQ
](https://www.youtube.com/watch?v=dPDJ1-Uy3uQ)
