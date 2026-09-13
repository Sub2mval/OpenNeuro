"""Append-only JSONL trace recorder.

Depends on the canonical `TraceEvent` / `TraceEventKind` domain contracts
defined by the domain-models worker (see 01_domain_models.md,
`openneuro.domain.models`). This module imports them directly rather than
duplicating them; in an isolated sandbox where that module has not been
merged in yet, importing `openneuro.observability.trace_recorder` will raise
`ImportError` until it is. Tests in this repo install a small in-test stub
for `openneuro.domain.models` instead of a second production copy.
"""
from __future__ import annotations

import json
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from openneuro.domain.models import TraceEvent  # canonical contract, see 01_domain_models.md

# Matches obvious secret-shaped keys anywhere in a (possibly nested) payload:
# token, secret, key, password/passwd, credential(s), api_key, auth, etc.
_SECRET_KEY_PATTERN = re.compile(
    r"(token|secret|password|passwd|credential|api[_-]?key|(?<!\w)key(?!\w)|auth)",
    re.IGNORECASE,
)
_REDACTED = "***REDACTED***"


def _looks_secret(key: Any) -> bool:
    return isinstance(key, str) and bool(_SECRET_KEY_PATTERN.search(key))


def redact(value: Any) -> Any:
    """Recursively redact obvious secret-shaped fields in a JSON-able value.

    Only dict *keys* are inspected for secret-ish names; the corresponding
    value (whatever its type/shape) is replaced wholesale. Lists/tuples are
    walked so secrets nested inside them are still caught. This is a
    best-effort safety net, not a guarantee - callers should still avoid
    putting raw secrets into trace payloads in the first place.
    """
    if isinstance(value, dict):
        return {
            k: (_REDACTED if _looks_secret(k) else redact(v))
            for k, v in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [redact(v) for v in value]
    return value


class TraceRecorder:
    """Append-only JSONL trace recorder.

    A single instance is safe to share across coroutines/threads within one
    process: writes are serialized behind a lock. Storage is a flat JSONL
    file at ``settings.trace_path`` (one JSON object per line), so it can be
    tailed, grepped, or reprocessed without holding the whole trace in memory.
    """

    def __init__(self, trace_path: str) -> None:
        self._path = Path(trace_path)
        if self._path.parent and str(self._path.parent) not in ("", "."):
            self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def record(self, kind: Any, payload: dict[str, Any], trace_id: str) -> TraceEvent:
        """Append one trace event and return the (redacted) `TraceEvent`.

        `kind` is expected to be a `TraceEventKind` member but a plain string
        is also accepted so callers that only have the raw event name still
        work; it is stored as `kind.value` when available, else `str(kind)`.
        """
        event = TraceEvent(
            trace_id=trace_id,
            ts=datetime.now(timezone.utc),
            kind=kind,
            payload=redact(payload),
        )
        row = {
            "trace_id": event.trace_id,
            "ts": event.ts.isoformat(),
            "kind": getattr(event.kind, "value", event.kind),
            "payload": event.payload,
        }
        line = json.dumps(row, default=str)
        with self._lock:
            with self._path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        return event

    def events_for(self, trace_id: str) -> list[TraceEvent]:
        """Return all recorded events for `trace_id`, in append order."""
        if not self._path.exists():
            return []

        with self._lock:
            lines = self._path.read_text(encoding="utf-8").splitlines()

        events: list[TraceEvent] = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("trace_id") != trace_id:
                continue
            events.append(
                TraceEvent(
                    trace_id=row["trace_id"],
                    ts=datetime.fromisoformat(row["ts"]),
                    kind=row["kind"],
                    payload=row["payload"],
                )
            )
        return events
