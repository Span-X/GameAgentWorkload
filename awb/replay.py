from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .tracing import read_trace


_CANONICAL_REQUEST_FIELDS = (
    "time_ms",
    "agent_id",
    "task_type",
    "priority",
    "deadline_ms",
    "input_tokens",
    "carried_state_tokens",
    "cache_reusable_tokens",
    "fresh_input_tokens",
    "expected_output_tokens",
    "target_id",
    "cognitive_domain",
    "cognitive_layer",
    "cognitive_revision",
    "continuation_of",
    "state_fingerprint",
)


def request_digest_from_trace(path: str | Path) -> str:
    records = read_trace(path)
    canonical = [
        {k: r[k] for k in _CANONICAL_REQUEST_FIELDS if k in r}
        for r in records
        if r.get("event") == "inference_request"
    ]
    raw = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()
