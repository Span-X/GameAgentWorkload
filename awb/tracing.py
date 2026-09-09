from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


class TraceRecorder:
    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []

    def emit(self, time_ms: int, event: str, **data: Any) -> None:
        self.records.append({"time_ms": int(time_ms), "event": event, **data})

    def write_jsonl(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for record in self.records:
                f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        return path

    def request_stream_digest(self) -> str:
        canonical = [
            {
                k: r[k]
                for k in (
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
                if k in r
            }
            for r in self.records
            if r.get("event") == "inference_request"
        ]
        payload = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(payload).hexdigest()


def read_trace(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
