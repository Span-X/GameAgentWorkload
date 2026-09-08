from __future__ import annotations

import csv
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .backends.llamacpp import LlamaCppBackend
from .metrics import percentile
from .tracing import read_trace


@dataclass(slots=True)
class RealReplayResult:
    request_id: str
    agent_id: int
    task_type: str
    cognitive_domain: str
    cognitive_layer: str
    scheduled_time_ms: int
    deadline_ms: int
    input_tokens_target: int
    output_tokens_target: int
    client_queue_ms: float
    wall_ms: float
    decision_ms: float
    ttft_ms: float
    server_prompt_ms: float
    server_predicted_ms: float
    server_prompt_tokens: int
    server_cache_tokens: int
    server_predicted_tokens: int
    server_prompt_tok_s: float
    server_decode_tok_s: float
    deadline_miss: bool
    zero_queue_deadline_miss: bool
    queue_only_deadline_miss: bool
    ttft_deadline_miss: bool
    deadline_slack_ms: float
    zero_queue_slack_ms: float
    error: str | None = None


@dataclass(slots=True)
class RealReplayReport:
    backend: str
    trace: str
    requests_selected: int
    completed: int
    failed: int
    concurrency: int
    timing_mode: str
    cache_prompt: bool
    deadline_misses: int
    zero_queue_deadline_misses: int
    queue_only_deadline_misses: int
    ttft_deadline_misses: int
    client_queue_ms_p50: float
    client_queue_ms_p95: float
    wall_ms_p50: float
    wall_ms_p95: float
    decision_ms_p50: float
    decision_ms_p95: float
    ttft_ms_p50: float
    ttft_ms_p95: float
    prompt_tok_s_mean: float
    decode_tok_s_mean: float
    cache_tokens_total: int
    prompt_tokens_total: int
    cache_ratio: float
    layer_summary: dict[str, dict] = field(default_factory=dict)

    @property
    def deadline_miss_rate(self) -> float:
        return self.deadline_misses / self.completed if self.completed else 0.0

    @property
    def zero_queue_deadline_floor(self) -> float:
        return self.zero_queue_deadline_misses / self.completed if self.completed else 0.0

    @property
    def queue_only_miss_rate(self) -> float:
        return self.queue_only_deadline_misses / self.completed if self.completed else 0.0

    @property
    def ttft_deadline_miss_rate(self) -> float:
        return self.ttft_deadline_misses / self.completed if self.completed else 0.0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["deadline_miss_rate"] = self.deadline_miss_rate
        d["zero_queue_deadline_floor"] = self.zero_queue_deadline_floor
        d["queue_only_miss_rate"] = self.queue_only_miss_rate
        d["ttft_deadline_miss_rate"] = self.ttft_deadline_miss_rate
        return d

    def render_text(self) -> str:
        layer_lines = []
        if self.layer_summary:
            layer_lines.append("\nBy cognitive layer")
            layer_lines.append("  Layer        N   Miss%   ZeroQ%   TTFT%   Decision p95")
            for layer, row in sorted(self.layer_summary.items()):
                layer_lines.append(
                    f"  {layer:<11} {row['completed']:>3} "
                    f"{row['deadline_miss_rate']*100:>6.1f} "
                    f"{row['zero_queue_deadline_floor']*100:>7.1f} "
                    f"{row['ttft_deadline_miss_rate']*100:>6.1f} "
                    f"{row['decision_ms_p95']:>12.1f}"
                )
        return f"""GameAgentWorkload 0.4-alpha Real Replay

Backend: {self.backend}
Trace: {self.trace}
Selected requests: {self.requests_selected}
Completed: {self.completed}
Failed: {self.failed}
Concurrency: {self.concurrency}
Timing mode: {self.timing_mode}
llama.cpp cache_prompt: {self.cache_prompt}

Deadline analysis
  total misses: {self.deadline_misses} ({self.deadline_miss_rate:.1%})
  zero-queue service floor: {self.zero_queue_deadline_misses} ({self.zero_queue_deadline_floor:.1%})
  queue-only misses: {self.queue_only_deadline_misses} ({self.queue_only_miss_rate:.1%})
  TTFT later than deadline: {self.ttft_deadline_misses} ({self.ttft_deadline_miss_rate:.1%})

Client queue latency ms
  p50: {self.client_queue_ms_p50:.1f}
  p95: {self.client_queue_ms_p95:.1f}

Request wall latency ms
  p50: {self.wall_ms_p50:.1f}
  p95: {self.wall_ms_p95:.1f}

End-to-end decision latency ms
  p50: {self.decision_ms_p50:.1f}
  p95: {self.decision_ms_p95:.1f}

TTFT ms
  p50: {self.ttft_ms_p50:.1f}
  p95: {self.ttft_ms_p95:.1f}

Server throughput
  prompt tok/s mean: {self.prompt_tok_s_mean:.1f}
  decode tok/s mean: {self.decode_tok_s_mean:.1f}
  cache tokens total: {self.cache_tokens_total}
  prompt tokens total: {self.prompt_tokens_total}
  observed cache ratio: {self.cache_ratio:.1%}
""" + "\n".join(layer_lines) + ("\n" if layer_lines else "")


def _request_records(trace_path: Path) -> list[dict]:
    return [r for r in read_trace(trace_path) if r.get("event") == "inference_request"]


def _filter_records(
    records: list[dict],
    task_types: set[str] | None,
    cognitive_layers: set[str] | None,
    max_requests: int | None,
) -> list[dict]:
    if task_types:
        records = [r for r in records if r.get("task_type") in task_types]
    if cognitive_layers:
        records = [r for r in records if r.get("cognitive_layer", "cognitive") in cognitive_layers]
    if max_requests is not None:
        records = records[: max(0, int(max_requests))]
    return records



def _deadline_flags(
    deadline_ms: int,
    queue_ms: float,
    wall_ms: float,
    ttft_ms: float,
) -> tuple[bool, bool, bool, bool]:
    """Return total_miss, zero_queue_miss, queue_only_miss, ttft_miss."""
    if deadline_ms <= 0:
        return False, False, False, False
    total_miss = queue_ms + wall_ms > deadline_ms
    zero_queue_miss = wall_ms > deadline_ms
    queue_only_miss = total_miss and not zero_queue_miss
    ttft_miss = ttft_ms > deadline_ms
    return total_miss, zero_queue_miss, queue_only_miss, ttft_miss


def _layer_summary(results: list[RealReplayResult]) -> dict[str, dict]:
    grouped: dict[str, list[RealReplayResult]] = {}
    for r in results:
        grouped.setdefault(r.cognitive_layer, []).append(r)
    out: dict[str, dict] = {}
    for layer, rows in grouped.items():
        n = len(rows)
        out[layer] = {
            "completed": n,
            "deadline_misses": sum(r.deadline_miss for r in rows),
            "deadline_miss_rate": sum(r.deadline_miss for r in rows) / n if n else 0.0,
            "zero_queue_deadline_misses": sum(r.zero_queue_deadline_miss for r in rows),
            "zero_queue_deadline_floor": sum(r.zero_queue_deadline_miss for r in rows) / n if n else 0.0,
            "queue_only_deadline_misses": sum(r.queue_only_deadline_miss for r in rows),
            "ttft_deadline_misses": sum(r.ttft_deadline_miss for r in rows),
            "ttft_deadline_miss_rate": sum(r.ttft_deadline_miss for r in rows) / n if n else 0.0,
            "decision_ms_p50": percentile([r.decision_ms for r in rows], 0.50),
            "decision_ms_p95": percentile([r.decision_ms for r in rows], 0.95),
            "wall_ms_p95": percentile([r.wall_ms for r in rows], 0.95),
            "ttft_ms_p95": percentile([r.ttft_ms for r in rows], 0.95),
        }
    return out


def run_real_replay(
    *,
    trace_path: Path,
    backend: LlamaCppBackend,
    out_dir: Path,
    concurrency: int = 1,
    max_requests: int | None = None,
    task_types: set[str] | None = None,
    cognitive_layers: set[str] | None = None,
    timing_mode: str = "trace",
    arrival_scale: float = 1.0,
    cache_prompt: bool = False,
    cache_experiment: bool = False,
    server_slots: int | None = None,
) -> tuple[Path, Path, RealReplayReport]:
    records = _filter_records(_request_records(trace_path), task_types, cognitive_layers, max_requests)
    if not records:
        raise ValueError("No inference_request records selected from trace")
    concurrency = max(1, int(concurrency))
    if timing_mode not in {"trace", "immediate"}:
        raise ValueError("timing_mode must be 'trace' or 'immediate'")
    if arrival_scale <= 0:
        raise ValueError("arrival_scale must be > 0")

    health = backend.health()
    if health.get("status") != "ok":
        raise RuntimeError(f"llama.cpp server not ready: {health}")

    prompt_by_request: dict[str, list[int]] = {}
    prompt_lock = threading.Lock()
    first_trace_ms = int(records[0].get("time_ms", 0))
    replay_start = time.perf_counter()

    def worker(record: dict, submitted_at: float) -> RealReplayResult:
        started_at = time.perf_counter()
        client_queue_ms = (started_at - submitted_at) * 1000.0
        previous = None
        continuation = record.get("continuation_of")
        if continuation:
            with prompt_lock:
                previous = prompt_by_request.get(str(continuation))
        try:
            prompt = backend.make_exact_prompt_tokens(
                record,
                previous_prompt_tokens=previous,
                cache_experiment=cache_experiment,
            )
            with prompt_lock:
                prompt_by_request[str(record["request_id"])] = prompt
            id_slot = -1
            if server_slots and server_slots > 0:
                id_slot = int(record.get("agent_id", 0)) % int(server_slots)
            result = backend.stream_completion(
                prompt_tokens=prompt,
                n_predict=int(record.get("expected_output_tokens", 1)),
                cache_prompt=cache_prompt,
                id_slot=id_slot,
            )
            wall_ms = float(result["wall_ms"])
            decision_ms = client_queue_ms + wall_ms
            ttft_ms = float(result["ttft_ms"])
            deadline_ms = int(record.get("deadline_ms", 0))
            total_miss, zero_queue_miss, queue_only_miss, ttft_miss = _deadline_flags(
                deadline_ms, client_queue_ms, wall_ms, ttft_ms
            )
            return RealReplayResult(
                request_id=str(record["request_id"]),
                agent_id=int(record.get("agent_id", -1)),
                task_type=str(record.get("task_type", "unknown")),
                cognitive_domain=str(record.get("cognitive_domain", "general")),
                cognitive_layer=str(record.get("cognitive_layer", "cognitive")),
                scheduled_time_ms=int(record.get("time_ms", 0)),
                deadline_ms=deadline_ms,
                input_tokens_target=int(record.get("input_tokens", 0)),
                output_tokens_target=int(record.get("expected_output_tokens", 0)),
                client_queue_ms=client_queue_ms,
                wall_ms=wall_ms,
                decision_ms=decision_ms,
                ttft_ms=ttft_ms,
                server_prompt_ms=float(result["prompt_ms"]),
                server_predicted_ms=float(result["predicted_ms"]),
                server_prompt_tokens=int(result["prompt_n"]),
                server_cache_tokens=int(result["cache_n"]),
                server_predicted_tokens=int(result["predicted_n"]),
                server_prompt_tok_s=float(result["prompt_per_second"]),
                server_decode_tok_s=float(result["predicted_per_second"]),
                deadline_miss=total_miss,
                zero_queue_deadline_miss=zero_queue_miss,
                queue_only_deadline_miss=queue_only_miss,
                ttft_deadline_miss=ttft_miss,
                deadline_slack_ms=float(deadline_ms) - decision_ms if deadline_ms > 0 else 0.0,
                zero_queue_slack_ms=float(deadline_ms) - wall_ms if deadline_ms > 0 else 0.0,
            )
        except Exception as exc:
            return RealReplayResult(
                request_id=str(record.get("request_id")),
                agent_id=int(record.get("agent_id", -1)),
                task_type=str(record.get("task_type", "unknown")),
                cognitive_domain=str(record.get("cognitive_domain", "general")),
                cognitive_layer=str(record.get("cognitive_layer", "cognitive")),
                scheduled_time_ms=int(record.get("time_ms", 0)),
                deadline_ms=int(record.get("deadline_ms", 0)),
                input_tokens_target=int(record.get("input_tokens", 0)),
                output_tokens_target=int(record.get("expected_output_tokens", 0)),
                client_queue_ms=client_queue_ms,
                wall_ms=0.0,
                decision_ms=client_queue_ms,
                ttft_ms=0.0,
                server_prompt_ms=0.0,
                server_predicted_ms=0.0,
                server_prompt_tokens=0,
                server_cache_tokens=0,
                server_predicted_tokens=0,
                server_prompt_tok_s=0.0,
                server_decode_tok_s=0.0,
                deadline_miss=False,
                zero_queue_deadline_miss=False,
                queue_only_deadline_miss=False,
                ttft_deadline_miss=False,
                deadline_slack_ms=0.0,
                zero_queue_slack_ms=0.0,
                error=f"{type(exc).__name__}: {exc}",
            )

    futures = []
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        for record in records:
            if timing_mode == "trace":
                relative_ms = (int(record.get("time_ms", 0)) - first_trace_ms) / arrival_scale
                target_wall = replay_start + relative_ms / 1000.0
                sleep_s = target_wall - time.perf_counter()
                if sleep_s > 0:
                    time.sleep(sleep_s)
            submitted_at = time.perf_counter()
            futures.append(pool.submit(worker, record, submitted_at))
        results = [f.result() for f in as_completed(futures)]

    results.sort(key=lambda r: (r.scheduled_time_ms, r.request_id))
    completed = [r for r in results if r.error is None]
    failed = [r for r in results if r.error is not None]
    prompt_speeds = [r.server_prompt_tok_s for r in completed if r.server_prompt_tok_s > 0]
    decode_speeds = [r.server_decode_tok_s for r in completed if r.server_decode_tok_s > 0]
    cache_total = sum(r.server_cache_tokens for r in completed)
    prompt_total = sum(r.server_prompt_tokens + r.server_cache_tokens for r in completed)

    report = RealReplayReport(
        backend=f"llama.cpp@{backend.base_url}",
        trace=str(trace_path),
        requests_selected=len(records),
        completed=len(completed),
        failed=len(failed),
        concurrency=concurrency,
        timing_mode=timing_mode,
        cache_prompt=cache_prompt,
        deadline_misses=sum(1 for r in completed if r.deadline_miss),
        zero_queue_deadline_misses=sum(1 for r in completed if r.zero_queue_deadline_miss),
        queue_only_deadline_misses=sum(1 for r in completed if r.queue_only_deadline_miss),
        ttft_deadline_misses=sum(1 for r in completed if r.ttft_deadline_miss),
        client_queue_ms_p50=percentile([r.client_queue_ms for r in completed], 0.50),
        client_queue_ms_p95=percentile([r.client_queue_ms for r in completed], 0.95),
        wall_ms_p50=percentile([r.wall_ms for r in completed], 0.50),
        wall_ms_p95=percentile([r.wall_ms for r in completed], 0.95),
        decision_ms_p50=percentile([r.decision_ms for r in completed], 0.50),
        decision_ms_p95=percentile([r.decision_ms for r in completed], 0.95),
        ttft_ms_p50=percentile([r.ttft_ms for r in completed], 0.50),
        ttft_ms_p95=percentile([r.ttft_ms for r in completed], 0.95),
        prompt_tok_s_mean=sum(prompt_speeds) / len(prompt_speeds) if prompt_speeds else 0.0,
        decode_tok_s_mean=sum(decode_speeds) / len(decode_speeds) if decode_speeds else 0.0,
        cache_tokens_total=cache_total,
        prompt_tokens_total=prompt_total,
        cache_ratio=cache_total / prompt_total if prompt_total else 0.0,
        layer_summary=_layer_summary(completed),
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "real_replay.requests.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = list(asdict(results[0]).keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(asdict(r) for r in results)

    report_path = out_dir / "real_replay.report.json"
    report_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    return csv_path, report_path, report
