from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Callable

from .types import InferenceRequest
from .tracing import TraceRecorder


@dataclass(order=True, slots=True)
class _Queued:
    sort_key: tuple[int, int, int]
    sequence: int
    request: InferenceRequest = field(compare=False)


@dataclass(slots=True)
class _Running:
    request: InferenceRequest
    worker_id: int
    started_at_ms: int
    finish_at_ms: int


StartHook = Callable[[InferenceRequest, int], None]
CompletionHook = Callable[[InferenceRequest, int], None]
InterruptionHook = Callable[[InferenceRequest, int, int, int, str], tuple[list[str], list[str]]]


class InferenceScheduler:
    """Discrete-event deadline/priority scheduler.

    v0.3 separates runtime work from cognitive state semantics. If a request is
    interrupted, the scheduler reports how much backend work had happened; the
    world decides which structured cognitive components survive.
    """

    def __init__(
        self,
        backend,
        trace: TraceRecorder,
        *,
        on_started: StartHook | None = None,
        on_completed: CompletionHook | None = None,
        on_interrupted: InterruptionHook | None = None,
    ) -> None:
        self.backend = backend
        self.trace = trace
        self.on_started = on_started
        self.on_completed = on_completed
        self.on_interrupted = on_interrupted
        self._queued: list[_Queued] = []
        self._running: dict[str, _Running] = {}
        self._sequence = 0
        self._workers_free_at = [0 for _ in range(backend.max_concurrency)]

    def enqueue(self, request: InferenceRequest) -> None:
        key = (-int(request.priority), request.absolute_deadline_ms, self._sequence)
        heapq.heappush(self._queued, _Queued(key, self._sequence, request))
        self._sequence += 1
        self.trace.emit(
            request.created_at_ms,
            "inference_request",
            request_id=request.request_id,
            agent_id=request.agent_id,
            task_type=request.task_type,
            priority=int(request.priority),
            deadline_ms=request.deadline_ms,
            input_tokens=request.input_tokens,
            carried_state_tokens=request.carried_state_tokens,
            cache_reusable_tokens=request.cache_reusable_tokens,
            fresh_input_tokens=request.fresh_input_tokens,
            expected_output_tokens=request.expected_output_tokens,
            target_id=request.target_id,
            world_version=request.world_version,
            cognitive_domain=request.cognitive_domain,
            cognitive_layer=request.cognitive_layer,
            cognitive_revision=request.cognitive_revision,
            continuation_of=request.continuation_of,
            state_fingerprint=request.state_fingerprint,
        )

    def _free_workers(self, now_ms: int) -> list[int]:
        return [i for i, free_at in enumerate(self._workers_free_at) if free_at <= now_ms]

    def start_ready(self, now_ms: int) -> None:
        while self._queued:
            free = self._free_workers(now_ms)
            if not free:
                break
            worker_id = free[0]
            item = heapq.heappop(self._queued)
            req = item.request
            service_ms = self.backend.estimate_service_ms(req)
            finish_at = now_ms + service_ms
            self._workers_free_at[worker_id] = finish_at
            self._running[req.request_id] = _Running(req, worker_id, now_ms, finish_at)
            self.trace.emit(
                now_ms,
                "inference_started",
                request_id=req.request_id,
                agent_id=req.agent_id,
                worker_id=worker_id,
                service_ms=service_ms,
                carried_state_tokens=req.carried_state_tokens,
                cache_reusable_tokens=req.cache_reusable_tokens,
                fresh_input_tokens=req.fresh_input_tokens,
            )
            if self.on_started is not None:
                self.on_started(req, now_ms)

    def next_completion_time(self) -> int | None:
        if not self._running:
            return None
        return min(r.finish_at_ms for r in self._running.values())

    def complete_at(self, now_ms: int) -> None:
        done = [rid for rid, r in self._running.items() if r.finish_at_ms <= now_ms]
        for rid in sorted(done):
            running = self._running.pop(rid)
            self.trace.emit(
                running.finish_at_ms,
                "inference_finished",
                request_id=rid,
                agent_id=running.request.agent_id,
                output_tokens=running.request.expected_output_tokens,
            )
            if self.on_completed is not None:
                self.on_completed(running.request, running.finish_at_ms)

    def cancel_invalid_target(self, target_id: int, now_ms: int) -> int:
        """Supersede queued and interrupt in-flight target-dependent cognition."""
        adapted = 0
        kept: list[_Queued] = []
        while self._queued:
            item = heapq.heappop(self._queued)
            req = item.request
            if req.cancel_if_target_dead and req.target_id == target_id:
                adapted += 1
                self.trace.emit(
                    now_ms,
                    "inference_superseded",
                    request_id=req.request_id,
                    agent_id=req.agent_id,
                    reason="target_dead_before_start",
                    stage="queued",
                )
            else:
                kept.append(item)
        for item in kept:
            heapq.heappush(self._queued, item)

        for rid, running in list(self._running.items()):
            req = running.request
            if req.cancel_if_target_dead and req.target_id == target_id:
                elapsed = max(0, now_ms - running.started_at_ms)
                progress = self.backend.estimate_progress(req, elapsed)
                preserved: list[str] = []
                invalidated: list[str] = []
                if self.on_interrupted is not None:
                    preserved, invalidated = self.on_interrupted(
                        req,
                        now_ms,
                        progress.processed_prefill_tokens,
                        progress.processed_decode_tokens,
                        "target_dead_inflight",
                    )

                self._running.pop(rid)
                self._workers_free_at[running.worker_id] = now_ms
                adapted += 1
                self.trace.emit(
                    now_ms,
                    "inference_interrupted",
                    request_id=rid,
                    agent_id=req.agent_id,
                    reason="target_dead_inflight",
                    stage=progress.stage,
                    processed_prefill_tokens=progress.processed_prefill_tokens,
                    processed_decode_tokens=progress.processed_decode_tokens,
                    preserved_components=preserved,
                    invalidated_components=invalidated,
                )
        return adapted

    def has_pending(self) -> bool:
        return bool(self._queued or self._running)

    def queued_count(self) -> int:
        return len(self._queued)
