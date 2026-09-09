from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from statistics import mean


def percentile(values: list[int | float], p: float) -> float:
    if not values:
        return 0.0
    xs = sorted(float(x) for x in values)
    if len(xs) == 1:
        return xs[0]
    rank = (len(xs) - 1) * p
    lo = math.floor(rank)
    hi = math.ceil(rank)
    if lo == hi:
        return xs[lo]
    weight = rank - lo
    return xs[lo] * (1 - weight) + xs[hi] * weight


@dataclass(slots=True)
class BenchmarkReport:
    scenario: str
    agents: int
    peak_active: int
    peak_cognitive: int
    requests: int
    completed: int
    superseded_before_start: int
    interrupted_inflight: int
    deadline_misses: int
    queue_ms_p50: float
    queue_ms_p95: float
    queue_ms_p99: float
    decision_ms_p50: float
    decision_ms_p95: float
    decision_ms_p99: float
    service_ms_mean: float
    carried_state_tokens: int
    simulated_cache_reusable_tokens: int
    interrupted_processed_prefill_tokens: int
    interrupted_processed_decode_tokens: int
    belief_updates: int
    working_memory_updates: int
    intent_transitions: int
    plan_transitions: int
    reflex_actions: int
    cognitive_commits: int
    cognitive_revisions: int
    preserved_state_components: int
    invalidated_state_components: int
    task_breakdown: dict[str, dict]
    layer_breakdown: dict[str, dict]
    trace_digest: str

    @property
    def adaptation_rate(self) -> float:
        adapted = self.superseded_before_start + self.interrupted_inflight
        return adapted / self.requests if self.requests else 0.0

    @property
    def deadline_miss_rate(self) -> float:
        return self.deadline_misses / self.completed if self.completed else 0.0

    def to_dict(self) -> dict:
        data = asdict(self)
        data["adaptation_rate"] = self.adaptation_rate
        data["deadline_miss_rate"] = self.deadline_miss_rate
        return data

    def write_json(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path

    def render_text(self) -> str:
        return f"""GameAgentWorkload 0.4-alpha Benchmark\n\nScenario: {self.scenario}\nAgents: {self.agents}\nPeak active: {self.peak_active}\nPeak cognitive: {self.peak_cognitive}\nInference requests: {self.requests}\nCompleted: {self.completed}\nSuperseded before start: {self.superseded_before_start}\nInterrupted in-flight: {self.interrupted_inflight}\nAdaptation rate: {self.adaptation_rate:.1%}\nDeadline misses: {self.deadline_misses} ({self.deadline_miss_rate:.1%})\n\nQueue latency ms\n  p50: {self.queue_ms_p50:.1f}\n  p95: {self.queue_ms_p95:.1f}\n  p99: {self.queue_ms_p99:.1f}\n\nEnd-to-end decision latency ms\n  p50: {self.decision_ms_p50:.1f}\n  p95: {self.decision_ms_p95:.1f}\n  p99: {self.decision_ms_p99:.1f}\n\nMean backend service: {self.service_ms_mean:.1f} ms\n\nStructured cognition\n  carried-state tokens (logical): {self.carried_state_tokens}\n  simulated cache-reusable tokens: {self.simulated_cache_reusable_tokens}\n  belief updates: {self.belief_updates}\n  working-memory updates: {self.working_memory_updates}\n  intent transitions: {self.intent_transitions}\n  plan transitions: {self.plan_transitions}\n  reflex actions: {self.reflex_actions}\n  cognitive commits: {self.cognitive_commits}\n  cognitive revisions after interruption: {self.cognitive_revisions}\n  preserved state components: {self.preserved_state_components}\n  invalidated state components: {self.invalidated_state_components}\n\nInterrupted runtime work (not automatically classified as waste)\n  processed prefill tokens: {self.interrupted_processed_prefill_tokens}\n  processed decode tokens: {self.interrupted_processed_decode_tokens}\n\nTrace digest: {self.trace_digest}\n"""


def build_report(scenario: str, agents: int, trace_records: list[dict], digest: str) -> BenchmarkReport:
    requests = [r for r in trace_records if r.get("event") == "inference_request"]
    starts = {r["request_id"]: r for r in trace_records if r.get("event") == "inference_started"}
    finishes = [r for r in trace_records if r.get("event") == "inference_finished"]
    superseded = [r for r in trace_records if r.get("event") == "inference_superseded"]
    interrupted = [r for r in trace_records if r.get("event") == "inference_interrupted"]
    activations = [r for r in trace_records if r.get("event") == "activation_snapshot"]
    beliefs = [r for r in trace_records if r.get("event") == "belief_updated"]
    wm_updates = [r for r in trace_records if r.get("event") == "working_memory_updated"]
    intents = [r for r in trace_records if r.get("event") == "intent_transition"]
    plans = [r for r in trace_records if r.get("event") == "plan_transition"]
    reflex_actions = [r for r in trace_records if r.get("event") == "reflex_action"]
    commits = [r for r in trace_records if r.get("event") == "cognitive_state_committed"]
    revisions = [r for r in trace_records if r.get("event") == "cognitive_state_revised"]

    queue_ms: list[int] = []
    decision_ms: list[int] = []
    service_ms: list[int] = []
    deadline_misses = 0

    req_by_id = {r["request_id"]: r for r in requests}
    task_breakdown: dict[str, dict] = {}
    layer_breakdown: dict[str, dict] = {}
    for req in requests:
        task_breakdown.setdefault(
            req["task_type"],
            {"requests": 0, "completed": 0, "deadline_misses": 0, "carried_state_tokens": 0},
        )
        task_breakdown[req["task_type"]]["requests"] += 1
        task_breakdown[req["task_type"]]["carried_state_tokens"] += req.get("carried_state_tokens", 0)
        layer = req.get("cognitive_layer", "cognitive")
        layer_breakdown.setdefault(layer, {"requests": 0, "completed": 0, "deadline_misses": 0})
        layer_breakdown[layer]["requests"] += 1

    for f in finishes:
        rid = f["request_id"]
        req = req_by_id[rid]
        start = starts[rid]
        queue_ms.append(start["time_ms"] - req["time_ms"])
        decision_ms.append(f["time_ms"] - req["time_ms"])
        service_ms.append(f["time_ms"] - start["time_ms"])
        task = task_breakdown[req["task_type"]]
        task["completed"] += 1
        layer = layer_breakdown[req.get("cognitive_layer", "cognitive")]
        layer["completed"] += 1
        if f["time_ms"] > req["time_ms"] + req["deadline_ms"]:
            deadline_misses += 1
            task["deadline_misses"] += 1
            layer["deadline_misses"] += 1

    for task in task_breakdown.values():
        completed = task["completed"]
        task["deadline_miss_rate"] = task["deadline_misses"] / completed if completed else 0.0
    for layer in layer_breakdown.values():
        completed = layer["completed"]
        layer["deadline_miss_rate"] = layer["deadline_misses"] / completed if completed else 0.0

    peak_active = max(
        (r.get("active", 0) + r.get("cognitive", 0) + r.get("interactive", 0) for r in activations),
        default=0,
    )
    peak_cognitive = max(
        (r.get("cognitive", 0) + r.get("interactive", 0) for r in activations),
        default=0,
    )

    return BenchmarkReport(
        scenario=scenario,
        agents=agents,
        peak_active=peak_active,
        peak_cognitive=peak_cognitive,
        requests=len(requests),
        completed=len(finishes),
        superseded_before_start=len(superseded),
        interrupted_inflight=len(interrupted),
        deadline_misses=deadline_misses,
        queue_ms_p50=percentile(queue_ms, 0.50),
        queue_ms_p95=percentile(queue_ms, 0.95),
        queue_ms_p99=percentile(queue_ms, 0.99),
        decision_ms_p50=percentile(decision_ms, 0.50),
        decision_ms_p95=percentile(decision_ms, 0.95),
        decision_ms_p99=percentile(decision_ms, 0.99),
        service_ms_mean=mean(service_ms) if service_ms else 0.0,
        carried_state_tokens=sum(r.get("carried_state_tokens", 0) for r in requests),
        simulated_cache_reusable_tokens=sum(r.get("cache_reusable_tokens", 0) for r in requests),
        interrupted_processed_prefill_tokens=sum(r.get("processed_prefill_tokens", 0) for r in interrupted),
        interrupted_processed_decode_tokens=sum(r.get("processed_decode_tokens", 0) for r in interrupted),
        belief_updates=len(beliefs),
        working_memory_updates=len(wm_updates),
        intent_transitions=len(intents),
        plan_transitions=len(plans),
        reflex_actions=len(reflex_actions),
        cognitive_commits=len(commits),
        cognitive_revisions=len(revisions),
        preserved_state_components=sum(len(r.get("preserved_components", [])) for r in revisions),
        invalidated_state_components=sum(len(r.get("invalidated_components", [])) for r in revisions),
        task_breakdown=task_breakdown,
        layer_breakdown=layer_breakdown,
        trace_digest=digest,
    )
