from __future__ import annotations

import hashlib
import json

from .types import AgentState, BeliefState, IntentState, PlanState, WorkingMemoryItem


def set_belief(
    agent: AgentState,
    *,
    key: str,
    value,
    confidence: float,
    source_event: str | None,
    now_ms: int,
) -> tuple[BeliefState | None, BeliefState]:
    before = agent.cognition.beliefs.get(key)
    revision = 1 if before is None else before.revision + 1
    belief = BeliefState(
        key=key,
        value=value,
        confidence=max(0.0, min(1.0, float(confidence))),
        source_event=source_event,
        updated_at_ms=now_ms,
        revision=revision,
    )
    agent.cognition.beliefs[key] = belief
    return before, belief


def remember_observation(
    agent: AgentState,
    *,
    event_type: str,
    source_id: int | None,
    target_id: int | None,
    confidence: float,
    now_ms: int,
) -> None:
    agent.cognition.working_memory.remember(
        WorkingMemoryItem(
            event_type=event_type,
            source_id=source_id,
            target_id=target_id,
            confidence=confidence,
            timestamp_ms=now_ms,
        )
    )


def set_intent(
    agent: AgentState,
    *,
    name: str,
    target_id: int | None,
    confidence: float,
    status: str,
    now_ms: int,
    source_request_id: str | None,
) -> tuple[IntentState | None, IntentState]:
    before = agent.cognition.intent
    intent = IntentState(
        name=name,
        target_id=target_id,
        confidence=max(0.0, min(1.0, float(confidence))),
        status=status,
        updated_at_ms=now_ms,
        source_request_id=source_request_id,
    )
    agent.cognition.intent = intent
    return before, intent


def set_plan(
    agent: AgentState,
    *,
    goal: str,
    steps: tuple[str, ...],
    target_id: int | None,
    status: str,
    now_ms: int,
    source_request_id: str | None,
) -> tuple[PlanState | None, PlanState]:
    before = agent.cognition.plan
    plan = PlanState(
        goal=goal,
        steps=steps,
        current_step=0,
        target_id=target_id,
        status=status,
        updated_at_ms=now_ms,
        source_request_id=source_request_id,
    )
    agent.cognition.plan = plan
    return before, plan


def structured_context_components(agent: AgentState, domain: str) -> dict[str, int]:
    """Estimate logical prompt footprint by structured component.

    These counts are workload-shaping estimates only. They do not reduce real
    prefill cost unless the inference runtime actually reports cache reuse.
    """

    beliefs = agent.cognition.beliefs
    if domain == "threat":
        relevant_beliefs = [b for k, b in beliefs.items() if "danger" in k or "dead" in k or "gun" in k]
    elif domain == "social":
        relevant_beliefs = list(beliefs.values())
    elif domain == "memory":
        relevant_beliefs = list(beliefs.values())
    else:
        relevant_beliefs = list(beliefs.values())[:4]

    wm = agent.cognition.working_memory.observations[-6:]
    components = {
        "beliefs": len(relevant_beliefs) * 28,
        "working_memory": len(wm) * 20,
        "intent": 34 if agent.cognition.intent is not None else 0,
        "plan": 0,
    }
    if agent.cognition.plan is not None:
        components["plan"] = 38 + len(agent.cognition.plan.steps) * 18
    return components


def state_fingerprint(agent: AgentState, domain: str) -> str:
    payload = {
        "domain": domain,
        "beliefs": [
            (b.key, b.value, round(b.confidence, 4), b.revision)
            for b in sorted(agent.cognition.beliefs.values(), key=lambda x: x.key)
        ],
        "intent": None
        if agent.cognition.intent is None
        else (
            agent.cognition.intent.name,
            agent.cognition.intent.target_id,
            agent.cognition.intent.status,
        ),
        "plan": None
        if agent.cognition.plan is None
        else (
            agent.cognition.plan.goal,
            agent.cognition.plan.steps,
            agent.cognition.plan.target_id,
            agent.cognition.plan.status,
        ),
        "wm": [
            (o.event_type, o.source_id, o.target_id, round(o.confidence, 4), o.timestamp_ms)
            for o in agent.cognition.working_memory.observations[-6:]
        ],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:16]


def invalidate_target_dependent_state(agent: AgentState, target_id: int, now_ms: int) -> tuple[list[str], list[str]]:
    """Revise cognition after a target disappears without erasing unrelated cognition."""

    preserved: list[str] = []
    invalidated: list[str] = []

    if agent.cognition.beliefs:
        preserved.append("beliefs")
    if agent.cognition.working_memory.observations:
        preserved.append("working_memory")

    intent = agent.cognition.intent
    if intent is not None:
        if intent.target_id == target_id and intent.status in {"active", "provisional"}:
            intent.status = "invalidated"
            intent.updated_at_ms = now_ms
            invalidated.append("intent")
        else:
            preserved.append("intent")

    plan = agent.cognition.plan
    if plan is not None:
        if plan.target_id == target_id and plan.status in {"active", "provisional"}:
            plan.status = "invalidated"
            plan.updated_at_ms = now_ms
            invalidated.append("plan")
        else:
            preserved.append("plan")

    return sorted(set(preserved)), sorted(set(invalidated))
