from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

from awb.bedroom import ActionOption, BedroomCase
from awb.control_state import AgentControlState, bedroom_control_state
from awb.semantic_decision import SemanticModelResponse


PROMPT_STYLE_VERSION = "agent_direct_control_v0.3_exception_state"
ESCALATE_TOKEN = "ESCALATE"
MOTOR_COMMIT_RESERVE_MS = 15


class DirectControlBackend(Protocol):
    def choose_action(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
    ) -> SemanticModelResponse:
        ...


@dataclass(slots=True, frozen=True)
class ParsedDirectDecision:
    kind: str
    action: ActionOption | None
    status: str


@dataclass(slots=True, frozen=True)
class DirectControlCaseResult:
    case: str
    run_index: int
    prompt_style: str
    prompt_sha256: str
    oracle_gate_used: bool
    engineered_gate_features_exposed: bool
    decision_kind: str
    action: str | None
    action_roles: tuple[str, ...]
    escalated: bool
    parse_status: str
    decision_valid: bool
    local_resolution_pass: bool
    world_deadline_ms: int | None
    decision_deadline_ms: int | None
    deadline_miss: bool
    survived: bool
    realtime_pass: bool
    wall_ms: float
    ttft_ms: float
    prompt_tokens: int | None
    completion_tokens: int | None
    finish_reason: str | None
    model_text: str
    reasoning_text: str
    violated_preconditions: tuple[str, ...]
    unknown_preconditions: tuple[str, ...]


SYSTEM_PROMPT = """You are an always-on low-latency controller for an autonomous agent.
Use only the structured control state supplied by the runtime.
Return exactly one executable action id from executable_actions, or return ESCALATE if the current observation cannot be responsibly resolved by one of those actions without additional cognition/context.
If there is no decision-relevant change and a continuation action is executable, choose that continuation action rather than escalating.
Do not invent actions or hidden world facts. Do not explain. Output only the action id or ESCALATE."""


def build_direct_control_prompt(state: AgentControlState) -> str:
    """Serialize the direct-controller input as auditable structured JSON."""

    payload = state.prompt_payload()
    payload["controller_output_contract"] = {
        "allowed": [
            *(action.name for action in state.available_actions),
            ESCALATE_TOKEN,
        ],
        "meaning_of_escalate": (
            "route this decision to a more expensive cognitive system; "
            "ESCALATE is not a physical world action"
        ),
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def parse_direct_decision(
    text: str,
    actions: tuple[ActionOption, ...],
) -> ParsedDirectDecision:
    """Parse only the executable final channel; never mine hidden reasoning."""

    by_name = {action.name: action for action in actions}
    cleaned = text.strip().strip("`").strip()
    if cleaned == ESCALATE_TOKEN:
        return ParsedDirectDecision("escalate", None, "exact")
    if cleaned in by_name:
        return ParsedDirectDecision("action", by_name[cleaned], "exact")

    # Tolerate a tiny JSON wrapper for instruct models, while keeping the set
    # of accepted values closed. Reasoning text is intentionally ignored.
    try:
        payload = json.loads(cleaned)
    except (json.JSONDecodeError, TypeError):
        payload = None
    if isinstance(payload, dict):
        candidate = payload.get("decision") or payload.get("action") or payload.get("action_id")
        if isinstance(candidate, str):
            candidate = candidate.strip()
            if candidate == ESCALATE_TOKEN:
                return ParsedDirectDecision("escalate", None, "json")
            if candidate in by_name:
                return ParsedDirectDecision("action", by_name[candidate], "json")

    return ParsedDirectDecision("invalid", None, "no_valid_direct_decision")


def _action_satisfies_case(case: BedroomCase, action: ActionOption | None) -> bool:
    if action is None:
        return False
    required = case.envelope.required_action_roles
    if required:
        return any(action.has_role(role) for role in required)
    # The normal control has no required role in the legacy envelope, so keep
    # the expected continuation explicit rather than treating any action as OK.
    if case.observation is None:
        return action.has_role("continue_skill")
    return True


def run_direct_control_case(
    case: BedroomCase,
    backend: DirectControlBackend,
    *,
    run_index: int = 1,
    max_tokens: int = 8,
) -> tuple[DirectControlCaseResult, str]:
    """Run one always-on SLM decision with no oracle Gate/Budget Controller."""

    state = bedroom_control_state(case)
    prompt = build_direct_control_prompt(state)
    prompt_sha = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    response = backend.choose_action(
        system_prompt=SYSTEM_PROMPT,
        user_prompt=prompt,
        max_tokens=max_tokens,
    )
    parsed = parse_direct_decision(response.text, state.available_actions)

    decision_valid = parsed.kind in {"action", "escalate"}
    local_resolution_pass = parsed.kind == "action" and _action_satisfies_case(case, parsed.action)

    world_deadline = state.hard_deadline_ms
    decision_deadline = (
        max(1, int(world_deadline) - MOTOR_COMMIT_RESERVE_MS)
        if world_deadline is not None
        else None
    )
    deadline_miss = (
        decision_deadline is not None and response.wall_ms > decision_deadline
    )

    # ESCALATE is a routing decision, not completion of the world action. For a
    # hard-deadline hazard it therefore cannot count as survival in this direct-only
    # experiment. A later cascade experiment may execute System-2.
    survived = True
    if world_deadline is not None:
        survived = local_resolution_pass and not deadline_miss

    realtime_pass = local_resolution_pass and survived
    action_roles = () if parsed.action is None else parsed.action.roles
    violated_preconditions = tuple(
        item.id for item in state.skill_preconditions if item.status.value == "violated"
    )
    unknown_preconditions = tuple(
        item.id for item in state.skill_preconditions if item.status.value == "unknown"
    )
    result = DirectControlCaseResult(
        case=case.name,
        run_index=run_index,
        prompt_style=PROMPT_STYLE_VERSION,
        prompt_sha256=prompt_sha,
        oracle_gate_used=False,
        engineered_gate_features_exposed=False,
        decision_kind=parsed.kind,
        action=None if parsed.action is None else parsed.action.name,
        action_roles=action_roles,
        escalated=parsed.kind == "escalate",
        parse_status=parsed.status,
        decision_valid=decision_valid,
        local_resolution_pass=local_resolution_pass,
        world_deadline_ms=world_deadline,
        decision_deadline_ms=decision_deadline,
        deadline_miss=deadline_miss,
        survived=survived,
        realtime_pass=realtime_pass,
        wall_ms=round(response.wall_ms, 3),
        ttft_ms=round(response.ttft_ms, 3),
        prompt_tokens=response.prompt_tokens,
        completion_tokens=response.completion_tokens,
        finish_reason=response.finish_reason,
        model_text=response.text,
        reasoning_text=response.reasoning_text,
        violated_preconditions=violated_preconditions,
        unknown_preconditions=unknown_preconditions,
    )
    return result, prompt


def write_direct_control_report(
    results: list[DirectControlCaseResult],
    prompts: dict[str, str],
    *,
    out_dir: Path,
    model: str,
    backend_name: str,
    health: dict | None = None,
    concurrency: int = 1,
    warmup_runs: int = 0,
    warmup_case: str | None = None,
) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "bedroom_1.direct_real.json"
    csv_path = out_dir / "bedroom_1.direct_real.csv"

    payload = {
        "prompt_style": PROMPT_STYLE_VERSION,
        "model": model,
        "backend": backend_name,
        "health": health,
        "experimental_scope": {
            "always_on_small_model_direct_control": True,
            "structured_control_state": True,
            "affordance_precondition_state": True,
            "precondition_states_are_runtime_derived": True,
            "exception_only_precondition_projection": True,
            "satisfied_preconditions_exposed": False,
            "real_hardware_latency": True,
            "oracle_gate_used": False,
            "oracle_budget_controller_used": False,
            "engineered_gate_features_exposed": False,
            "reasoning_text_used_for_decision": False,
            "system2_escalation_executed": False,
            "note": (
                "This experiment tests whether one small model can directly choose "
                "CONTINUE-equivalent actions, other executable actions, or ESCALATE "
                "without an engineered/oracle meta-control gate. v0.3 keeps full "
                "runtime-derived precondition state for execution/audit but projects "
                "only violated or unknown precondition exceptions to the small model. "
                "These are affordance facts, not cognition labels."
            ),
        },
        "run_config": {
            "concurrency": concurrency,
            "warmup_runs": warmup_runs,
            "warmup_case": warmup_case,
        },
        "prompts": prompts,
        "results": [asdict(result) for result in results],
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    fieldnames = list(asdict(results[0]).keys()) if results else []
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            row = asdict(result)
            row["action_roles"] = "|".join(result.action_roles)
            row["violated_preconditions"] = "|".join(result.violated_preconditions)
            row["unknown_preconditions"] = "|".join(result.unknown_preconditions)
            writer.writerow(row)

    return json_path, csv_path
