from __future__ import annotations

import csv
import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Protocol

from awb.bedroom import (
    ActionOption,
    BedroomCase,
    CognitiveBudgetController,
    CognitiveInterruptGate,
    CognitiveMode,
)


PROMPT_STYLE_VERSION = "bedroom_semantic_action_v0.1"


@dataclass(slots=True, frozen=True)
class SemanticModelResponse:
    """One real model response to a bounded game-agent action choice."""

    text: str
    reasoning_text: str
    wall_ms: float
    ttft_ms: float
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    finish_reason: str | None = None


class SemanticDecisionBackend(Protocol):
    def choose_action(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
    ) -> SemanticModelResponse:
        ...


@dataclass(slots=True, frozen=True)
class ParsedAction:
    action: ActionOption | None
    status: str
    matched_from: str | None = None


@dataclass(slots=True, frozen=True)
class SemanticCaseResult:
    case: str
    run_index: int
    prompt_style: str
    prompt_sha256: str | None
    engineered_gate_features_exposed: bool
    oracle_interrupt: bool
    oracle_mode: str
    action: str | None
    action_roles: tuple[str, ...]
    parse_status: str
    semantic_pass: bool
    cognitive_budget_ms: int
    cognitive_budget_miss: bool
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


SYSTEM_PROMPT = """You control one NPC inside a game world.
The game engine has already limited you to actions that are physically executable now.
Choose the single action that best serves the NPC's current goal given only the perceived change below.
Do not invent objects, actions, observations, or hidden world facts.
Do not explain your answer. Return exactly one action id from AVAILABLE ACTIONS and nothing else."""


def build_bedroom_semantic_prompt(
    case: BedroomCase,
    *,
    current_phase: str = "sit_on_bed",
) -> str:
    """Build the semantic action prompt without oracle gate/budget scalars.

    Deliberately omitted: case name, goal_relevance, prediction_error,
    immediate_risk, novelty, uncertainty, skill_validity, action_ambiguity,
    planning_horizon, cognitive mode, and reference-policy roles.
    """

    observation = case.observation
    if observation is None:
        raise ValueError("The normal control case requires no semantic model call")

    attrs = _render_attributes(observation.attributes)
    actions = "\n".join(
        f"- {action.name}: {action.description}" for action in observation.available_actions
    )
    deadline_line = ""
    if observation.deadline_ms is not None:
        deadline_line = (
            f"\nObserved time remaining before contact: {int(observation.deadline_ms)} ms."
        )

    return (
        "CURRENT GOAL\n"
        "sleep\n\n"
        "CURRENT SKILL\n"
        "sleep_at_bed\n\n"
        "CURRENT SKILL PHASE\n"
        f"{current_phase}\n\n"
        "PERCEIVED CHANGE\n"
        f"{observation.summary}\n"
        f"{attrs}"
        f"{deadline_line}\n\n"
        "AVAILABLE ACTIONS\n"
        f"{actions}\n\n"
        "ACTION ID"
    )


def parse_action_choice(
    text: str,
    actions: Iterable[ActionOption],
    *,
    reasoning_text: str = "",
) -> ParsedAction:
    """Parse one bounded action choice without silently inventing a fallback."""

    options = tuple(actions)
    by_name = {a.name: a for a in options}
    cleaned = text.strip().strip("`").strip()

    if cleaned in by_name:
        return ParsedAction(by_name[cleaned], "exact", "content")

    # Accept a tiny JSON wrapper because some instruct models insist on one
    # even when asked for a bare action id.  Do not accept arbitrary fields.
    try:
        payload = json.loads(cleaned)
    except (json.JSONDecodeError, TypeError):
        payload = None
    if isinstance(payload, dict):
        candidate = payload.get("action") or payload.get("action_id")
        if isinstance(candidate, str) and candidate.strip() in by_name:
            return ParsedAction(by_name[candidate.strip()], "json", "content")

    def unique_match(source: str, source_name: str) -> ParsedAction | None:
        hits = [a for a in options if a.name in source]
        if len(hits) == 1:
            return ParsedAction(hits[0], "embedded", source_name)
        if len(hits) > 1:
            return ParsedAction(None, "ambiguous", source_name)
        return None

    matched = unique_match(cleaned, "content")
    if matched is not None:
        return matched

    # Some reasoning-capable llama.cpp chat templates separate reasoning from
    # final content. This fallback is auditable in matched_from; it is not
    # treated as an exact answer.
    if reasoning_text:
        matched = unique_match(reasoning_text, "reasoning")
        if matched is not None:
            return matched

    return ParsedAction(None, "no_valid_action", None)


def evaluate_semantic_action(case: BedroomCase, action: ActionOption | None) -> bool:
    required = case.envelope.required_action_roles
    if not required:
        return action is not None
    return action is not None and any(action.has_role(role) for role in required)


def run_semantic_case(
    case: BedroomCase,
    backend: SemanticDecisionBackend,
    *,
    run_index: int = 1,
    max_tokens: int = 16,
) -> tuple[SemanticCaseResult, str | None]:
    """Run one real semantic action choice.

    v0.1 intentionally keeps the existing oracle-feature interrupt/budget
    controller fixed so this experiment isolates *action semantics + hardware
    latency*. The model is not told those engineered scalar features or the
    assigned cognition mode.
    """

    if case.observation is None:
        result = SemanticCaseResult(
            case=case.name,
            run_index=run_index,
            prompt_style=PROMPT_STYLE_VERSION,
            prompt_sha256=None,
            engineered_gate_features_exposed=False,
            oracle_interrupt=False,
            oracle_mode=CognitiveMode.AUTOMATIC.value,
            action="continue_sleep_skill",
            action_roles=("continue_skill",),
            parse_status="no_model_call_control",
            semantic_pass=True,
            cognitive_budget_ms=0,
            cognitive_budget_miss=False,
            world_deadline_ms=None,
            decision_deadline_ms=None,
            deadline_miss=False,
            survived=True,
            realtime_pass=True,
            wall_ms=0.0,
            ttft_ms=0.0,
            prompt_tokens=0,
            completion_tokens=0,
            finish_reason="control",
            model_text="",
            reasoning_text="",
        )
        return result, None

    gate = CognitiveInterruptGate()
    budget_controller = CognitiveBudgetController()
    assessment = gate.assess(case.observation)
    budget = budget_controller.assign(case.observation, assessment)

    if not assessment.interrupt:
        # The semantic backend must not be called merely to confirm that no
        # cognition was needed.  This keeps the control boundary explicit.
        result = SemanticCaseResult(
            case=case.name,
            run_index=run_index,
            prompt_style=PROMPT_STYLE_VERSION,
            prompt_sha256=None,
            engineered_gate_features_exposed=False,
            oracle_interrupt=False,
            oracle_mode=budget.mode.value,
            action="continue_sleep_skill",
            action_roles=("continue_skill",),
            parse_status="no_model_call_gate",
            semantic_pass=True,
            cognitive_budget_ms=0,
            cognitive_budget_miss=False,
            world_deadline_ms=case.observation.deadline_ms,
            decision_deadline_ms=None,
            deadline_miss=False,
            survived=True,
            realtime_pass=True,
            wall_ms=0.0,
            ttft_ms=0.0,
            prompt_tokens=0,
            completion_tokens=0,
            finish_reason="gate_no_interrupt",
            model_text="",
            reasoning_text="",
        )
        return result, None

    prompt = build_bedroom_semantic_prompt(case)
    prompt_sha = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    response = backend.choose_action(
        system_prompt=SYSTEM_PROMPT,
        user_prompt=prompt,
        max_tokens=max_tokens,
    )
    parsed = parse_action_choice(
        response.text,
        case.observation.available_actions,
        reasoning_text=response.reasoning_text,
    )
    semantic_pass = evaluate_semantic_action(case, parsed.action)

    world_deadline = case.observation.deadline_ms
    decision_deadline: int | None = None
    if budget.stages:
        # For emergency, the synthetic contract reserves motor/action time and
        # therefore has a tighter decision deadline than world contact time.
        # For fast/deliberate modes this is the stage/budget contract, not a
        # physical hazard deadline.
        if budget.mode == CognitiveMode.EMERGENCY:
            decision_deadline = budget.stages[-1].deadline_ms
        else:
            decision_deadline = budget.budget_ms

    deadline_miss = (
        decision_deadline is not None and response.wall_ms > decision_deadline
    )
    cognitive_budget_miss = budget.budget_ms > 0 and response.wall_ms > budget.budget_ms

    action_roles = () if parsed.action is None else parsed.action.roles
    survived = True
    if world_deadline is not None:
        # Emergency action success requires both a protective semantic choice
        # and completion within the decision budget that reserves 15ms for the
        # minimal motor/action commit in Bedroom-1 v0.1.
        survived = semantic_pass and not deadline_miss

    realtime_pass = semantic_pass and survived
    result = SemanticCaseResult(
        case=case.name,
        run_index=run_index,
        prompt_style=PROMPT_STYLE_VERSION,
        prompt_sha256=prompt_sha,
        engineered_gate_features_exposed=False,
        oracle_interrupt=assessment.interrupt,
        oracle_mode=budget.mode.value,
        action=None if parsed.action is None else parsed.action.name,
        action_roles=action_roles,
        parse_status=parsed.status,
        semantic_pass=semantic_pass,
        cognitive_budget_ms=budget.budget_ms,
        cognitive_budget_miss=cognitive_budget_miss,
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
    )
    return result, prompt


def write_semantic_report(
    results: list[SemanticCaseResult],
    prompts: dict[str, str],
    *,
    out_dir: Path,
    model: str,
    backend_name: str,
    health: dict | None = None,
) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "bedroom_1.semantic_real.json"
    csv_path = out_dir / "bedroom_1.semantic_real.csv"

    payload = {
        "prompt_style": PROMPT_STYLE_VERSION,
        "model": model,
        "backend": backend_name,
        "health": health,
        "experimental_scope": {
            "real_model_action_selection": True,
            "real_hardware_latency": True,
            "oracle_feature_interrupt_gate": True,
            "oracle_feature_budget_controller": True,
            "model_selects_interrupt": False,
            "model_selects_cognition_depth": False,
            "reference_decision_policy_used": False,
        },
        "prompts": prompts,
        "results": [_jsonable_result(result) for result in results],
    }
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    rows = [_jsonable_result(result) for result in results]
    fieldnames = [
        "case",
        "run_index",
        "oracle_mode",
        "action",
        "parse_status",
        "semantic_pass",
        "realtime_pass",
        "wall_ms",
        "ttft_ms",
        "cognitive_budget_ms",
        "cognitive_budget_miss",
        "world_deadline_ms",
        "decision_deadline_ms",
        "deadline_miss",
        "survived",
        "prompt_tokens",
        "completion_tokens",
        "finish_reason",
        "prompt_sha256",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return json_path, csv_path


def _jsonable_result(result: SemanticCaseResult) -> dict:
    row = asdict(result)
    row["action_roles"] = list(result.action_roles)
    return row


def _render_attributes(attributes: dict) -> str:
    if not attributes:
        return ""
    lines = ["OBSERVED ATTRIBUTES"]
    for key in sorted(attributes):
        value = attributes[key]
        lines.append(f"- {key}: {_render_value(value)}")
    return "\n" + "\n".join(lines) + "\n"


def _render_value(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True, ensure_ascii=False)
    return str(value)
