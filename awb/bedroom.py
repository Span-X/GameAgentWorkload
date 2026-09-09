from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class CognitiveMode(StrEnum):
    AUTOMATIC = "automatic"
    FAST = "fast"
    DELIBERATE = "deliberate"
    EMERGENCY = "emergency"


@dataclass(slots=True, frozen=True)
class ActionOption:
    """An executable action exposed by the current world state.

    Roles are semantic affordance tags used by the deterministic reference
    policy. They are not scenario names and do not encode a complete answer.
    A learned/model backend may ignore roles and decide from the observation
    plus action descriptions instead.
    """

    name: str
    roles: tuple[str, ...]
    description: str

    def has_role(self, role: str) -> bool:
        return role in self.roles


@dataclass(slots=True, frozen=True)
class ObservationDelta:
    """A perception-bounded change relevant to a currently executing skill.

    All scalar features are normalized to [0, 1]. The benchmark deliberately
    receives a compact observation rather than omniscient world truth. v0.5
    fixes the perception boundary so cognition/runtime behavior can be studied
    without simultaneously benchmarking vision.
    """

    delta_id: str
    summary: str
    goal_relevance: float
    prediction_error: float
    immediate_risk: float
    novelty: float
    uncertainty: float
    skill_validity: float
    action_ambiguity: float
    planning_horizon: float
    available_actions: tuple[ActionOption, ...]
    deadline_ms: int | None = None
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class InterruptAssessment:
    interrupt: bool
    score: float
    reasons: tuple[str, ...]


@dataclass(slots=True, frozen=True)
class CognitiveStage:
    name: str
    cognitive_layer: str
    deadline_ms: int
    input_tokens: int
    output_tokens: int
    model_class: str


@dataclass(slots=True, frozen=True)
class CognitiveBudget:
    mode: CognitiveMode
    complexity_score: float
    budget_ms: int
    stages: tuple[CognitiveStage, ...]


@dataclass(slots=True, frozen=True)
class BehavioralEnvelope:
    """Architectural correctness constraints, not a single scripted answer."""

    expected_interrupt: bool
    acceptable_modes: tuple[CognitiveMode, ...]
    required_action_roles: tuple[str, ...] = ()
    require_sleep_completed: bool = False
    require_survival: bool = False
    require_social_engagement: bool = False
    max_brain_calls: int | None = None


@dataclass(slots=True, frozen=True)
class BedroomCase:
    name: str
    description: str
    observation: ObservationDelta | None
    envelope: BehavioralEnvelope
    inject_at_ms: int = 250


class CognitiveInterruptGate:
    """Cheap gate deciding whether automatic skill execution must pause.

    The gate is intentionally generic: it consumes normalized features rather
    than case/event labels such as "cockroach" or "bullet".
    """

    def __init__(self, *, threshold: float = 0.48) -> None:
        self.threshold = float(threshold)

    def assess(self, observation: ObservationDelta) -> InterruptAssessment:
        invalidity = 1.0 - _clamp(observation.skill_validity)
        score = (
            0.28 * _clamp(observation.prediction_error)
            + 0.24 * _clamp(observation.goal_relevance)
            + 0.25 * _clamp(observation.immediate_risk)
            + 0.18 * invalidity
            + 0.05 * _clamp(observation.novelty)
        )
        reasons: list[str] = []
        if observation.prediction_error >= 0.65:
            reasons.append("prediction_error")
        if observation.goal_relevance >= 0.65:
            reasons.append("goal_relevant")
        if observation.immediate_risk >= 0.70:
            reasons.append("risk")
        if observation.skill_validity <= 0.45:
            reasons.append("skill_invalidated")
        if observation.novelty >= 0.65:
            reasons.append("novelty")

        # Hard safety/validity conditions can interrupt even when a weighted
        # score is diluted by otherwise-low features.
        interrupt = (
            score >= self.threshold
            or observation.immediate_risk >= 0.90
            or observation.skill_validity <= 0.05
        )
        return InterruptAssessment(interrupt, round(score, 4), tuple(reasons))


class CognitiveBudgetController:
    """Allocate cognition depth after an interrupt has already been justified."""

    def assign(self, observation: ObservationDelta, assessment: InterruptAssessment) -> CognitiveBudget:
        if not assessment.interrupt:
            return CognitiveBudget(CognitiveMode.AUTOMATIC, 0.0, 0, ())

        complexity = (
            0.32 * _clamp(observation.uncertainty)
            + 0.32 * _clamp(observation.action_ambiguity)
            + 0.24 * _clamp(observation.planning_horizon)
            + 0.12 * _clamp(observation.novelty)
        )
        complexity = round(complexity, 4)

        # A physical deadline plus serious risk defines an emergency decision
        # class. This does not mandate a rule implementation: the backend can
        # be rules, a tiny NN, an action head, an SLM/LLM, or future hardware.
        if observation.deadline_ms is not None and observation.immediate_risk >= 0.75:
            budget_ms = max(1, min(max(1, int(observation.deadline_ms) - 15), 120))
            stage = CognitiveStage(
                name="bedroom_emergency_decision",
                cognitive_layer="emergency",
                deadline_ms=budget_ms,
                input_tokens=96,
                output_tokens=2,
                model_class="tiny",
            )
            return CognitiveBudget(CognitiveMode.EMERGENCY, complexity, budget_ms, (stage,))

        if complexity < 0.42:
            budget_ms = 280
            stage = CognitiveStage(
                name="bedroom_fast_decision",
                cognitive_layer="fast",
                deadline_ms=budget_ms,
                input_tokens=192 + round(observation.goal_relevance * 64),
                output_tokens=8,
                model_class="tiny",
            )
            return CognitiveBudget(CognitiveMode.FAST, complexity, budget_ms, (stage,))

        # Deliberation uses progressive deepening: a short triage call first,
        # followed by a longer stage only because the generic complexity score
        # remains high. This avoids assigning a fixed large budget to all
        # interrupted situations.
        budget_ms = 1_800 + round(observation.planning_horizon * 1_000)
        triage = CognitiveStage(
            name="bedroom_fast_triage",
            cognitive_layer="fast",
            deadline_ms=300,
            input_tokens=256,
            output_tokens=8,
            model_class="tiny",
        )
        deliberate = CognitiveStage(
            name="bedroom_deliberation",
            cognitive_layer="deliberate",
            deadline_ms=budget_ms,
            input_tokens=768 + round(complexity * 512),
            output_tokens=48 + round(complexity * 32),
            model_class="small",
        )
        return CognitiveBudget(
            CognitiveMode.DELIBERATE,
            complexity,
            budget_ms,
            (triage, deliberate),
        )


class ReferenceDecisionPolicy:
    """Deterministic non-LLM policy used only to close synthetic traces.

    It is a baseline implementation, not benchmark truth. It does not inspect
    case names. The semantic LLM/backend experiments can replace this policy
    while preserving the same observation and action contracts.
    """

    def choose(self, observation: ObservationDelta, mode: CognitiveMode) -> ActionOption:
        actions = observation.available_actions
        if not actions:
            raise ValueError("Bedroom observation exposes no executable actions")

        def first(role: str) -> ActionOption | None:
            return next((a for a in actions if a.has_role(role)), None)

        if mode == CognitiveMode.EMERGENCY:
            action = first("protective")
            if action is not None:
                return action

        if observation.skill_validity < 0.55:
            action = first("restore_precondition")
            if action is not None:
                return action

        if observation.action_ambiguity >= 0.60 or observation.planning_horizon >= 0.60:
            action = first("social_engagement")
            if action is not None:
                return action

        if observation.novelty >= 0.65 or observation.uncertainty >= 0.65:
            action = first("inspect")
            if action is not None:
                return action

        return first("continue_skill") or actions[0]


def evaluate_envelope(
    envelope: BehavioralEnvelope,
    *,
    interrupt_triggered: bool,
    mode: CognitiveMode,
    action: ActionOption | None,
    sleep_completed: bool,
    survived: bool,
    social_engagement: bool,
    brain_calls: int,
) -> tuple[bool, tuple[str, ...]]:
    violations: list[str] = []
    if interrupt_triggered != envelope.expected_interrupt:
        violations.append("interrupt_mismatch")
    if mode not in envelope.acceptable_modes:
        violations.append("cognitive_mode_mismatch")
    if envelope.required_action_roles:
        if action is None or not any(action.has_role(r) for r in envelope.required_action_roles):
            violations.append("required_action_role_missing")
    if envelope.require_sleep_completed and not sleep_completed:
        violations.append("sleep_not_completed")
    if envelope.require_survival and not survived:
        violations.append("survival_failed")
    if envelope.require_social_engagement and not social_engagement:
        violations.append("social_engagement_missing")
    if envelope.max_brain_calls is not None and brain_calls > envelope.max_brain_calls:
        violations.append("brain_call_budget_exceeded")
    return (not violations, tuple(violations))


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
