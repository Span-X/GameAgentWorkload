from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from awb.affordance import (
    GenericAffordanceResolver,
    ObservedRelation,
    PreconditionStatus,
    SkillPreconditionSpec,
    SkillPreconditionState,
)
from awb.bedroom import ActionOption, BedroomCase


BEDROOM_SKILL_EXPECTATION = (
    "sleep_at_bed normally continues automatically while the intended sleeping "
    "surface remains usable, the agent is not facing an immediate physical "
    "hazard, and no consequential unresolved interaction meaningfully conflicts "
    "with the current sleep goal."
)

# These describe objective execution requirements of the skill. They are not
# cognition-mode labels and do not encode any Bedroom case name or entity class.
BEDROOM_SKILL_PRECONDITIONS = (
    SkillPreconditionSpec(
        id="sleep_surface_clear",
        description="The intended sleep surface is not occupied or blocked.",
        kind="resource_clear",
        target="intended_sleep_surface",
    ),
    SkillPreconditionSpec(
        id="body_trajectory_clear",
        description="No observed trajectory currently intersects the agent body.",
        kind="trajectory_clear",
        target="agent_body",
    ),
)

BEDROOM_BASELINE_PRECONDITIONS = {
    "sleep_surface_clear": PreconditionStatus.SATISFIED,
    "body_trajectory_clear": PreconditionStatus.SATISFIED,
}


@dataclass(slots=True, frozen=True)
class AgentControlState:
    """Domain-neutral state exposed to a metacognitive/direct controller.

    The state excludes Bedroom-1's engineered/oracle gate scalars. A domain
    adapter may retain perceived facts, structural relations, objective skill
    precondition state, current goal/skill, executable actions, and hard
    deadline facts, but must not pre-solve cognition depth for the controller.
    Prompt projection is exception-oriented: stable satisfied preconditions stay
    in runtime state but are not redundantly emitted to the small model.
    """

    current_goal: str
    current_skill: str
    skill_phase: str
    skill_expectation: str
    skill_preconditions: tuple[SkillPreconditionState, ...]
    perceived_change: str | None
    observed_attributes: dict[str, Any]
    observed_relations: tuple[ObservedRelation, ...]
    available_actions: tuple[ActionOption, ...]
    hard_deadline_ms: int | None = None

    def prompt_payload(self) -> dict[str, Any]:
        """Project persistent runtime state into a decision-relevant delta view.

        The runtime keeps the complete skill-precondition state for auditing and
        execution. The small-model prompt receives only exceptions: violated or
        unknown requirements. Stable SATISFIED entries are intentionally omitted
        so nominal state does not overwhelm a new decision-relevant observation.
        """

        precondition_exceptions = tuple(
            state
            for state in self.skill_preconditions
            if state.status in {PreconditionStatus.VIOLATED, PreconditionStatus.UNKNOWN}
        )
        return {
            "current_goal": self.current_goal,
            "current_skill": self.current_skill,
            "skill_phase": self.skill_phase,
            "skill_expectation": self.skill_expectation,
            "precondition_exceptions": [
                state.prompt_payload() for state in precondition_exceptions
            ],
            "perceived_change": self.perceived_change,
            "observed_attributes": self.observed_attributes,
            "observed_relations": [
                relation.prompt_payload() for relation in self.observed_relations
            ],
            "hard_deadline_ms": self.hard_deadline_ms,
            "executable_actions": [
                {"id": action.name, "description": action.description}
                for action in self.available_actions
            ],
        }


def _bedroom_relations(case: BedroomCase) -> tuple[ObservedRelation, ...]:
    """Normalize Bedroom perception fields into structural relations.

    The mapping is relation-based, not case/entity-name based. In a real game
    integration these relations should normally come directly from the engine's
    perception/affordance layer rather than from string normalization.
    """

    observation = case.observation
    if observation is None:
        return ()

    attrs = observation.attributes
    relations: list[ObservedRelation] = []

    if attrs.get("relation") == "on_goal_surface":
        confidence = attrs.get("classification_confidence")
        relations.append(
            ObservedRelation(
                subject="observed_entity",
                predicate="occupies",
                object="intended_sleep_surface",
                confidence=float(confidence) if isinstance(confidence, (int, float)) else None,
            )
        )

    if attrs.get("trajectory_relation") == "intersects_self":
        confidence = attrs.get("confidence")
        relations.append(
            ObservedRelation(
                subject="observed_object_trajectory",
                predicate="trajectory_intersects",
                object="agent_body",
                confidence=float(confidence) if isinstance(confidence, (int, float)) else None,
            )
        )

    if attrs.get("interaction") == "discussion_requested":
        relations.append(
            ObservedRelation(
                subject="observed_person",
                predicate="initiates_interaction",
                object="agent",
            )
        )

    return tuple(relations)


def bedroom_control_state(
    case: BedroomCase,
    *,
    current_phase: str = "sit_on_bed",
) -> AgentControlState:
    """Adapt Bedroom-1 into AgentControlState without reading oracle features."""

    from scenarios.bedroom_1 import CONTINUE_SLEEP

    relations = _bedroom_relations(case)
    preconditions = GenericAffordanceResolver().resolve(
        BEDROOM_SKILL_PRECONDITIONS,
        relations,
        baseline=BEDROOM_BASELINE_PRECONDITIONS,
    )

    observation = case.observation
    if observation is None:
        # The direct-controller experiment intentionally calls the SLM even for
        # the no-change control, unlike the event-triggered oracle baseline.
        return AgentControlState(
            current_goal="sleep",
            current_skill="sleep_at_bed",
            skill_phase=current_phase,
            skill_expectation=BEDROOM_SKILL_EXPECTATION,
            skill_preconditions=preconditions,
            perceived_change=None,
            observed_attributes={},
            observed_relations=relations,
            available_actions=(CONTINUE_SLEEP,),
            hard_deadline_ms=None,
        )

    # Deliberately read only perception-bounded descriptive fields, normalized
    # structural relations, objective precondition state, and the executable
    # affordance/deadline contract. Do not read goal_relevance, prediction_error,
    # immediate_risk, novelty, uncertainty, skill_validity, action_ambiguity, or
    # planning_horizon here.
    return AgentControlState(
        current_goal="sleep",
        current_skill="sleep_at_bed",
        skill_phase=current_phase,
        skill_expectation=BEDROOM_SKILL_EXPECTATION,
        skill_preconditions=preconditions,
        perceived_change=observation.summary,
        observed_attributes=dict(observation.attributes),
        observed_relations=relations,
        available_actions=tuple(observation.available_actions),
        hard_deadline_ms=observation.deadline_ms,
    )
