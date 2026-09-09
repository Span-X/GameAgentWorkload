from __future__ import annotations

from dataclasses import dataclass

from awb.bedroom import (
    ActionOption,
    BehavioralEnvelope,
    BedroomCase,
    CognitiveBudget,
    CognitiveBudgetController,
    CognitiveInterruptGate,
    CognitiveMode,
    ObservationDelta,
    ReferenceDecisionPolicy,
    evaluate_envelope,
)
from awb.types import ActivationLevel, AgentState, InferenceRequest, RequestPriority, WorldEvent

AGENT_ID = 0
BEDROOM_ID = 101


PROTECT_SELF = ActionOption(
    "protect_self",
    ("protective",),
    "Perform an immediately available protective movement before impact.",
)
MOVE_TO_COVER = ActionOption(
    "move_to_cover",
    ("protective", "restore_precondition"),
    "Move into nearby cover using an executable movement skill.",
)
CLEAR_SURFACE = ActionOption(
    "clear_sleep_surface",
    ("restore_precondition",),
    "Remove or displace the obstruction from the intended sleep surface.",
)
INSPECT = ActionOption(
    "inspect_change",
    ("inspect",),
    "Inspect the unexpected observation before committing to another plan.",
)
ENGAGE_SOCIAL = ActionOption(
    "engage_socially",
    ("social_engagement",),
    "Pause the sleep routine and engage with the person who initiated interaction.",
)
DEFER_SOCIAL = ActionOption(
    "defer_interaction",
    ("social_engagement", "continue_skill"),
    "Acknowledge the person but defer the discussion and preserve the sleep goal.",
)
CONTINUE_SLEEP = ActionOption(
    "continue_sleep_skill",
    ("continue_skill",),
    "Continue the current sleep-at-bed skill.",
)


def core_cases() -> dict[str, BedroomCase]:
    return {
        "normal": BedroomCase(
            name="normal",
            description="Control: the agent is beside a usable bed and nothing unexpected happens.",
            observation=None,
            envelope=BehavioralEnvelope(
                expected_interrupt=False,
                acceptable_modes=(CognitiveMode.AUTOMATIC,),
                require_sleep_completed=True,
                max_brain_calls=0,
            ),
        ),
        "roach": BedroomCase(
            name="roach",
            description="A small moving unwanted entity appears on the intended sleep surface.",
            observation=ObservationDelta(
                delta_id="surface_change_01",
                summary="small erratically moving entity observed on intended sleep surface",
                goal_relevance=0.92,
                prediction_error=0.93,
                immediate_risk=0.18,
                novelty=0.20,
                uncertainty=0.12,
                skill_validity=0.28,
                action_ambiguity=0.18,
                planning_horizon=0.08,
                available_actions=(CLEAR_SURFACE, INSPECT, CONTINUE_SLEEP),
                attributes={
                    "entity_scale": "small",
                    "motion": "erratic",
                    "relation": "on_goal_surface",
                    "classification": "pest_like",
                    "classification_confidence": 0.88,
                },
            ),
            envelope=BehavioralEnvelope(
                expected_interrupt=True,
                acceptable_modes=(CognitiveMode.FAST,),
                required_action_roles=("restore_precondition",),
                require_sleep_completed=True,
                max_brain_calls=1,
            ),
        ),
        "projectile": BedroomCase(
            name="projectile",
            description="A fast object is observed on a trajectory intersecting the agent.",
            observation=ObservationDelta(
                delta_id="trajectory_change_01",
                summary="fast object trajectory intersects agent body volume",
                goal_relevance=1.00,
                prediction_error=0.98,
                immediate_risk=1.00,
                novelty=0.12,
                uncertainty=0.10,
                skill_validity=0.00,
                action_ambiguity=0.24,
                planning_horizon=0.05,
                available_actions=(PROTECT_SELF, MOVE_TO_COVER),
                deadline_ms=120,
                attributes={
                    "trajectory_relation": "intersects_self",
                    "time_to_contact_ms": 120,
                    "estimated_speed_class": "very_fast",
                    "confidence": 0.97,
                },
            ),
            envelope=BehavioralEnvelope(
                expected_interrupt=True,
                acceptable_modes=(CognitiveMode.EMERGENCY,),
                required_action_roles=("protective",),
                require_survival=True,
                max_brain_calls=1,
            ),
        ),
        "complex_visitor": BedroomCase(
            name="complex_visitor",
            description="A trusted person initiates a consequential, ambiguous discussion before sleep.",
            observation=ObservationDelta(
                delta_id="social_change_01",
                summary="trusted person initiates consequential ambiguous interaction",
                goal_relevance=0.82,
                prediction_error=0.72,
                immediate_risk=0.04,
                novelty=0.22,
                uncertainty=0.86,
                skill_validity=0.52,
                action_ambiguity=0.92,
                planning_horizon=0.90,
                available_actions=(ENGAGE_SOCIAL, DEFER_SOCIAL, CONTINUE_SLEEP),
                attributes={
                    "relation": "trusted_person",
                    "interaction": "discussion_requested",
                    "stakes": "high",
                    "future_consequences": "material",
                },
            ),
            envelope=BehavioralEnvelope(
                expected_interrupt=True,
                acceptable_modes=(CognitiveMode.DELIBERATE,),
                required_action_roles=("social_engagement",),
                require_social_engagement=True,
                max_brain_calls=2,
            ),
        ),
    }


@dataclass(slots=True)
class _RuntimeState:
    active_skill: str = "sleep_at_bed"
    phase_index: int = 0
    paused: bool = False
    deferred_phase: int | None = None
    interrupt_triggered: bool = False
    mode: CognitiveMode = CognitiveMode.AUTOMATIC
    budget: CognitiveBudget | None = None
    observation: ObservationDelta | None = None
    chosen_action: ActionOption | None = None
    brain_calls: int = 0
    survived: bool = True
    protected: bool = False
    social_engagement: bool = False
    sleep_completed: bool = False
    finalized: bool = False


class BedroomScenario:
    PHASES = ("face_bed", "sit_on_bed", "lie_down", "sleep")
    PHASE_DELAYS_MS = (180, 200, 260, 0)

    def __init__(self, case: BedroomCase) -> None:
        self.case = case
        self.name = f"bedroom_1_{case.name}"
        self.gate = CognitiveInterruptGate()
        self.budget_controller = CognitiveBudgetController()
        self.reference_policy = ReferenceDecisionPolicy()
        self.state = _RuntimeState()
        self._request_stage: dict[str, int] = {}
        self._world = None

    def build(self, world) -> None:
        # Reset so the same scenario object is deterministic across sweeps/reruns.
        self.state = _RuntimeState()
        self._request_stage = {}
        self._world = world
        world.set_simulation_end(4_000)
        world.add_agent(
            AgentState(
                agent_id=AGENT_ID,
                location_id=BEDROOM_ID,
                activation=ActivationLevel.BACKGROUND,
                current_goal="sleep",
            )
        )
        world.add_inference_completion_listener(self._on_inference_completed)
        world.schedule(0, "bedroom_skill_step", target_id=AGENT_ID, location_id=BEDROOM_ID, payload={"phase": 0})
        if self.case.observation is not None:
            world.schedule(
                self.case.inject_at_ms,
                "bedroom_observation_delta",
                target_id=AGENT_ID,
                location_id=BEDROOM_ID,
            )
            if self.case.observation.deadline_ms is not None:
                world.schedule(
                    self.case.inject_at_ms + self.case.observation.deadline_ms,
                    "bedroom_hazard_deadline",
                    target_id=AGENT_ID,
                    location_id=BEDROOM_ID,
                )
        world.schedule(3_500, "bedroom_finalize", target_id=AGENT_ID, location_id=BEDROOM_ID)

    def handle(self, world, event: WorldEvent) -> None:
        if event.event_type == "bedroom_skill_step":
            self._handle_skill_step(world, int(event.payload.get("phase", 0)))
        elif event.event_type == "bedroom_observation_delta":
            self._handle_observation_delta(world)
        elif event.event_type == "bedroom_hazard_deadline":
            self._handle_hazard_deadline(world)
        elif event.event_type == "bedroom_finalize":
            self._finalize(world)

    def _handle_skill_step(self, world, phase: int) -> None:
        if self.state.sleep_completed or not self.state.survived:
            return
        if self.state.paused:
            self.state.deferred_phase = phase
            world.trace.emit(
                world.clock_ms,
                "skill_step_deferred",
                agent_id=AGENT_ID,
                skill=self.state.active_skill,
                phase=self.PHASES[phase],
                reason="cognitive_interrupt",
            )
            return

        self.state.phase_index = phase
        phase_name = self.PHASES[phase]
        world.trace.emit(
            world.clock_ms,
            "skill_step",
            agent_id=AGENT_ID,
            skill=self.state.active_skill,
            phase=phase_name,
            execution_mode="automatic_skill_runtime",
        )
        if phase_name == "sleep":
            self.state.sleep_completed = True
            world.trace.emit(
                world.clock_ms,
                "skill_completed",
                agent_id=AGENT_ID,
                skill=self.state.active_skill,
                goal="sleep",
            )
            return

        delay = self.PHASE_DELAYS_MS[phase]
        world.schedule(
            world.clock_ms + delay,
            "bedroom_skill_step",
            target_id=AGENT_ID,
            location_id=BEDROOM_ID,
            payload={"phase": phase + 1},
        )

    def _handle_observation_delta(self, world) -> None:
        observation = self.case.observation
        if observation is None:
            return
        self.state.observation = observation
        world.trace.emit(
            world.clock_ms,
            "observation_delta",
            agent_id=AGENT_ID,
            delta_id=observation.delta_id,
            summary=observation.summary,
            perception_boundary="controlled_structured_observation",
            attributes=observation.attributes,
            goal_relevance=observation.goal_relevance,
            prediction_error=observation.prediction_error,
            immediate_risk=observation.immediate_risk,
            novelty=observation.novelty,
            uncertainty=observation.uncertainty,
            skill_validity=observation.skill_validity,
            action_ambiguity=observation.action_ambiguity,
            planning_horizon=observation.planning_horizon,
            deadline_ms=observation.deadline_ms,
            available_actions=[a.name for a in observation.available_actions],
        )

        assessment = self.gate.assess(observation)
        self.state.interrupt_triggered = assessment.interrupt
        world.trace.emit(
            world.clock_ms,
            "cognitive_interrupt_assessed",
            agent_id=AGENT_ID,
            delta_id=observation.delta_id,
            interrupt=assessment.interrupt,
            score=assessment.score,
            reasons=list(assessment.reasons),
            gate="generic_feature_gate_v0.1",
        )
        budget = self.budget_controller.assign(observation, assessment)
        self.state.mode = budget.mode
        self.state.budget = budget
        world.trace.emit(
            world.clock_ms,
            "cognitive_budget_assigned",
            agent_id=AGENT_ID,
            mode=budget.mode.value,
            complexity_score=budget.complexity_score,
            budget_ms=budget.budget_ms,
            stages=[stage.name for stage in budget.stages],
        )

        if not assessment.interrupt:
            return

        self.state.paused = True
        world.trace.emit(
            world.clock_ms,
            "skill_paused",
            agent_id=AGENT_ID,
            skill=self.state.active_skill,
            reason="decision_relevant_deviation",
        )
        self._enqueue_stage(world, 0)

    def _enqueue_stage(self, world, stage_index: int) -> None:
        assert self.state.budget is not None
        stage = self.state.budget.stages[stage_index]
        priority = RequestPriority.CRITICAL if self.state.mode == CognitiveMode.EMERGENCY else RequestPriority.HIGH
        req = world.create_request(
            agent_id=AGENT_ID,
            task_type=stage.name,
            priority=priority,
            deadline_ms=stage.deadline_ms,
            input_tokens=stage.input_tokens,
            output_tokens=stage.output_tokens,
            model_class=stage.model_class,
            cognitive_domain="bedroom_decision",
            cognitive_layer=stage.cognitive_layer,
        )
        self._request_stage[req.request_id] = stage_index
        self.state.brain_calls += 1
        world.trace.emit(
            world.clock_ms,
            "brain_call_dispatched",
            agent_id=AGENT_ID,
            request_id=req.request_id,
            stage=stage.name,
            mode=self.state.mode.value,
            case_agnostic_contract=True,
        )

    def _on_inference_completed(self, req: InferenceRequest, now_ms: int) -> None:
        if req.request_id not in self._request_stage:
            return
        # World._on_inference_completed calls listeners after updating the world
        # clock, so trace/apply actions at the actual synthetic decision-ready time.
        stage_index = self._request_stage.pop(req.request_id)
        world = self._world
        if world is None:
            raise RuntimeError("BedroomScenario completion listener fired before build")
        world.trace.emit(
            now_ms,
            "cognitive_stage_completed",
            agent_id=AGENT_ID,
            request_id=req.request_id,
            stage=req.task_type,
            mode=self.state.mode.value,
        )

        assert self.state.budget is not None
        if stage_index + 1 < len(self.state.budget.stages):
            self._enqueue_stage(world, stage_index + 1)
            return

        assert self.state.observation is not None
        action = self.reference_policy.choose(self.state.observation, self.state.mode)
        self.state.chosen_action = action
        world.trace.emit(
            now_ms,
            "decision_committed",
            agent_id=AGENT_ID,
            action=action.name,
            action_roles=list(action.roles),
            mode=self.state.mode.value,
            policy="deterministic_reference_policy_not_benchmark_truth",
        )
        self._apply_action(world, action)

    def _apply_action(self, world, action: ActionOption) -> None:
        world.trace.emit(
            world.clock_ms,
            "action_executed",
            agent_id=AGENT_ID,
            action=action.name,
            roles=list(action.roles),
        )
        if action.has_role("protective"):
            self.state.protected = True
            return
        if action.has_role("social_engagement") and self.state.mode == CognitiveMode.DELIBERATE:
            self.state.social_engagement = True
            self.state.paused = True
            world.trace.emit(
                world.clock_ms,
                "goal_suspended",
                agent_id=AGENT_ID,
                goal="sleep",
                reason="social_engagement",
            )
            return
        if action.has_role("restore_precondition") or action.has_role("continue_skill"):
            self._resume_sleep_skill(world, reason=action.name)

    def _resume_sleep_skill(self, world, *, reason: str) -> None:
        self.state.paused = False
        world.trace.emit(
            world.clock_ms,
            "skill_resumed",
            agent_id=AGENT_ID,
            skill=self.state.active_skill,
            reason=reason,
        )
        phase = self.state.deferred_phase
        if phase is None:
            phase = min(self.state.phase_index + 1, len(self.PHASES) - 1)
        self.state.deferred_phase = None
        world.schedule(
            world.clock_ms + 80,
            "bedroom_skill_step",
            target_id=AGENT_ID,
            location_id=BEDROOM_ID,
            payload={"phase": phase},
        )

    def _handle_hazard_deadline(self, world) -> None:
        if self.state.protected:
            self.state.survived = True
            world.trace.emit(
                world.clock_ms,
                "hazard_resolved",
                agent_id=AGENT_ID,
                result="survived",
                protected_before_deadline=True,
            )
            self._resume_sleep_skill(world, reason="immediate_hazard_passed")
        else:
            self.state.survived = False
            self.state.paused = True
            world.trace.emit(
                world.clock_ms,
                "hazard_resolved",
                agent_id=AGENT_ID,
                result="hit_before_decision",
                protected_before_deadline=False,
            )

    def _finalize(self, world) -> None:
        if self.state.finalized:
            return
        self.state.finalized = True
        passed, violations = evaluate_envelope(
            self.case.envelope,
            interrupt_triggered=self.state.interrupt_triggered,
            mode=self.state.mode,
            action=self.state.chosen_action,
            sleep_completed=self.state.sleep_completed,
            survived=self.state.survived,
            social_engagement=self.state.social_engagement,
            brain_calls=self.state.brain_calls,
        )
        world.trace.emit(
            world.clock_ms,
            "bedroom_case_result",
            case=self.case.name,
            passed=passed,
            violations=list(violations),
            interrupt=self.state.interrupt_triggered,
            cognitive_mode=self.state.mode.value,
            brain_calls=self.state.brain_calls,
            action=None if self.state.chosen_action is None else self.state.chosen_action.name,
            action_roles=[] if self.state.chosen_action is None else list(self.state.chosen_action.roles),
            sleep_completed=self.state.sleep_completed,
            survived=self.state.survived,
            social_engagement=self.state.social_engagement,
        )

def build_scenarios() -> dict[str, BedroomScenario]:
    return {
        scenario.name: scenario
        for scenario in (BedroomScenario(case) for case in core_cases().values())
    }
