from __future__ import annotations

import random
from collections import defaultdict

from .cognition import (
    invalidate_target_dependent_state,
    remember_observation,
    set_belief as cognition_set_belief,
    set_intent,
    set_plan,
    state_fingerprint,
    structured_context_components,
)
from .events import EventQueue
from .scheduler import InferenceScheduler
from .tracing import TraceRecorder
from .types import (
    ActivationLevel,
    AgentState,
    InferenceRequest,
    Observation,
    RequestPriority,
    WorldEvent,
)


_COGNITIVE_PULSE = "__cognitive_pulse__"


class World:
    def __init__(
        self,
        *,
        seed: int,
        backend,
        trace: TraceRecorder,
        enable_cognitive_loop: bool = True,
    ) -> None:
        self.seed = seed
        self.rng = random.Random(seed)
        self.clock_ms = 0
        self.version = 0
        self.events = EventQueue()
        self.trace = trace
        self.agents: dict[int, AgentState] = {}
        self.location_members: dict[int, set[int]] = defaultdict(set)
        self.observations: list[Observation] = []
        self._request_sequence = 0
        self.enable_cognitive_loop = enable_cognitive_loop
        self.simulation_end_ms: int | None = None
        self.scheduler = InferenceScheduler(
            backend,
            trace,
            on_started=self._on_inference_started,
            on_completed=self._on_inference_completed,
            on_interrupted=self._on_inference_interrupted,
        )

    def set_simulation_end(self, end_ms: int) -> None:
        self.simulation_end_ms = int(end_ms)

    def add_agent(self, agent: AgentState) -> None:
        self.agents[agent.agent_id] = agent
        self.location_members[agent.location_id].add(agent.agent_id)
        self._schedule_next_cognitive_pulse(agent.agent_id, reason="initial_activation")

    def bump_version(self, reason: str) -> None:
        self.version += 1
        self.trace.emit(self.clock_ms, "world_version", version=self.version, reason=reason)

    def schedule(self, *args, **kwargs) -> None:
        self.events.push(*args, **kwargs)

    def set_activation(self, agent_id: int, level: ActivationLevel, reason: str) -> None:
        agent = self.agents[agent_id]
        if agent.activation != level:
            old = agent.activation
            agent.activation = level
            agent.last_update_ms = self.clock_ms
            self.trace.emit(
                self.clock_ms,
                "activation_changed",
                agent_id=agent_id,
                old=int(old),
                new=int(level),
                reason=reason,
            )
            self._schedule_next_cognitive_pulse(agent_id, reason=reason, force_earlier=True)

    def promote_activation(self, agent_id: int, level: ActivationLevel, reason: str) -> None:
        if self.agents[agent_id].activation < level:
            self.set_activation(agent_id, level, reason)

    def activation_snapshot(self) -> None:
        counts = {level: 0 for level in ActivationLevel}
        for agent in self.agents.values():
            if agent.alive:
                counts[agent.activation] += 1
        self.trace.emit(
            self.clock_ms,
            "activation_snapshot",
            dormant=counts[ActivationLevel.DORMANT],
            background=counts[ActivationLevel.BACKGROUND],
            active=counts[ActivationLevel.ACTIVE],
            cognitive=counts[ActivationLevel.COGNITIVE],
            interactive=counts[ActivationLevel.INTERACTIVE],
        )

    def set_belief(
        self,
        agent_id: int,
        key: str,
        value,
        confidence: float,
        *,
        source_event: str | None,
    ) -> None:
        agent = self.agents[agent_id]
        before, after = cognition_set_belief(
            agent,
            key=key,
            value=value,
            confidence=confidence,
            source_event=source_event,
            now_ms=self.clock_ms,
        )
        self.trace.emit(
            self.clock_ms,
            "belief_updated",
            agent_id=agent_id,
            key=key,
            old_confidence=None if before is None else before.confidence,
            new_confidence=after.confidence,
            revision=after.revision,
            source_event=source_event,
        )

    def observe_local_event(self, event: WorldEvent, confidence: float = 0.95) -> list[int]:
        if event.location_id is None:
            return []
        observers: list[int] = []
        for agent_id in sorted(self.location_members[event.location_id]):
            if agent_id == event.source_id:
                continue
            agent = self.agents[agent_id]
            if not agent.alive:
                continue
            if agent.activation >= ActivationLevel.ACTIVE or self.rng.random() < 0.58:
                obs = Observation(
                    observer_id=agent_id,
                    event_type=event.event_type,
                    source_id=event.source_id,
                    target_id=event.target_id,
                    confidence=confidence,
                    timestamp_ms=self.clock_ms,
                )
                self.observations.append(obs)
                remember_observation(
                    agent,
                    event_type=event.event_type,
                    source_id=event.source_id,
                    target_id=event.target_id,
                    confidence=confidence,
                    now_ms=self.clock_ms,
                )
                observers.append(agent_id)
                self.trace.emit(
                    self.clock_ms,
                    "observation",
                    observer_id=agent_id,
                    observed_event=event.event_type,
                    source_id=event.source_id,
                    target_id=event.target_id,
                    confidence=confidence,
                )
                self.trace.emit(
                    self.clock_ms,
                    "working_memory_updated",
                    agent_id=agent_id,
                    item_count=len(agent.cognition.working_memory.observations),
                    reason=event.event_type,
                )
        return observers

    def create_request(
        self,
        *,
        agent_id: int,
        task_type: str,
        priority: RequestPriority,
        deadline_ms: int,
        input_tokens: int,
        output_tokens: int,
        target_id: int | None = None,
        cancel_if_target_dead: bool = False,
        model_class: str = "small",
        cognitive_domain: str = "general",
        cognitive_layer: str = "cognitive",
        cache_reusable_tokens: int = 0,
    ) -> InferenceRequest:
        rid = f"r{self._request_sequence:05d}"
        self._request_sequence += 1
        agent = self.agents[agent_id]
        components = structured_context_components(agent, cognitive_domain)
        carried_state_tokens = min(input_tokens, sum(components.values()))
        continuation_of = agent.cognition.last_request_id_by_domain.get(cognitive_domain)
        fingerprint = state_fingerprint(agent, cognitive_domain)
        req = InferenceRequest(
            request_id=rid,
            agent_id=agent_id,
            task_type=task_type,
            priority=priority,
            created_at_ms=self.clock_ms,
            deadline_ms=deadline_ms,
            input_tokens=input_tokens,
            expected_output_tokens=output_tokens,
            world_version=self.version,
            target_id=target_id,
            cancel_if_target_dead=cancel_if_target_dead,
            model_class=model_class,
            cognitive_domain=cognitive_domain,
            cognitive_layer=cognitive_layer,
            carried_state_tokens=carried_state_tokens,
            cache_reusable_tokens=min(input_tokens, max(0, cache_reusable_tokens)),
            cognitive_revision=agent.cognition.revision,
            continuation_of=continuation_of,
            state_fingerprint=fingerprint,
        )
        self.trace.emit(
            self.clock_ms,
            "cognitive_context_built",
            agent_id=agent_id,
            request_id=rid,
            cognitive_domain=cognitive_domain,
            cognitive_layer=cognitive_layer,
            carried_state_tokens=carried_state_tokens,
            components=components,
            state_fingerprint=fingerprint,
        )
        agent.cognition.last_request_id_by_domain[cognitive_domain] = rid
        self.scheduler.enqueue(req)
        return req

    def kill_agent(self, agent_id: int, reason: str) -> None:
        agent = self.agents[agent_id]
        if not agent.alive:
            return
        agent.alive = False
        self.bump_version(reason)
        self.trace.emit(self.clock_ms, "agent_died", agent_id=agent_id, reason=reason)
        self.scheduler.cancel_invalid_target(agent_id, self.clock_ms)

    def _cadence_ms(self, level: ActivationLevel, agent_id: int) -> int | None:
        base = {
            ActivationLevel.ACTIVE: 2_500,
            ActivationLevel.COGNITIVE: 650,
            ActivationLevel.INTERACTIVE: 350,
        }.get(level)
        if base is None:
            return None
        return base + (agent_id % 5) * 17

    def _schedule_next_cognitive_pulse(
        self,
        agent_id: int,
        *,
        reason: str,
        force_earlier: bool = False,
    ) -> None:
        if not self.enable_cognitive_loop or self.simulation_end_ms is None:
            return
        agent = self.agents.get(agent_id)
        if agent is None or not agent.alive:
            return
        cadence = self._cadence_ms(agent.activation, agent_id)
        if cadence is None:
            return
        candidate = self.clock_ms + cadence
        if candidate > self.simulation_end_ms:
            return
        if agent.next_cognitive_pulse_ms is not None and agent.next_cognitive_pulse_ms <= candidate:
            return
        agent.next_cognitive_pulse_ms = candidate
        self.events.push(
            candidate,
            _COGNITIVE_PULSE,
            target_id=agent_id,
            location_id=agent.location_id,
            payload={"reason": reason, "forced": force_earlier},
        )

    def _handle_cognitive_pulse(self, event: WorldEvent) -> None:
        if event.target_id is None or event.target_id not in self.agents:
            return
        agent = self.agents[event.target_id]
        if agent.next_cognitive_pulse_ms != event.timestamp_ms:
            return
        agent.next_cognitive_pulse_ms = None
        if not agent.alive or agent.activation < ActivationLevel.ACTIVE:
            return

        danger_belief = agent.cognition.beliefs.get("armed_player_is_dangerous")
        if agent.activation == ActivationLevel.INTERACTIVE:
            priority = RequestPriority.HIGH
            deadline_ms = 550
            input_tokens = 800 + (agent.agent_id % 4) * 60
            output_tokens = 36
            model_class = "small"
            domain = "social"
        elif agent.activation == ActivationLevel.COGNITIVE:
            priority = RequestPriority.HIGH
            deadline_ms = 800
            input_tokens = 620 + (agent.agent_id % 5) * 55
            output_tokens = 24
            model_class = "tiny"
            domain = "threat" if danger_belief is not None else "ambient"
        else:
            priority = RequestPriority.NORMAL
            deadline_ms = 1_800
            input_tokens = 420 + (agent.agent_id % 5) * 40
            output_tokens = 16
            model_class = "tiny"
            domain = "ambient"

        self.trace.emit(
            self.clock_ms,
            "cognitive_pulse",
            agent_id=agent.agent_id,
            activation=int(agent.activation),
            cognitive_domain=domain,
        )
        self.create_request(
            agent_id=agent.agent_id,
            task_type="ambient_cognition",
            priority=priority,
            deadline_ms=deadline_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model_class=model_class,
            cognitive_domain=domain,
            cognitive_layer="cognitive",
        )
        self._schedule_next_cognitive_pulse(agent.agent_id, reason="continuous_cognition")

    def _emit_intent_transition(self, agent_id: int, before, after, reason: str) -> None:
        self.trace.emit(
            self.clock_ms,
            "intent_transition",
            agent_id=agent_id,
            before=None if before is None else before.name,
            before_status=None if before is None else before.status,
            after=after.name,
            after_status=after.status,
            target_id=after.target_id,
            reason=reason,
        )

    def _emit_plan_transition(self, agent_id: int, before, after, reason: str) -> None:
        self.trace.emit(
            self.clock_ms,
            "plan_transition",
            agent_id=agent_id,
            before=None if before is None else before.goal,
            before_status=None if before is None else before.status,
            after=after.goal,
            after_status=after.status,
            target_id=after.target_id,
            step_count=len(after.steps),
            reason=reason,
        )


    def _on_inference_started(self, req: InferenceRequest, now_ms: int) -> None:
        """Expose provisional cognition for tasks that begin forming a plan before completion."""
        agent = self.agents.get(req.agent_id)
        if agent is None or not agent.alive:
            return
        if req.task_type != "assist_target":
            return
        before_i, after_i = set_intent(
            agent,
            name="assist_injured_person",
            target_id=req.target_id,
            confidence=0.45,
            status="provisional",
            now_ms=now_ms,
            source_request_id=req.request_id,
        )
        self._emit_intent_transition(req.agent_id, before_i, after_i, "inference_started:assist_target")
        before_p, after_p = set_plan(
            agent,
            goal="assist_target",
            steps=("approach_target", "check_condition", "seek_help"),
            target_id=req.target_id,
            status="provisional",
            now_ms=now_ms,
            source_request_id=req.request_id,
        )
        self._emit_plan_transition(req.agent_id, before_p, after_p, "inference_started:assist_target")
        self.trace.emit(
            now_ms,
            "provisional_cognition_started",
            agent_id=req.agent_id,
            request_id=req.request_id,
            target_id=req.target_id,
            intent=after_i.name,
            plan=after_p.goal,
        )

    def _on_inference_completed(self, req: InferenceRequest, now_ms: int) -> None:
        agent = self.agents.get(req.agent_id)
        if agent is None or not agent.alive:
            return
        self.clock_ms = now_ms
        state = agent.cognition

        if req.task_type in {"threat_assessment", "threat_update", "combat_replan"}:
            before_i, after_i = set_intent(
                agent,
                name="seek_safety" if req.agent_id % 3 else "monitor_threat",
                target_id=None,
                confidence=0.82,
                status="active",
                now_ms=now_ms,
                source_request_id=req.request_id,
            )
            self._emit_intent_transition(req.agent_id, before_i, after_i, req.task_type)
            before_p, after_p = set_plan(
                agent,
                goal=after_i.name,
                steps=("locate_cover", "move_to_cover", "reassess"),
                target_id=None,
                status="active",
                now_ms=now_ms,
                source_request_id=req.request_id,
            )
            self._emit_plan_transition(req.agent_id, before_p, after_p, req.task_type)

        elif req.task_type == "assist_target":
            before_i, after_i = set_intent(
                agent,
                name="assist_injured_person",
                target_id=req.target_id,
                confidence=0.74,
                status="active",
                now_ms=now_ms,
                source_request_id=req.request_id,
            )
            self._emit_intent_transition(req.agent_id, before_i, after_i, req.task_type)
            before_p, after_p = set_plan(
                agent,
                goal="assist_target",
                steps=("approach_target", "check_condition", "seek_help"),
                target_id=req.target_id,
                status="active",
                now_ms=now_ms,
                source_request_id=req.request_id,
            )
            self._emit_plan_transition(req.agent_id, before_p, after_p, req.task_type)

        elif req.task_type == "dialogue":
            before_i, after_i = set_intent(
                agent,
                name="continue_conversation",
                target_id=None,
                confidence=0.70,
                status="active",
                now_ms=now_ms,
                source_request_id=req.request_id,
            )
            self._emit_intent_transition(req.agent_id, before_i, after_i, req.task_type)

        elif req.task_type == "memory_consolidation":
            # Consolidation changes the cognitive revision but deliberately keeps
            # raw working memory in v0.3; compaction policy comes later.
            self.trace.emit(
                now_ms,
                "memory_consolidated",
                agent_id=req.agent_id,
                request_id=req.request_id,
                observation_count=len(state.working_memory.observations),
            )

        state.revision += 1
        state.last_update_ms = now_ms
        state.last_reason = f"completed:{req.task_type}"
        self.trace.emit(
            now_ms,
            "cognitive_state_committed",
            agent_id=req.agent_id,
            request_id=req.request_id,
            cognitive_domain=req.cognitive_domain,
            revision=state.revision,
            belief_count=len(state.beliefs),
            working_memory_items=len(state.working_memory.observations),
            intent=None if state.intent is None else state.intent.name,
            intent_status=None if state.intent is None else state.intent.status,
            plan=None if state.plan is None else state.plan.goal,
            plan_status=None if state.plan is None else state.plan.status,
        )

    def _on_inference_interrupted(
        self,
        req: InferenceRequest,
        now_ms: int,
        processed_prefill_tokens: int,
        processed_decode_tokens: int,
        reason: str,
    ) -> tuple[list[str], list[str]]:
        agent = self.agents.get(req.agent_id)
        if agent is None or not agent.alive:
            return ([], [])

        preserved, invalidated = invalidate_target_dependent_state(
            agent, req.target_id if req.target_id is not None else -1, now_ms
        )
        state = agent.cognition
        state.revision += 1
        state.last_update_ms = now_ms
        state.last_reason = reason
        self.trace.emit(
            now_ms,
            "cognitive_state_revised",
            agent_id=req.agent_id,
            request_id=req.request_id,
            cognitive_domain=req.cognitive_domain,
            revision=state.revision,
            preserved_components=preserved,
            invalidated_components=invalidated,
            processed_prefill_tokens=processed_prefill_tokens,
            processed_decode_tokens=processed_decode_tokens,
            reason=reason,
        )
        return (preserved, invalidated)

    def _advance_scheduler_until(self, target_ms: int) -> None:
        while True:
            self.scheduler.start_ready(self.clock_ms)
            next_done = self.scheduler.next_completion_time()
            if next_done is None or next_done > target_ms:
                break
            self.clock_ms = next_done
            self.scheduler.complete_at(self.clock_ms)
        self.clock_ms = target_ms
        self.scheduler.complete_at(self.clock_ms)
        self.scheduler.start_ready(self.clock_ms)

    def run(self, handler) -> None:
        while self.events:
            event = self.events.pop()
            self._advance_scheduler_until(event.timestamp_ms)
            self.clock_ms = event.timestamp_ms

            if event.event_type == _COGNITIVE_PULSE:
                self._handle_cognitive_pulse(event)
                self.scheduler.start_ready(self.clock_ms)
                continue

            self.trace.emit(
                self.clock_ms,
                "world_event",
                event_type=event.event_type,
                source_id=event.source_id,
                target_id=event.target_id,
                location_id=event.location_id,
            )
            handler(self, event)
            self.scheduler.start_ready(self.clock_ms)
            self.activation_snapshot()

        while self.scheduler.has_pending():
            self.scheduler.start_ready(self.clock_ms)
            next_done = self.scheduler.next_completion_time()
            if next_done is None:
                break
            self.clock_ms = next_done
            self.scheduler.complete_at(self.clock_ms)
