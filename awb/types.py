from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any


class ActivationLevel(IntEnum):
    DORMANT = 0
    BACKGROUND = 1
    ACTIVE = 2
    COGNITIVE = 3
    INTERACTIVE = 4


class RequestPriority(IntEnum):
    BACKGROUND = 10
    NORMAL = 30
    HIGH = 60
    CRITICAL = 100


@dataclass(slots=True)
class BeliefState:
    key: str
    value: Any
    confidence: float
    source_event: str | None
    updated_at_ms: int
    revision: int = 1


@dataclass(slots=True)
class IntentState:
    name: str
    target_id: int | None
    confidence: float
    status: str
    updated_at_ms: int
    source_request_id: str | None = None


@dataclass(slots=True)
class PlanState:
    goal: str
    steps: tuple[str, ...]
    current_step: int
    target_id: int | None
    status: str
    updated_at_ms: int
    source_request_id: str | None = None


@dataclass(slots=True)
class WorkingMemoryItem:
    event_type: str
    source_id: int | None
    target_id: int | None
    confidence: float
    timestamp_ms: int


@dataclass(slots=True)
class WorkingMemory:
    observations: list[WorkingMemoryItem] = field(default_factory=list)
    max_observations: int = 12

    def remember(self, item: WorkingMemoryItem) -> None:
        self.observations.append(item)
        overflow = len(self.observations) - self.max_observations
        if overflow > 0:
            del self.observations[:overflow]


@dataclass(slots=True)
class CognitiveState:
    """Persistent structured cognition outside any individual model call."""

    revision: int = 0
    last_update_ms: int = 0
    last_reason: str | None = None
    beliefs: dict[str, BeliefState] = field(default_factory=dict)
    intent: IntentState | None = None
    plan: PlanState | None = None
    working_memory: WorkingMemory = field(default_factory=WorkingMemory)
    last_request_id_by_domain: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class AgentState:
    agent_id: int
    location_id: int
    activation: ActivationLevel = ActivationLevel.BACKGROUND
    fear: float = 0.0
    alive: bool = True
    current_goal: str | None = None
    last_update_ms: int = 0
    cognition: CognitiveState = field(default_factory=CognitiveState)
    next_cognitive_pulse_ms: int | None = None


@dataclass(order=True, slots=True)
class WorldEvent:
    timestamp_ms: int
    sequence: int
    event_type: str = field(compare=False)
    source_id: int | None = field(compare=False, default=None)
    target_id: int | None = field(compare=False, default=None)
    location_id: int | None = field(compare=False, default=None)
    payload: dict[str, Any] = field(compare=False, default_factory=dict)


@dataclass(slots=True)
class Observation:
    observer_id: int
    event_type: str
    source_id: int | None
    target_id: int | None
    confidence: float
    timestamp_ms: int


@dataclass(slots=True)
class InferenceRequest:
    request_id: str
    agent_id: int
    task_type: str
    priority: RequestPriority
    created_at_ms: int
    deadline_ms: int
    input_tokens: int
    expected_output_tokens: int
    world_version: int
    target_id: int | None = None
    cancel_if_target_dead: bool = False
    model_class: str = "small"
    cognitive_domain: str = "general"
    cognitive_layer: str = "cognitive"
    carried_state_tokens: int = 0
    cache_reusable_tokens: int = 0
    cognitive_revision: int = 0
    continuation_of: str | None = None
    state_fingerprint: str = ""

    @property
    def absolute_deadline_ms(self) -> int:
        return self.created_at_ms + self.deadline_ms

    @property
    def fresh_input_tokens(self) -> int:
        # Logical state persistence does NOT imply KV reuse. Only an explicit
        # runtime cache claim may reduce prefill work.
        return max(0, self.input_tokens - self.cache_reusable_tokens)


@dataclass(slots=True)
class InferenceResult:
    request_id: str
    started_at_ms: int
    finished_at_ms: int
    output_tokens: int
    interrupted: bool = False
    interruption_reason: str | None = None

    @property
    def service_time_ms(self) -> int:
        return self.finished_at_ms - self.started_at_ms
