from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Iterable, Mapping


class PreconditionStatus(StrEnum):
    SATISFIED = "satisfied"
    VIOLATED = "violated"
    UNKNOWN = "unknown"


@dataclass(slots=True, frozen=True)
class ObservedRelation:
    """Perception-bounded structural relation.

    The relation is descriptive world/agent state, not a cognition-mode label or
    a hand-authored risk score. Domain adapters may normalize engine-specific
    fields into this compact contract before meta/direct control.
    """

    subject: str
    predicate: str
    object: str
    confidence: float | None = None

    def prompt_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.object,
        }
        if self.confidence is not None:
            payload["confidence"] = self.confidence
        return payload


@dataclass(slots=True, frozen=True)
class SkillPreconditionSpec:
    """A domain-neutral requirement of the currently executing skill."""

    id: str
    description: str
    kind: str
    target: str


@dataclass(slots=True, frozen=True)
class SkillPreconditionState:
    """Runtime-derived status for one skill precondition."""

    id: str
    description: str
    kind: str
    target: str
    status: PreconditionStatus
    evidence: tuple[str, ...] = ()
    source: str = "affordance_resolver"

    def prompt_payload(self) -> dict[str, object]:
        return {
            "id": self.id,
            "description": self.description,
            "kind": self.kind,
            "target": self.target,
            "status": self.status.value,
            "evidence": list(self.evidence),
            "source": self.source,
        }


class GenericAffordanceResolver:
    """Resolve objective skill-precondition validity from structural relations.

    This resolver intentionally does not inspect scenario names, entity classes,
    cognition modes, or engineered risk/relevance scalars. It only applies
    generic relation semantics to explicit skill preconditions.
    """

    _VIOLATING_PREDICATES: Mapping[str, frozenset[str]] = {
        "resource_clear": frozenset({"occupies", "blocks"}),
        "trajectory_clear": frozenset({"trajectory_intersects"}),
    }

    def resolve(
        self,
        specs: Iterable[SkillPreconditionSpec],
        relations: Iterable[ObservedRelation],
        *,
        baseline: Mapping[str, PreconditionStatus] | None = None,
    ) -> tuple[SkillPreconditionState, ...]:
        relations = tuple(relations)
        baseline = baseline or {}
        states: list[SkillPreconditionState] = []

        for spec in specs:
            violating = self._VIOLATING_PREDICATES.get(spec.kind, frozenset())
            evidence = tuple(
                f"{rel.subject} {rel.predicate} {rel.object}"
                for rel in relations
                if rel.object == spec.target and rel.predicate in violating
            )
            if evidence:
                status = PreconditionStatus.VIOLATED
            else:
                status = baseline.get(spec.id, PreconditionStatus.UNKNOWN)
            states.append(
                SkillPreconditionState(
                    id=spec.id,
                    description=spec.description,
                    kind=spec.kind,
                    target=spec.target,
                    status=status,
                    evidence=evidence,
                )
            )

        return tuple(states)
