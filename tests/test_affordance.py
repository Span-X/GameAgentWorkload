from __future__ import annotations

from awb.affordance import (
    GenericAffordanceResolver,
    ObservedRelation,
    PreconditionStatus,
    SkillPreconditionSpec,
)


def test_resource_clear_resolver_is_entity_class_agnostic():
    spec = SkillPreconditionSpec(
        id="target_clear",
        description="Target resource must be clear.",
        kind="resource_clear",
        target="target_surface",
    )
    resolver = GenericAffordanceResolver()

    for subject in ("unknown_object", "animal", "tool", "debris"):
        state = resolver.resolve(
            (spec,),
            (ObservedRelation(subject, "occupies", "target_surface"),),
            baseline={"target_clear": PreconditionStatus.SATISFIED},
        )[0]
        assert state.status == PreconditionStatus.VIOLATED
        assert subject in state.evidence[0]


def test_unrelated_relation_does_not_invalidate_resource_clear():
    spec = SkillPreconditionSpec(
        id="target_clear",
        description="Target resource must be clear.",
        kind="resource_clear",
        target="target_surface",
    )
    state = GenericAffordanceResolver().resolve(
        (spec,),
        (ObservedRelation("object", "occupies", "other_surface"),),
        baseline={"target_clear": PreconditionStatus.SATISFIED},
    )[0]
    assert state.status == PreconditionStatus.SATISFIED


def test_trajectory_clear_resolver_uses_structural_relation_not_risk_score():
    spec = SkillPreconditionSpec(
        id="collision_clear",
        description="No trajectory intersects the body.",
        kind="trajectory_clear",
        target="agent_body",
    )
    state = GenericAffordanceResolver().resolve(
        (spec,),
        (ObservedRelation("moving_object", "trajectory_intersects", "agent_body", 0.97),),
        baseline={"collision_clear": PreconditionStatus.SATISFIED},
    )[0]
    assert state.status == PreconditionStatus.VIOLATED


def test_missing_baseline_remains_unknown_for_exception_projection_contract():
    spec = SkillPreconditionSpec(
        id="visibility_known",
        description="Visibility must be known.",
        kind="visibility_known",
        target="target",
    )
    state = GenericAffordanceResolver().resolve((spec,), (), baseline={})[0]
    assert state.status == PreconditionStatus.UNKNOWN
