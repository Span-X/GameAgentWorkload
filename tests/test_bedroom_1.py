from __future__ import annotations

from awb.backends import FakeBackend
from awb.bedroom import (
    ActionOption,
    BehavioralEnvelope,
    BedroomCase,
    CognitiveBudgetController,
    CognitiveInterruptGate,
    CognitiveMode,
    ObservationDelta,
)
from awb.metrics import build_report
from awb.tracing import TraceRecorder
from awb.world import World
from scenarios.bedroom_1 import BedroomScenario, core_cases


def run_case(case: BedroomCase, *, concurrency: int = 1):
    scenario = BedroomScenario(case)
    trace = TraceRecorder()
    world = World(seed=7, backend=FakeBackend(max_concurrency=concurrency), trace=trace)
    scenario.build(world)
    world.activation_snapshot()
    world.run(scenario.handle)
    report = build_report(scenario.name, len(world.agents), trace.records, trace.request_stream_digest())
    result = next(r for r in trace.records if r.get("event") == "bedroom_case_result")
    return trace.records, report, result


def test_normal_sleep_uses_no_expensive_cognition():
    _, report, result = run_case(core_cases()["normal"])
    assert report.requests == 0
    assert result["passed"] is True
    assert result["cognitive_mode"] == "automatic"
    assert result["sleep_completed"] is True


def test_roach_interrupts_once_and_resumes_sleep():
    records, report, result = run_case(core_cases()["roach"])
    assert report.requests == 1
    assert result["passed"] is True
    assert result["cognitive_mode"] == "fast"
    assert "restore_precondition" in result["action_roles"]
    assert result["sleep_completed"] is True
    assert any(r.get("event") == "skill_resumed" for r in records)


def test_projectile_is_hard_deadline_emergency_and_survives_fake_fast_backend():
    records, report, result = run_case(core_cases()["projectile"])
    assert report.requests == 1
    assert result["passed"] is True
    assert result["cognitive_mode"] == "emergency"
    assert "protective" in result["action_roles"]
    assert result["survived"] is True
    req = next(r for r in records if r.get("event") == "inference_request")
    assert req["deadline_ms"] <= 120


def test_complex_visitor_progressively_deepens_cognition():
    records, report, result = run_case(core_cases()["complex_visitor"])
    assert report.requests == 2
    assert result["passed"] is True
    assert result["cognitive_mode"] == "deliberate"
    assert result["social_engagement"] is True
    tasks = [r["task_type"] for r in records if r.get("event") == "inference_request"]
    assert tasks == ["bedroom_fast_triage", "bedroom_deliberation"]


def test_core_gate_does_not_need_case_name_for_unseen_broken_bed_case():
    # Extension case intentionally defined only in the test. The generic gate
    # and budget controller are unchanged.
    alt = ActionOption(
        "seek_alternate_sleep_surface",
        ("restore_precondition",),
        "Find another valid sleep surface.",
    )
    broken_bed = BedroomCase(
        name="extension_broken_bed",
        description="The target bed becomes unusable before lie-down.",
        observation=ObservationDelta(
            delta_id="target_validity_02",
            summary="intended sleep surface is no longer structurally usable",
            goal_relevance=0.98,
            prediction_error=0.88,
            immediate_risk=0.08,
            novelty=0.15,
            uncertainty=0.20,
            skill_validity=0.00,
            action_ambiguity=0.22,
            planning_horizon=0.25,
            available_actions=(alt,),
            attributes={"target_usable": False},
        ),
        envelope=BehavioralEnvelope(
            expected_interrupt=True,
            acceptable_modes=(CognitiveMode.FAST,),
            required_action_roles=("restore_precondition",),
            max_brain_calls=1,
        ),
    )
    _, report, result = run_case(broken_bed)
    assert report.requests == 1
    assert result["passed"] is True
    assert result["action"] == "seek_alternate_sleep_surface"


def test_harmless_goal_irrelevant_change_does_not_overtrigger():
    gate = CognitiveInterruptGate()
    budget = CognitiveBudgetController()
    noop = ActionOption("continue", ("continue_skill",), "Continue current skill")
    thunder = ObservationDelta(
        delta_id="ambient_01",
        summary="loud but recognized distant thunder outside",
        goal_relevance=0.12,
        prediction_error=0.60,
        immediate_risk=0.06,
        novelty=0.08,
        uncertainty=0.08,
        skill_validity=0.98,
        action_ambiguity=0.05,
        planning_horizon=0.02,
        available_actions=(noop,),
    )
    assessment = gate.assess(thunder)
    assigned = budget.assign(thunder, assessment)
    assert assessment.interrupt is False
    assert assigned.mode == CognitiveMode.AUTOMATIC
    assert assigned.stages == ()
