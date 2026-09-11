from __future__ import annotations

import json

from awb.control_state import bedroom_control_state
from awb.direct_control import (
    ESCALATE_TOKEN,
    build_direct_control_prompt,
    parse_direct_decision,
    run_direct_control_case,
)
from awb.semantic_decision import SemanticModelResponse
from scenarios.bedroom_1 import core_cases


class StubDirectBackend:
    def __init__(self, text: str, *, wall_ms: float = 20.0, ttft_ms: float = 10.0, reasoning: str = ""):
        self.text = text
        self.wall_ms = wall_ms
        self.ttft_ms = ttft_ms
        self.reasoning = reasoning
        self.calls = 0

    def choose_action(self, *, system_prompt: str, user_prompt: str, max_tokens: int):
        self.calls += 1
        return SemanticModelResponse(
            text=self.text,
            reasoning_text=self.reasoning,
            wall_ms=self.wall_ms,
            ttft_ms=self.ttft_ms,
            prompt_tokens=100,
            completion_tokens=2,
            finish_reason="stop",
        )


def test_bedroom_adapter_excludes_engineered_gate_features_from_payload():
    state = bedroom_control_state(core_cases()["projectile"])
    payload = state.prompt_payload()
    rendered = json.dumps(payload)
    for forbidden in (
        "goal_relevance",
        "prediction_error",
        "immediate_risk",
        "novelty",
        "uncertainty",
        "skill_validity",
        "action_ambiguity",
        "planning_horizon",
        "oracle_mode",
    ):
        assert forbidden not in rendered
    assert payload["hard_deadline_ms"] == 120
    assert payload["observed_attributes"]["trajectory_relation"] == "intersects_self"


def test_direct_prompt_is_structured_and_closed_over_actions_plus_escalate():
    state = bedroom_control_state(core_cases()["roach"])
    prompt = build_direct_control_prompt(state)
    payload = json.loads(prompt)
    allowed = payload["controller_output_contract"]["allowed"]
    assert "clear_sleep_surface" in allowed
    assert ESCALATE_TOKEN in allowed
    assert "immediate_risk" not in prompt


def test_normal_always_calls_small_model_and_can_continue():
    backend = StubDirectBackend("continue_sleep_skill")
    result, prompt = run_direct_control_case(core_cases()["normal"], backend)
    assert backend.calls == 1
    assert prompt
    assert result.oracle_gate_used is False
    assert result.action == "continue_sleep_skill"
    assert result.local_resolution_pass is True
    assert result.realtime_pass is True


def test_projectile_direct_action_can_meet_hard_deadline_without_oracle_budget():
    backend = StubDirectBackend("protect_self", wall_ms=24.0, ttft_ms=15.0)
    result, _ = run_direct_control_case(core_cases()["projectile"], backend)
    assert result.oracle_gate_used is False
    assert result.decision_deadline_ms == 105
    assert result.deadline_miss is False
    assert result.local_resolution_pass is True
    assert result.survived is True
    assert result.realtime_pass is True


def test_projectile_escalate_is_valid_route_but_not_survival():
    backend = StubDirectBackend("ESCALATE", wall_ms=20.0, ttft_ms=10.0)
    result, _ = run_direct_control_case(core_cases()["projectile"], backend)
    assert result.decision_valid is True
    assert result.escalated is True
    assert result.local_resolution_pass is False
    assert result.survived is False
    assert result.realtime_pass is False


def test_direct_parser_does_not_mine_reasoning_for_an_action():
    actions = bedroom_control_state(core_cases()["projectile"]).available_actions
    parsed = parse_direct_decision("", actions)
    assert parsed.kind == "invalid"
    backend = StubDirectBackend("", reasoning="I would choose protect_self")
    result, _ = run_direct_control_case(core_cases()["projectile"], backend)
    assert result.action is None
    assert result.parse_status == "no_valid_direct_decision"


def test_complex_visitor_can_resolve_locally_or_escalate_without_mode_label():
    local = StubDirectBackend("defer_interaction")
    local_result, _ = run_direct_control_case(core_cases()["complex_visitor"], local)
    assert local_result.local_resolution_pass is True
    assert local_result.decision_kind == "action"

    escalator = StubDirectBackend("ESCALATE")
    escalated, _ = run_direct_control_case(core_cases()["complex_visitor"], escalator)
    assert escalated.decision_valid is True
    assert escalated.escalated is True
    assert escalated.local_resolution_pass is False


def test_roach_control_state_keeps_full_runtime_state_but_projects_only_exception():
    state = bedroom_control_state(core_cases()["roach"])
    by_id = {item.id: item for item in state.skill_preconditions}
    assert by_id["sleep_surface_clear"].status.value == "violated"
    assert by_id["body_trajectory_clear"].status.value == "satisfied"

    payload = state.prompt_payload()
    exceptions = payload["precondition_exceptions"]
    assert [item["id"] for item in exceptions] == ["sleep_surface_clear"]
    assert exceptions[0]["status"] == "violated"
    rendered = json.dumps(payload)
    assert "body_trajectory_clear" not in rendered
    assert "goal_relevance" not in rendered
    assert "immediate_risk" not in rendered


def test_projectile_control_state_exposes_only_collision_exception_to_model():
    state = bedroom_control_state(core_cases()["projectile"])
    by_id = {item.id: item for item in state.skill_preconditions}
    assert by_id["body_trajectory_clear"].status.value == "violated"
    assert by_id["sleep_surface_clear"].status.value == "satisfied"
    payload = state.prompt_payload()
    assert [item["id"] for item in payload["precondition_exceptions"]] == [
        "body_trajectory_clear"
    ]


def test_roach_inspection_is_a_valid_immediate_next_action_not_only_clear():
    backend = StubDirectBackend("inspect_change")
    result, _ = run_direct_control_case(core_cases()["roach"], backend)
    assert result.local_resolution_pass is True
    assert "inspect" in result.action_roles


def test_roach_continuation_still_fails_when_surface_precondition_is_violated():
    backend = StubDirectBackend("continue_sleep_skill")
    result, _ = run_direct_control_case(core_cases()["roach"], backend)
    assert result.local_resolution_pass is False
    assert result.violated_preconditions == ("sleep_surface_clear",)


def test_nominal_and_social_cases_do_not_dump_satisfied_preconditions():
    for case_name in ("normal", "complex_visitor"):
        state = bedroom_control_state(core_cases()[case_name])
        assert all(item.status.value == "satisfied" for item in state.skill_preconditions)
        payload = state.prompt_payload()
        assert payload["precondition_exceptions"] == []
        rendered = json.dumps(payload)
        assert "sleep_surface_clear" not in rendered
        assert "body_trajectory_clear" not in rendered


def test_complex_visitor_keeps_decision_relevant_interaction_after_exception_projection():
    payload = bedroom_control_state(core_cases()["complex_visitor"]).prompt_payload()
    assert payload["precondition_exceptions"] == []
    assert payload["perceived_change"] == "trusted person initiates consequential ambiguous interaction"
    assert payload["observed_attributes"]["stakes"] == "high"
    assert any(
        relation["predicate"] == "initiates_interaction"
        for relation in payload["observed_relations"]
    )
