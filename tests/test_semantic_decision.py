from __future__ import annotations

from awb.semantic_decision import (
    SemanticModelResponse,
    build_bedroom_semantic_prompt,
    parse_action_choice,
    run_semantic_case,
)
from scenarios.bedroom_1 import core_cases


class StubSemanticBackend:
    def __init__(self, text: str, *, wall_ms: float = 20.0, ttft_ms: float = 10.0):
        self.text = text
        self.wall_ms = wall_ms
        self.ttft_ms = ttft_ms
        self.calls = 0

    def choose_action(self, *, system_prompt: str, user_prompt: str, max_tokens: int):
        self.calls += 1
        return SemanticModelResponse(
            text=self.text,
            reasoning_text="",
            wall_ms=self.wall_ms,
            ttft_ms=self.ttft_ms,
            prompt_tokens=123,
            completion_tokens=3,
            finish_reason="stop",
        )


def test_prompt_does_not_expose_oracle_gate_features_or_case_name():
    case = core_cases()["projectile"]
    prompt = build_bedroom_semantic_prompt(case)
    assert "projectile" not in prompt.lower()
    for forbidden in (
        "goal_relevance",
        "prediction_error",
        "immediate_risk",
        "novelty",
        "uncertainty",
        "skill_validity",
        "action_ambiguity",
        "planning_horizon",
        "emergency",
    ):
        assert forbidden not in prompt
    assert "time_to_contact_ms" in prompt
    assert "protect_self" in prompt


def test_parser_accepts_exact_and_tiny_json_but_not_unknown_action():
    actions = core_cases()["roach"].observation.available_actions
    assert parse_action_choice("clear_sleep_surface", actions).action.name == "clear_sleep_surface"
    assert parse_action_choice('{"action":"inspect_change"}', actions).action.name == "inspect_change"
    assert parse_action_choice("teleport_away", actions).action is None


def test_real_semantic_path_uses_model_choice_not_reference_policy():
    # The reference policy would choose clear_sleep_surface. The stub model
    # deliberately chooses inspect_change, proving this path evaluates the
    # backend's semantic choice rather than silently falling back. Bedroom-1's
    # behavioral envelope now accepts inspection as a reasonable immediate next
    # action when the entity classification is not perfectly certain.
    backend = StubSemanticBackend("inspect_change")
    result, _ = run_semantic_case(core_cases()["roach"], backend)
    assert backend.calls == 1
    assert result.action == "inspect_change"
    assert result.semantic_pass is True
    assert result.realtime_pass is True


def test_projectile_correct_action_can_still_fail_real_time_deadline():
    backend = StubSemanticBackend("protect_self", wall_ms=140.0, ttft_ms=100.0)
    result, _ = run_semantic_case(core_cases()["projectile"], backend)
    assert result.semantic_pass is True
    assert result.decision_deadline_ms == 105
    assert result.deadline_miss is True
    assert result.survived is False
    assert result.realtime_pass is False


def test_projectile_correct_and_fast_action_survives():
    backend = StubSemanticBackend("protect_self", wall_ms=70.0, ttft_ms=30.0)
    result, _ = run_semantic_case(core_cases()["projectile"], backend)
    assert result.semantic_pass is True
    assert result.deadline_miss is False
    assert result.survived is True
    assert result.realtime_pass is True


def test_normal_control_never_calls_model():
    backend = StubSemanticBackend("anything")
    result, prompt = run_semantic_case(core_cases()["normal"], backend)
    assert backend.calls == 0
    assert prompt is None
    assert result.oracle_mode == "automatic"
    assert result.realtime_pass is True
