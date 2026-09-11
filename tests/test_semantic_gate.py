from awb.bedroom import CognitiveMode
from awb.semantic_decision import SemanticModelResponse
from awb.semantic_gate import (
    build_bedroom_gate_prompt,
    parse_gate_choice,
    run_semantic_gate_case,
)
from scenarios.bedroom_1 import core_cases


class StubGateBackend:
    def __init__(self, label: str) -> None:
        self.label = label
        self.calls = 0

    def choose_action(self, *, system_prompt: str, user_prompt: str, max_tokens: int) -> SemanticModelResponse:
        self.calls += 1
        return SemanticModelResponse(
            text=self.label,
            reasoning_text="",
            wall_ms=12.0,
            ttft_ms=9.0,
            prompt_tokens=10,
            completion_tokens=1,
            finish_reason="stop",
        )


def test_gate_prompt_hides_engineered_features():
    prompt = build_bedroom_gate_prompt(core_cases()["projectile"])
    for forbidden in (
        "goal_relevance",
        "prediction_error",
        "immediate_risk",
        "novelty",
        "uncertainty",
        "skill_validity",
        "action_ambiguity",
        "planning_horizon",
        "oracle",
    ):
        assert forbidden not in prompt
    assert "time_to_contact_ms" in prompt
    assert "fast object trajectory intersects agent body volume" in prompt


def test_parse_gate_choice_exact_labels():
    assert parse_gate_choice("CONTINUE").mode == CognitiveMode.AUTOMATIC
    assert parse_gate_choice("FAST").mode == CognitiveMode.FAST
    assert parse_gate_choice("EMERGENCY").mode == CognitiveMode.EMERGENCY
    assert parse_gate_choice("DELIBERATE").mode == CognitiveMode.DELIBERATE


def test_normal_case_does_not_call_semantic_gate_model():
    backend = StubGateBackend("FAST")
    result, prompt = run_semantic_gate_case(core_cases()["normal"], backend)
    assert prompt is None
    assert backend.calls == 0
    assert result.predicted_mode == "automatic"
    assert result.gate_pass is True


def test_roach_semantic_gate_probe():
    backend = StubGateBackend("FAST")
    result, _ = run_semantic_gate_case(core_cases()["roach"], backend)
    assert result.predicted_mode == "fast"
    assert result.predicted_interrupt is True
    assert result.gate_pass is True


def test_projectile_semantic_gate_probe():
    backend = StubGateBackend("EMERGENCY")
    result, _ = run_semantic_gate_case(core_cases()["projectile"], backend)
    assert result.predicted_mode == "emergency"
    assert result.gate_pass is True


def test_complex_visitor_semantic_gate_probe():
    backend = StubGateBackend("DELIBERATE")
    result, _ = run_semantic_gate_case(core_cases()["complex_visitor"], backend)
    assert result.predicted_mode == "deliberate"
    assert result.gate_pass is True
