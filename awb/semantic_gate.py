from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

from awb.bedroom import BedroomCase, CognitiveMode
from awb.semantic_decision import SemanticModelResponse


PROMPT_STYLE_VERSION = "bedroom_semantic_gate_v0.1"


class SemanticGateBackend(Protocol):
    def choose_action(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
    ) -> SemanticModelResponse:
        ...


@dataclass(slots=True, frozen=True)
class ParsedGateChoice:
    mode: CognitiveMode | None
    status: str
    matched_from: str | None = None


@dataclass(slots=True, frozen=True)
class SemanticGateCaseResult:
    case: str
    run_index: int
    prompt_style: str
    prompt_sha256: str | None
    engineered_gate_features_exposed: bool
    expected_modes: tuple[str, ...]
    predicted_mode: str | None
    predicted_interrupt: bool | None
    parse_status: str
    gate_pass: bool
    wall_ms: float
    ttft_ms: float
    prompt_tokens: int | None
    completion_tokens: int | None
    finish_reason: str | None
    model_text: str
    reasoning_text: str


SYSTEM_PROMPT = """You are a low-cost cognitive interrupt classifier for one game NPC.
Classify whether the NPC's currently executing skill can continue automatically or whether cognition should interrupt it.
Use only the current goal, current skill expectation, and perceived change. Do not invent hidden world facts.
Return exactly one label and nothing else:
CONTINUE - the current skill can proceed automatically without meaningful cognition.
FAST - a brief local decision is needed, but there is no hard immediate physical deadline.
EMERGENCY - an immediate physical threat or hard deadline requires an urgent decision.
DELIBERATE - a consequential, ambiguous, or long-horizon decision merits deeper cognition."""


SKILL_EXPECTATION = """sleep_at_bed normally continues automatically while the intended sleeping surface remains usable, the agent is not facing an immediate physical hazard, and no consequential unresolved interaction meaningfully conflicts with the current sleep goal."""


_LABEL_TO_MODE = {
    "CONTINUE": CognitiveMode.AUTOMATIC,
    "FAST": CognitiveMode.FAST,
    "EMERGENCY": CognitiveMode.EMERGENCY,
    "DELIBERATE": CognitiveMode.DELIBERATE,
}


def build_bedroom_gate_prompt(
    case: BedroomCase,
    *,
    current_phase: str = "sit_on_bed",
) -> str:
    """Build a semantic gate prompt without engineered oracle scalars.

    Deliberately omitted: case name, goal_relevance, prediction_error,
    immediate_risk, novelty, uncertainty, skill_validity, action_ambiguity,
    planning_horizon, oracle interrupt score, oracle mode, action roles, and
    the benchmark behavioral envelope.
    """

    observation = case.observation
    if observation is None:
        raise ValueError("The normal control has no observation delta and therefore no gate call")

    attrs = ""
    if observation.attributes:
        attrs = "\nOBSERVED ATTRIBUTES\n" + "\n".join(
            f"- {key}: {observation.attributes[key]}" for key in sorted(observation.attributes)
        )

    deadline = ""
    if observation.deadline_ms is not None:
        deadline = f"\nOBSERVED TIME TO CONTACT\n{int(observation.deadline_ms)} ms"

    return (
        "CURRENT GOAL\n"
        "sleep\n\n"
        "CURRENT SKILL\n"
        "sleep_at_bed\n\n"
        "CURRENT SKILL PHASE\n"
        f"{current_phase}\n\n"
        "CURRENT SKILL EXPECTATION\n"
        f"{SKILL_EXPECTATION}\n\n"
        "PERCEIVED CHANGE\n"
        f"{observation.summary}"
        f"{attrs}"
        f"{deadline}\n\n"
        "GATE LABEL"
    )


def parse_gate_choice(text: str, *, reasoning_text: str = "") -> ParsedGateChoice:
    cleaned = text.strip().strip("`").strip().upper()
    if cleaned in _LABEL_TO_MODE:
        return ParsedGateChoice(_LABEL_TO_MODE[cleaned], "exact", "content")

    def embedded(source: str, source_name: str) -> ParsedGateChoice | None:
        upper = source.upper()
        hits = [label for label in _LABEL_TO_MODE if label in upper]
        if len(hits) == 1:
            return ParsedGateChoice(_LABEL_TO_MODE[hits[0]], "embedded", source_name)
        if len(hits) > 1:
            return ParsedGateChoice(None, "ambiguous", source_name)
        return None

    matched = embedded(cleaned, "content")
    if matched is not None:
        return matched
    if reasoning_text:
        matched = embedded(reasoning_text, "reasoning")
        if matched is not None:
            return matched
    return ParsedGateChoice(None, "no_valid_label", None)


def run_semantic_gate_case(
    case: BedroomCase,
    backend: SemanticGateBackend,
    *,
    run_index: int = 1,
    max_tokens: int = 8,
) -> tuple[SemanticGateCaseResult, str | None]:
    expected_modes = tuple(mode.value for mode in case.envelope.acceptable_modes)

    # Event-driven boundary: with no observation delta, the gate is not called.
    if case.observation is None:
        predicted = CognitiveMode.AUTOMATIC
        result = SemanticGateCaseResult(
            case=case.name,
            run_index=run_index,
            prompt_style=PROMPT_STYLE_VERSION,
            prompt_sha256=None,
            engineered_gate_features_exposed=False,
            expected_modes=expected_modes,
            predicted_mode=predicted.value,
            predicted_interrupt=False,
            parse_status="no_delta_no_gate_call",
            gate_pass=predicted.value in expected_modes,
            wall_ms=0.0,
            ttft_ms=0.0,
            prompt_tokens=0,
            completion_tokens=0,
            finish_reason="control",
            model_text="",
            reasoning_text="",
        )
        return result, None

    prompt = build_bedroom_gate_prompt(case)
    prompt_sha = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    response = backend.choose_action(
        system_prompt=SYSTEM_PROMPT,
        user_prompt=prompt,
        max_tokens=max_tokens,
    )
    parsed = parse_gate_choice(response.text, reasoning_text=response.reasoning_text)
    predicted = parsed.mode
    gate_pass = predicted is not None and predicted.value in expected_modes

    result = SemanticGateCaseResult(
        case=case.name,
        run_index=run_index,
        prompt_style=PROMPT_STYLE_VERSION,
        prompt_sha256=prompt_sha,
        engineered_gate_features_exposed=False,
        expected_modes=expected_modes,
        predicted_mode=None if predicted is None else predicted.value,
        predicted_interrupt=None if predicted is None else predicted != CognitiveMode.AUTOMATIC,
        parse_status=parsed.status,
        gate_pass=gate_pass,
        wall_ms=round(response.wall_ms, 3),
        ttft_ms=round(response.ttft_ms, 3),
        prompt_tokens=response.prompt_tokens,
        completion_tokens=response.completion_tokens,
        finish_reason=response.finish_reason,
        model_text=response.text,
        reasoning_text=response.reasoning_text,
    )
    return result, prompt


def write_semantic_gate_report(
    results: list[SemanticGateCaseResult],
    prompts: dict[str, str],
    *,
    out_dir: Path,
    model: str,
    backend_name: str,
    health: dict | None = None,
) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "bedroom_1.semantic_gate_real.json"
    csv_path = out_dir / "bedroom_1.semantic_gate_real.csv"

    payload = {
        "prompt_style": PROMPT_STYLE_VERSION,
        "model": model,
        "backend": backend_name,
        "health": health,
        "experimental_scope": {
            "real_model_gate_classification": True,
            "real_hardware_latency": True,
            "engineered_gate_features_exposed": False,
            "oracle_gate_used_for_prediction": False,
            "action_selection_measured": False,
            "final_low_cost_gate_claimed": False,
            "note": "The real LLM is an experimental semantic-gate probe, not the intended final cheap gate/action-head implementation.",
        },
        "prompts": prompts,
        "results": [asdict(result) for result in results],
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    fieldnames = list(asdict(results[0]).keys()) if results else []
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            row = asdict(result)
            row["expected_modes"] = "|".join(result.expected_modes)
            writer.writerow(row)

    return json_path, csv_path
