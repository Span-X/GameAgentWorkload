from __future__ import annotations

from dataclasses import dataclass

from awb.backends.llamacpp import LlamaCppBackend
from awb.semantic_decision import SemanticModelResponse


@dataclass(slots=True)
class LlamaCppSemanticDecisionBackend:
    """Semantic action backend backed by a real llama.cpp chat completion."""

    client: LlamaCppBackend

    def choose_action(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
    ) -> SemanticModelResponse:
        result = self.client.stream_chat_completion(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=max_tokens,
        )
        return SemanticModelResponse(
            text=str(result.get("text", "")),
            reasoning_text=str(result.get("reasoning_text", "")),
            wall_ms=float(result.get("wall_ms", 0.0)),
            ttft_ms=float(result.get("ttft_ms", 0.0)),
            prompt_tokens=result.get("prompt_tokens"),
            completion_tokens=result.get("completion_tokens"),
            finish_reason=result.get("finish_reason"),
        )
