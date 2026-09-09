from __future__ import annotations

from dataclasses import dataclass

from awb.types import InferenceRequest


@dataclass(slots=True, frozen=True)
class ServiceEstimate:
    overhead_ms: float
    prefill_ms: float
    decode_ms: float
    total_ms: int


@dataclass(slots=True, frozen=True)
class ProgressEstimate:
    processed_prefill_tokens: int
    processed_decode_tokens: int
    stage: str


@dataclass(slots=True)
class FakeBackend:
    """Deterministic latency model for workload development.

    Structured cognitive state does not magically reduce prefill. Only the
    explicit `cache_reusable_tokens` field can do that, which is zero by
    default in v0.3 until a real runtime/cache experiment justifies otherwise.
    """

    prefill_tokens_per_second: float = 30_000.0
    decode_tokens_per_second: float = 320.0
    fixed_overhead_ms: float = 4.0
    max_concurrency: int = 4

    def estimate(self, request: InferenceRequest) -> ServiceEstimate:
        model_factor = {"tiny": 0.65, "small": 1.0, "medium": 1.6}.get(
            request.model_class, 1.0
        )
        overhead_ms = self.fixed_overhead_ms * model_factor
        prefill_ms = (
            request.fresh_input_tokens / self.prefill_tokens_per_second * 1000.0
        ) * model_factor
        decode_ms = (
            request.expected_output_tokens / self.decode_tokens_per_second * 1000.0
        ) * model_factor
        total_ms = max(1, round(overhead_ms + prefill_ms + decode_ms))
        return ServiceEstimate(overhead_ms, prefill_ms, decode_ms, total_ms)

    def estimate_service_ms(self, request: InferenceRequest) -> int:
        return self.estimate(request).total_ms

    def estimate_progress(self, request: InferenceRequest, elapsed_ms: int) -> ProgressEstimate:
        est = self.estimate(request)
        remaining = max(0.0, float(elapsed_ms) - est.overhead_ms)

        if remaining <= 0:
            return ProgressEstimate(0, 0, "overhead")

        if est.prefill_ms > 0 and remaining < est.prefill_ms:
            frac = remaining / est.prefill_ms
            processed_prefill = min(
                request.fresh_input_tokens,
                max(0, round(request.fresh_input_tokens * frac)),
            )
            return ProgressEstimate(processed_prefill, 0, "prefill")

        processed_prefill = request.fresh_input_tokens
        remaining -= est.prefill_ms
        if remaining <= 0 or est.decode_ms <= 0:
            return ProgressEstimate(processed_prefill, 0, "prefill")

        frac = min(1.0, remaining / est.decode_ms)
        processed_decode = min(
            request.expected_output_tokens,
            max(0, round(request.expected_output_tokens * frac)),
        )
        stage = "decode" if processed_decode < request.expected_output_tokens else "complete"
        return ProgressEstimate(processed_prefill, processed_decode, stage)
