from __future__ import annotations

from typing import Protocol

from awb.types import InferenceRequest


class InferenceBackend(Protocol):
    max_concurrency: int

    def estimate_service_ms(self, request: InferenceRequest) -> int:
        ...
