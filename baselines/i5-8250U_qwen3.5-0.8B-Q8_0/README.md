# First real hardware baseline — i5-8250U

This directory preserves the first non-simulated AWB replay supplied from the target Windows machine.

## Observed baseline

- CPU: Intel Core i5-8250U
- Model: Qwen3.5-0.8B-Q8_0 GGUF
- Backend: llama.cpp, CPU-only (`--n-gpu-layers 0`)
- Context: 2048
- Replay: 8 `threat_assessment` requests, concurrency 1, immediate arrival
- Prompt throughput mean: ~73 tok/s
- Decode throughput mean: ~19.2 tok/s
- TTFT p50: ~12.1 s
- Request wall p50: ~13.7 s
- Original deadline: 220 ms
- Deadline miss: 100%

## Important interpretation

This result is not primarily a queue problem. Every sampled request has a server wall time far above the 220 ms deadline. Even with zero client queue, the original cognitive request cannot meet its deadline on this CPU/model configuration.

The baseline motivated AWB-0.4's split between:

1. **Reflex** — non-LLM / deterministic or micro-policy fast path.
2. **Reactive** — very short model request, tiny structured output.
3. **Cognitive** — slower high-level reasoning that must not block immediate action.
4. **Strategic** — background / long-horizon cognition.

The original raw report is retained rather than rewritten so the first real result remains auditable.
