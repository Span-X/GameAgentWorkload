# AWB-0.4 Design Notes — Layered Cognition + Deadline Floors

## Why v0.4 exists

The first real hardware replay on an Intel Core i5-8250U with Qwen3.5-0.8B-Q8_0 exposed a structural issue in the original workload: a 220 ms threat deadline was attached to a full 760-1040 token LLM request producing 24-36 tokens. On that CPU, prompt prefill alone took roughly 8.8-15.1 seconds and decoding roughly 1.1-2.2 seconds. Queue tuning cannot fix a request whose service time already exceeds the deadline.

AWB-0.4 therefore separates immediate action from deeper cognition and makes deadline causes observable.

## 1. Layered cognition

`saloon_64` is retained unchanged as the historical workload.

A new scenario, `saloon_64_layered`, introduces four conceptual layers:

1. **Reflex** — deterministic or micro-policy action. No LLM request is generated. Examples: orient to a gun draw, duck after a gunshot.
2. **Reactive** — tiny structured model request, usually 64-128 input tokens and 1-2 output tokens. Used for fast gating/status updates.
3. **Cognitive** — slower high-level reasoning such as threat assessment, combat replanning, conversation, and target assistance.
4. **Strategic** — background long-horizon work such as memory consolidation.

The key rule is that an immediate safety action must not wait for a complete high-level language-model response.

## 2. Real replay deadline decomposition

Every real replay request now reports:

- `deadline_miss`: queue + service exceeded deadline.
- `zero_queue_deadline_miss`: service time alone exceeded deadline.
- `queue_only_deadline_miss`: service could have met the deadline but client queue caused the miss.
- `ttft_deadline_miss`: even the first generated token arrived after the deadline.
- `deadline_slack_ms`: deadline minus end-to-end decision time.
- `zero_queue_slack_ms`: deadline minus service time.

The aggregate `zero_queue_deadline_floor` is especially important: it estimates the minimum deadline-miss rate achievable by adding concurrency alone, assuming identical per-request service latency.

## 3. Cognitive layer is part of the trace contract

`InferenceRequest` and canonical traces now carry `cognitive_layer`.

Real replay can filter by layer with:

```powershell
python run.py replay-real traces\saloon_64_layered.trace.jsonl `
  --base-url http://127.0.0.1:8080 `
  --cognitive-layers reactive `
  --concurrency 1 `
  --max-requests 8 `
  --timing-mode immediate
```

This lets one hardware target be tested separately for fast reactive work and slower cognitive work.

## 4. First real baseline is preserved

`baselines/i5-8250U_qwen3.5-0.8B-Q8_0/` contains the original raw v0.3 report/CSV plus a v0.4-derived deadline analysis. The raw files are not rewritten.

Known conclusion from that baseline:

- total deadline miss: 100%
- zero-queue deadline floor: 100%
- queue-only miss: 0%
- TTFT later than deadline: 100%

Therefore the original 220 ms full-LLM threat request is intrinsically incompatible with this CPU/model configuration.

## 5. What v0.4 still does not claim

- Reflex latency is a workload contract, not measured hardware latency yet.
- Reactive micro-requests still use the same llama.cpp model during real replay unless a different model/backend is supplied.
- AWB does not yet route layers to different models automatically.
- No CPU power, RAM bandwidth, temperature, or energy-per-decision telemetry is captured yet.
- No GPU/NPU benchmark has been run yet.

## Next engineering target

AWB-0.5 should add **model routing + hardware telemetry**:

- layer -> model/runtime mapping (`reactive` vs `cognitive`)
- Windows/Linux CPU/RAM/GPU telemetry collection
- real cache/prefix experiments
- standardized hardware metadata capture
- automated hardware-matrix result aggregation
