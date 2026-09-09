# Bedroom-1 Hardware Matrix v0.1

Bedroom-1 is designed to be replayed on both low-end CPU and representative
consumer gaming GPUs.

## Canonical trace generation

```bash
python run.py bedroom-suite --seed 7 --concurrency 1 --out traces/bedroom_1
```

The core traces intentionally contain different cognition shapes:

- `bedroom_1_normal`: zero model calls;
- `bedroom_1_roach`: one short `fast` decision;
- `bedroom_1_projectile`: one hard-deadline `emergency` decision;
- `bedroom_1_complex_visitor`: `fast` triage followed by a larger `deliberate` request.

## Primary consumer matrix

The first comparative matrix should use the same model/runtime configuration on:

1. i5-8250U CPU-only (existing floor / plumbing reference)
2. RTX 3090 24 GB
3. RTX 4090 24 GB
4. RTX 5090 32 GB

Datacenter GPUs can be added later as architecture references, not as the primary
claim about gaming PCs.

## Open-loop replay

Start the same llama.cpp-compatible model server on each machine, then replay the
same canonical trace. Example:

```bash
python run.py replay-real traces/bedroom_1/bedroom_1_projectile.trace.jsonl \
  --base-url http://127.0.0.1:8080 \
  --concurrency 1 \
  --timing-mode immediate \
  --cognitive-layers emergency
```

Repeat for `fast` and `deliberate` layers. Record at least:

- TTFT;
- prompt/prefill time;
- decode time;
- end-to-end wall latency;
- deadline miss;
- prompt/decode tokens per second;
- runtime cache tokens when actually reported;
- GPU/CPU model and memory capacity;
- power/energy later when a reproducible telemetry path is available.

## Closed-loop follow-up

Open-loop replay isolates runtime/hardware. It does **not** yet prove gameplay
success. The next implementation stage will feed measured decision latency back
into Bedroom-1's world clock so, for example, an emergency decision completed
after the 120 ms projectile deadline fails even if its semantic action is good.

This separation is deliberate: first measure the hardware truth, then connect it
to world outcome without inventing synthetic latency.
