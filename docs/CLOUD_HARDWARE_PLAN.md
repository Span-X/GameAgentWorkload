# AWB Cloud Hardware Test Plan

Snapshot date: 2026-09-08. Prices and availability change; verify before renting.

AWB should separate two purposes:

1. **cheap exploratory runs** — find bottlenecks and sanity-check many GPU classes;
2. **reproducible benchmark runs** — stable host quality, known GPU model, controlled software stack.

## Recommended order

### 1. Local i5-8250U

Purpose: CPU floor / portability baseline.

Start with a 0.8B-class GGUF, concurrency 1, and short-context task subsets. Do not attempt the full 273-request trace first.

### 2. Runpod — first GPU matrix

Best first choice for AWB because consumer and datacenter NVIDIA GPUs are both commonly available, including RTX 3090 / 4090 / 5090 and A100 / H100-class devices.

Suggested first matrix:

- RTX 3090 24GB — older consumer baseline
- RTX 4090 24GB — strong Ada consumer baseline
- RTX 5090 32GB — Blackwell consumer baseline
- A100 80GB — datacenter memory-bandwidth baseline
- H100 80GB — modern datacenter baseline

Use a persistent Pod rather than serverless for latency benchmarks so cold starts do not contaminate the workload.

### 3. Vast.ai — low-cost exploratory / unusual inventory

Useful when the goal is inexpensive access to a broad marketplace. Host CPU, PCIe topology, storage and networking vary between listings, so record the full machine metadata and avoid mixing different hosts into a single published comparison.

### 4. Lambda — controlled datacenter reference

Useful for A100/H100/B200-class reference points when a more standardized cloud environment matters more than minimum price. Better suited to reproducible datacenter comparisons than consumer-GPU coverage.

### 5. Modal — automation / CI-style hardware smoke

Useful for scripting repeated GPU jobs with per-second billing and little idle cost. It is less suitable as the primary close-to-metal gaming-GPU benchmark because the available fleet is mostly datacenter GPUs and serverless/container lifecycle can add platform effects.

## Benchmark discipline

For every rented machine record:

- provider
- region
- exact GPU name
- GPU count
- VRAM
- driver version
- CUDA version
- llama.cpp commit/version
- llama.cpp launch command
- CPU model
- RAM
- PCIe topology when available
- model GGUF filename + hash
- context size
- server parallel slots (`-np`)
- flash-attention setting
- AWB trace digest
- AWB command

Do not compare two providers solely by GPU name if host CPU, power limits or virtualization differ.

## When to leave the local i5

Move to cloud when one of these becomes true:

- even an 0.8B-class model makes an 8-request smoke impractically slow;
- a test needs CUDA/VRAM telemetry;
- concurrency >1 is required with multi-thousand-token contexts;
- a 3B/7B model is required;
- consumer-GPU generation comparisons are required;
- datacenter memory-bandwidth / HBM comparisons are required.

The local CPU baseline remains useful even after cloud testing; it should not be discarded.
