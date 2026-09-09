# Contributing

GameAgentWorkload is intentionally open early. Strong criticism, counterexamples, and competing implementations are welcome.

## Good contributions

Especially useful contributions include:

- a new game-agent workload scenario;
- a correction to an unrealistic deadline or arrival pattern;
- a new runtime/backend implementation;
- reproducible GPU/NPU/CPU results;
- scheduler or cache-policy experiments;
- profiling of rendering/inference contention;
- tests that expose nondeterminism or world/knowledge leakage;
- design critiques with concrete alternatives.

## Scenario proposals

A proposed scenario should document:

1. the game-agent archetype (open-world NPC, FPS teammate, crowd, RTS agent, etc.);
2. objective world events;
3. what each agent is allowed to observe;
4. how/when agents become active;
5. cognition-layer/task arrivals;
6. deadlines and why they are plausible;
7. expected output size / structure;
8. interruption or replanning conditions;
9. which parts are synthetic assumptions.

Do not silently encode a model-quality preference into a systems benchmark.

## Hardware results

Please record at minimum:

- provider / machine source;
- exact GPU and GPU count;
- CPU model;
- system RAM and VRAM;
- operating system;
- driver and CUDA/runtime versions;
- llama.cpp (or other backend) version/commit;
- model filename and ideally hash;
- launch command and context size;
- parallel slots / batching configuration;
- GameAgentWorkload commit SHA;
- trace digest;
- exact benchmark command.

Consumer-GPU and datacenter-GPU results should not be compared as if only the GPU model changed when host CPU, power limits, PCIe topology, or virtualization differ.

## Development

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
```

All existing tests should pass before a PR is opened. New behavior should include deterministic regression coverage where practical.

## Design principles

Please preserve these boundaries unless a proposal explicitly argues for changing them:

- world truth is not automatically agent knowledge;
- semantic state reuse is not automatically KV-cache reuse;
- synthetic timing is labeled synthetic;
- real hardware claims require real hardware measurements;
- historical canonical scenarios/traces should not be silently rewritten.

## Discussion first is fine

For large changes, open an Issue/Discussion with the architecture argument before investing heavily in code. This project is still at the stage where changing the question can be more valuable than optimizing the current answer.
