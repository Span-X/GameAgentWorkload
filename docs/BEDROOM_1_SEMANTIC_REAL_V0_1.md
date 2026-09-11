# Bedroom-1 Semantic Real v0.1

This experiment is the first Bedroom-1 path in which a real model chooses the NPC action.
It is intentionally narrower than the full cognition architecture.

## What is real

For each interrupted Bedroom-1 case, the model receives the NPC goal, current skill and phase,
the perception-bounded observation summary/attributes, and only currently executable action ids
plus descriptions. A llama.cpp chat completion chooses the action. Client-side wall latency and
TTFT are measured on the actual machine running llama.cpp.

The prompt does **not** expose the case name, reference-policy action roles, cognition mode, or the
engineered gate/budget scalars (`goal_relevance`, `prediction_error`, `immediate_risk`, `novelty`,
`uncertainty`, `skill_validity`, `action_ambiguity`, `planning_horizon`).

The projectile result is latency-aware: Bedroom-1 reserves 15 ms of its 120 ms time-to-contact for
minimal action commit, leaving a 105 ms decision deadline. A semantically correct protective action
that becomes ready after that deadline is recorded as a real-time failure.

## What is still oracle/synthetic

This v0.1 does **not** ask the model whether cognition should be invoked, and it does not ask the
model how deeply it should think. The existing feature-based CognitiveInterruptGate and
CognitiveBudgetController are held fixed as an oracle experimental condition. Therefore this path
isolates two questions only:

1. Can the real model choose an acceptable executable action from the structured observation?
2. Does the real hardware/runtime make that decision ready before the relevant cognition/deadline budget?

`ReferenceDecisionPolicy` is not used by this experiment.

## Run against llama.cpp

Start a llama.cpp server with an instruct/chat-capable GGUF, then run:

```powershell
python run.py bedroom-real `
  --base-url http://127.0.0.1:8080 `
  --model local-model `
  --cases roach,projectile,complex_visitor `
  --runs 1
```

Outputs:

- `traces/bedroom_1_real/bedroom_1.semantic_real.json`
- `traces/bedroom_1_real/bedroom_1.semantic_real.csv`

For hardware comparison, keep the model file/quantization, prompt style, case set, max tokens, and
llama.cpp settings fixed. Increase `--runs` only after the one-run smoke succeeds.

## Interpretation boundary

A successful run is not yet evidence that Bedroom-1 has solved metacognition. It is evidence only
for semantic action selection under a fixed interrupt/budget policy. Later experiments may replace
the oracle gate and budget controller independently.
