# Bedroom-1 Semantic Gate Probe v0.1

Status: experimental research probe.

## Why this exists

Bedroom-1 v0.5.0a1 still used manually engineered normalized scalars
(`goal_relevance`, `prediction_error`, `immediate_risk`, `novelty`,
`uncertainty`, `skill_validity`, `action_ambiguity`, `planning_horizon`) to
select whether cognition interrupts the current skill and how deep cognition
should be. The real model selected only the final action.

This probe asks a real model to classify the gate directly from the current
goal, current skill expectation, and perception-bounded observation. The
engineered scalar features, oracle score, case name, action roles, benchmark
envelope, and oracle cognition mode are not exposed to the model.

## Important limitation

This is *not* the intended final runtime gate. Calling an autoregressive LLM to
decide whether the expensive brain should be called is circular and too slow.
The real LLM is used only as a semantic probe. If the classification problem is
well-posed and case-expansion robust, later work can distill/replace it with a
cheap classifier, action head, tiny network, or other low-latency gate.

## Event-driven trigger

No world change means no gate call. A perception layer emits an
`ObservationDelta`; only then is the gate evaluated. The normal control has no
observation delta and therefore incurs zero gate/model latency.

## Labels

- `CONTINUE`: current skill can continue automatically.
- `FAST`: brief local cognition.
- `EMERGENCY`: hard immediate physical deadline / urgent threat.
- `DELIBERATE`: consequential, ambiguous, or long-horizon cognition.

The single label is a convenient experimental encoding. Architecturally,
interrupt/no-interrupt and cognition-budget depth remain distinct concepts and
may become separate heads sharing one cheap forward pass.

## Command

With llama.cpp running (reasoning disabled for the fast classification probe):

```powershell
python run.py bedroom-gate-real `
  --base-url http://127.0.0.1:8080 `
  --model local-model `
  --cases normal,roach,projectile,complex_visitor `
  --runs 1 `
  --max-tokens 8
```

Output is written to `traces/bedroom_1_gate_real/`.

## What this probe does not test

- final action selection;
- learned gate efficiency;
- visual perception;
- a final metareasoning/halting policy;
- multi-agent scheduling;
- production-quality cognition-depth control.
