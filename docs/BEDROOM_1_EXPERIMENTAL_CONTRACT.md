# Bedroom-1 Experimental Contract v0.1

Bedroom-1 is the first deliberately small scenario family in GameAgentWorkload v0.5.
It exists to study **when a persistent agent should spend cognition at all** before
scaling back up to many-NPC worlds.

## Scope

One agent starts beside one usable bed with the persistent goal `sleep`. The normal
skill is intentionally trivial:

`face_bed -> sit_on_bed -> lie_down -> sleep`

The experiment changes only the observation injected between `sit_on_bed` and
`lie_down`.

The first four core cases are:

1. **normal** — no meaningful deviation; expensive cognition should not run.
2. **roach** — a goal-relevant but low-complexity obstruction on the sleep surface;
   the skill should interrupt and use short cognition.
3. **projectile** — an imminent trajectory intersecting the agent; the decision has
   a hard 120 ms world deadline.
4. **complex_visitor** — a trusted person starts a consequential, ambiguous social
   discussion; the sleep skill pauses and cognition deepens progressively.

## What the benchmark fixes

Bedroom-1 deliberately fixes perception upstream. The agent receives a compact,
structured `ObservationDelta` rather than pixels. This controls the experiment so
we can study cognition/runtime behavior without simultaneously benchmarking visual
perception.

The boundary remains strict:

`World Truth != Agent Observation != Agent Belief`

A later benchmark can replace the controlled perception adapter with VLM/engine
perception while retaining the same downstream contract.

## Generic interrupt contract

The runtime does **not** inspect case names such as `roach` or `projectile`.
`CognitiveInterruptGate` consumes generic normalized features:

- prediction error;
- current-goal relevance;
- immediate risk;
- current-skill validity;
- novelty.

The gate answers only: **must automatic skill execution pause?**

It does not decide how long to think.

## Cognitive budget contract

After an interrupt, `CognitiveBudgetController` separately estimates:

- uncertainty;
- action ambiguity;
- planning horizon;
- novelty;
- hard world deadline, if one exists.

It assigns one of three model-backed modes:

- `fast` — short cognition for a relatively obvious deviation;
- `emergency` — hard-deadline decision class;
- `deliberate` — progressive deepening: fast triage followed by a longer reasoning
  stage.

`automatic` means no model call was needed.

These are workload contracts, **not fixed implementations**. A future backend may
use rules, a tiny NN, an action head, an SLM/LLM, or dedicated hardware.

## Evaluation philosophy

Bedroom-1 does not prescribe one exact action string. Each case defines a
**behavioral envelope**:

- should the skill interrupt?
- what cognition-depth family is acceptable?
- what semantic action role must be represented (protective, restore precondition,
  social engagement, etc.)?
- did the world-level outcome remain valid (survival, sleep completion, social
  engagement)?
- was cognition called unnecessarily often?

This lets multiple reasonable behaviors pass while still catching architectural
failures.

## Case-expansion robustness

The main research protocol is adversarial case expansion:

1. implement the generic runtime from a small set of core cases;
2. freeze it;
3. add a new case using only world state, perception features, and affordances;
4. observe whether the existing runtime absorbs the case naturally;
5. record any core-runtime or prompt change required to make the new case pass.

A new case is **not** allowed to add a `case_name == ...` branch to the core gate,
budget controller, or decision runtime.

The benchmark becomes more credible as new cases require fewer case-specific
patches.

## Hardware experiments

Bedroom-1 emits ordinary `inference_request` records, so the existing
`replay-real` path can replay the exact request stream on real hardware.

Two hardware modes are intended:

1. **Open-loop replay** — same trace on i5-8250U, RTX 3090, 4090, 5090, etc. to
   isolate runtime/hardware latency.
2. **Closed-loop replay** — measured inference time advances the world clock so a
   late but semantically-good decision can still fail the game deadline.

The current v0.5.0a0 implementation establishes the workload/trace contract and a
deterministic reference policy. Semantic LLM decision execution and true
hardware-in-the-world closed-loop evaluation are the next layer, not silently
simulated here.
