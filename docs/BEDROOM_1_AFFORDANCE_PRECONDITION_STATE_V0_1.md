# Bedroom-1 Affordance / Skill-Precondition State v0.1

## Motivation

The v0.5.0a3 direct-SLM experiment showed that a small model can resolve explicit, high-salience decisions such as an imminent projectile without the engineered/oracle Gate, while a lower-salience bed obstruction can be mishandled. This revision tests a representation-layer hypothesis rather than adding a case-specific prompt rule.

## Boundary

`AgentControlState` now exposes two additional kinds of runtime state:

- **observed structural relations**, such as `observed_entity occupies intended_sleep_surface`;
- **objective skill-precondition state**, such as `sleep_surface_clear = violated`.

These are intended to represent execution/affordance facts that a game engine or domain adapter can determine cheaply. They are not cognition-demand labels.

The direct-controller prompt still does **not** expose the Bedroom engineered Gate scalars:

- `goal_relevance`;
- `prediction_error`;
- `immediate_risk`;
- `novelty`;
- `uncertainty`;
- `skill_validity`;
- `action_ambiguity`;
- `planning_horizon`.

It also does not call `CognitiveInterruptGate` or `CognitiveBudgetController` before the model.

## Generic resolver

`awb.affordance.GenericAffordanceResolver` consumes:

1. a set of skill-precondition specifications;
2. perception-bounded structural relations;
3. a baseline status for already-running skill requirements.

It currently supports two objective relation families:

- `resource_clear`: violated by `occupies` / `blocks` relations on the target resource;
- `trajectory_clear`: violated by `trajectory_intersects` on the target body/resource.

The resolver does not inspect `roach`, `projectile`, `pest_like`, or other case/entity labels.

## Bedroom adapter

For `sleep_at_bed`, the current objective preconditions are:

- `sleep_surface_clear` targeting `intended_sleep_surface`;
- `body_trajectory_clear` targeting `agent_body`.

Examples:

```text
observed_entity occupies intended_sleep_surface
    -> sleep_surface_clear = violated
```

```text
observed_object_trajectory trajectory_intersects agent_body
    -> body_trajectory_clear = violated
```

A future real game integration should normally obtain these relations directly from the engine/perception layer rather than reconstructing them from Bedroom-1 string fields.

## Behavioral-envelope correction

The roach case no longer treats `clear_sleep_surface` as the only acceptable immediate semantic response. `inspect_change` is also accepted because the observation is probabilistic (`classification_confidence = 0.88`) and inspection can be a reasonable immediate next action. `continue_sleep_skill` remains a failure while the sleep-surface precondition is violated.

This preserves the benchmark principle that correctness is an acceptable behavioral envelope, not one scripted human action.

## Next experiment

Re-run the direct SLM path on the same hardware and compare with v0.5.0a3:

```bash
python run.py bedroom-direct-real \
  --base-url http://127.0.0.1:8080 \
  --model local-model \
  --cases normal,roach,projectile,complex_visitor \
  --runs 5 \
  --max-tokens 8 \
  --concurrency 1
```

The key question is whether the representation change improves the obstruction case without regressing projectile, normal continuation, or social interaction. Do not interpret one repeated-prompt latency series as an uncached-latency benchmark.
