# Bedroom-1 Exception-Oriented Control State v0.1

## Motivation

v0.5.0a4 improved the bed-obstruction case by exposing generic, runtime-derived skill-precondition state, but the same representation also exposed every nominal SATISFIED precondition. On the RTX 6000 Ada real-model run, the obstruction and projectile cases resolved correctly while the complex social interaction regressed to `continue_sleep_skill`. This revision tests whether redundant nominal-state disclosure biased the small controller toward continuation.

## Design rule

The runtime retains the **complete** objective skill-precondition state for execution, audit, and metrics. The direct-controller prompt receives only **precondition exceptions**:

- `violated`;
- `unknown`.

Stable `satisfied` entries are omitted from the model-facing projection. This is an exception/delta projection, not deletion of runtime truth.

## Why this is domain-neutral

The rule is not specific to beds, pests, projectiles, or social visitors. A persistent controller usually benefits from decision-relevant deltas and exceptions rather than a repeated dump of every nominal fact. The same representation can apply to UI agents, robots, and game agents:

```text
full persistent runtime state
        ↓
changed / violated / unknown execution facts
        ↓
small-model control decision
```

## Bedroom examples

Normal:

```json
"precondition_exceptions": []
```

Bed obstruction:

```json
"precondition_exceptions": [
  {
    "id": "sleep_surface_clear",
    "status": "violated",
    "evidence": ["observed_entity occupies intended_sleep_surface"]
  }
]
```

Projectile:

```json
"precondition_exceptions": [
  {
    "id": "body_trajectory_clear",
    "status": "violated",
    "evidence": ["observed_object_trajectory trajectory_intersects agent_body"]
  }
]
```

Complex visitor keeps its perceived change, attributes, and structural interaction relation, but no longer receives unrelated `sleep_surface_clear=satisfied` / `body_trajectory_clear=satisfied` reminders.

## Invariants

- no oracle interrupt Gate before direct inference;
- no oracle budget controller before direct inference;
- no engineered risk/relevance/complexity scalars exposed;
- no case name exposed as a control label;
- full precondition state remains available to runtime metrics;
- hard deadlines remain deterministic runtime facts.

## Real-hardware test

Re-run the same four-core-case command on the same model/server/hardware. The experiment asks whether exception-only projection preserves the a4 obstruction/projectile gains while removing the social-interaction regression.
