# Bedroom-1 research results through v0.5.0a5

Status: research snapshot for GitHub consolidation before the playable persistent-NPC phase.

This document records what has actually been demonstrated, what failed, and what remains open. It is intentionally conservative: failures are retained as evidence rather than hidden by prompt tuning.

## Research progression

### v0.1-v0.4: workload/runtime foundations

The earlier AWB/GameAgentWorkload line established the basic workload harness:

- deterministic world/event traces;
- activation/scheduling/deadline accounting;
- persistent cognition and interruption/supersession semantics;
- strict separation between semantic cognition reuse and actual runtime KV/prefix reuse;
- real llama.cpp replay and hardware timing;
- layered workload reporting and explicit deadline-miss categories.

The early saloon workloads remain historical synthetic scalability probes. They are not presented as a final model of believable NPC cognition.

### v0.5 Bedroom-1: move from synthetic scale to one detailed persistent NPC

Bedroom-1 intentionally narrowed the research target to one NPC, one persistent goal (`sleep`), one room, and a small number of controlled perturbations. The key architectural rule was preserved:

```text
World Truth != Agent Observation/Knowledge != Agent Belief
```

The initial four core cases were:

- `normal`: no decision-relevant change;
- `roach`: a low-salience obstruction on the intended sleep surface;
- `projectile`: a hard-deadline physical threat;
- `complex_visitor`: a consequential social interruption.

These cases were used to expose architecture assumptions, not as an intended permanent benchmark taxonomy.

### v0.5.0a1: real semantic action selection

A real Qwen3.5-0.8B-Q8_0 model served through llama.cpp selected executable actions from perception-bounded prompts. The reference decision policy was removed from action selection.

A key failure occurred with unbounded reasoning on `projectile`: the model identified `protect_self` but continued reconsidering until the generation limit. This separated semantic competence from deliberation-halting and output-protocol competence.

Turning reasoning off and using a short final action channel made `projectile` semantically correct.

### v0.5.0a2: semantic Gate probe

The model was asked to classify cognition mode directly from structured observation, without exposure to the engineered Gate scalars.

Observed result on the 0.8B model:

```text
normal           -> automatic   (pass)
roach            -> emergency   (expected fast)
projectile       -> emergency   (pass)
complex_visitor  -> emergency   (expected deliberate)
```

Conclusion: a prompt-level four-way cognition-mode classifier was not a convincing replacement for meta-control. Importance, urgency, and deliberation demand were conflated.

### v0.5.0a3: always-on small-model direct control

The competing architecture removed the oracle interrupt/budget Gate from the direct path. A structured `AgentControlState` was sent to the 0.8B model, which could return an executable action or `ESCALATE`.

RTX 6000 Ada, reasoning disabled, Qwen3.5-0.8B-Q8_0:

- hard-deadline `projectile` repeatedly selected `protect_self` inside the 105 ms decision budget after warm-up;
- hot repeated-request latency was commonly in the tens of milliseconds;
- mixed core run: `normal`, `projectile`, and `complex_visitor` resolved locally 5/5, but `roach` selected `continue_sleep_skill` 5/5.

Interpretation: direct small-model control was fast enough to be architecturally interesting, but the raw control-state representation was not sufficient for a lower-salience skill-precondition violation.

### v0.5.0a4: generic affordance / skill-precondition state

A domain-neutral `GenericAffordanceResolver` and structural relation representation were added. The controller could receive runtime-derived execution facts such as a violated sleep-surface or body-trajectory precondition without receiving engineered cognition-demand scores.

Result on RTX 6000 Ada:

- `normal`: 5/5 continuation;
- `roach`: 5/5 `clear_sleep_surface`;
- `projectile`: 5/5 `protect_self`;
- `complex_visitor`: 0/5, regressed to `continue_sleep_skill`.

Interpretation: exposing all satisfied preconditions created a nominal-state bias that overwhelmed the new social interruption in the 0.8B controller.

### v0.5.0a5: exception-oriented control-state projection

The runtime kept the full objective precondition state, but the model-facing projection exposed only `violated` or `unknown` precondition exceptions. Stable `satisfied` preconditions were omitted.

Result on RTX 6000 Ada:

- `normal`: 5/5 `continue_sleep_skill`;
- `roach`: 5/5 `clear_sleep_surface`;
- `complex_visitor`: 5/5 `engage_socially`;
- `projectile`: 0/5 local resolution because the model consistently returned `ESCALATE`.

The `projectile` result should not be reduced to "the model did not understand the threat". It demonstrates a different architecture gap: the model requested more cognition even though the world deadline made escalation infeasible.

## Current interpretation

The evidence now favors a separation between semantic control and deterministic runtime admission:

```text
perception-bounded delta / exception state
                ↓
small System-1 model
                ↓
CONTINUE | local ACTION | ESCALATE request
                ↓
hard runtime admission
  - action legality
  - world deadlines
  - escalation ETA
  - hardware availability
  - safety margin
                ↓
admitted action / System-2 route
```

A future learned tiny MetaControl policy remains a competing architecture, not the assumed winner. The direct-SLM route is sufficiently fast on RTX 6000 Ada to justify continued investigation.

## Important non-claims

These results do not establish:

- a universal NPC architecture;
- stable p99 latency for uncached, high-concurrency production workloads;
- a final System-1/System-2 boundary;
- that RTX 6000 Ada performance equals an in-game rendering-contended deployment;
- that the four Bedroom cases are representative of all gameplay;
- that the learned Gate is unnecessary at scale.

The next phase intentionally moves away from repeatedly tuning these four cases.
