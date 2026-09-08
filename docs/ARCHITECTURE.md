# Architecture and research boundaries

GameAgentWorkload is organized around a strict separation between **world state**, **agent cognition**, **workload generation**, and **runtime execution**.

## 1. Objective world vs agent knowledge

The world simulator owns objective state. An agent does not receive that state directly.

```text
Objective world state
        ↓
Perception / observation boundary
        ↓
Agent working memory + beliefs
        ↓
Intent / plan
        ↓
Action
```

This prevents an agent from becoming accidentally omniscient. It also makes the same architecture useful for thinking about embodied agents and robotics, where the physical world is only partially observed.

## 2. Persistent cognition

The agent is not modeled as a stateless chat request. Cognition can persist across events and includes:

- working memory;
- beliefs;
- provisional and active intents;
- provisional and active plans;
- interruption, supersession, and replanning.

A world update may invalidate a target-dependent plan without erasing unrelated beliefs or memories.

## 3. Cognition layers are workload classes

Current scenarios use four classes:

| Layer | Intended role | Typical latency pressure |
|---|---|---|
| Reflex | immediate action primitive / action head | tens of ms |
| Reactive | compact status/gating decision | ~100-300 ms |
| Cognitive | high-level local reasoning | ~0.5-several s |
| Strategic | background/long-horizon work | seconds+ |

These are **contracts**, not permanent implementation prescriptions. A future reflex implementation could be a tiny neural policy or a transformer action head rather than a handwritten rule.

## 4. Runtime accounting is distinct from semantic reuse

An agent carrying a belief from a previous cognition step does **not** imply the inference runtime reused KV cache or skipped prefill.

Semantic reuse and runtime cache reuse must be measured separately. Real cache reuse should only be credited when the backend/runtime provides evidence for it.

## 5. Why traces matter

The simulator emits a deterministic canonical trace. That trace can be replayed on different runtimes/hardware while holding the workload constant.

This is necessary for comparisons such as:

```text
same trace
  ├─ CPU
  ├─ RTX 3090
  ├─ RTX 4090
  ├─ RTX 5090
  ├─ A100
  └─ H100
```

Without this separation, different agent behavior can be mistaken for a hardware performance difference.

## 6. Non-goals at this stage

GameAgentWorkload does not currently claim to:

- reproduce human cognition;
- define a universal game-agent architecture;
- measure agent believability or gameplay fun;
- replace game-engine physics/navigation/animation systems;
- establish final latency targets for all game genres;
- prove that one hardware architecture is superior.

The purpose of the alpha is to expose assumptions early enough for the community to challenge them.
