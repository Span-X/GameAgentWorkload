# Bedroom-1 Direct SLM Crossover v0.1

## Question

Can a sufficiently fast small language model replace a separate engineered meta-control gate for simple persistent-agent decisions?

This experiment is a competing architecture, not a claim that an always-on SLM is universally better.

## Direct path

```text
Bedroom domain observation
        -> Bedroom adapter
        -> AgentControlState
        -> small real model
        -> executable action id | ESCALATE
```

The direct path does **not** invoke `CognitiveInterruptGate` or `CognitiveBudgetController` before the model call.

`AgentControlState` exposes only:

- current goal;
- current skill and phase;
- skill expectation;
- perceived change;
- observed attributes;
- currently executable action ids/descriptions;
- hard deadline facts when the world exposes one.

It does not expose the Bedroom-1 engineered features `goal_relevance`, `prediction_error`, `immediate_risk`, `novelty`, `uncertainty`, `skill_validity`, `action_ambiguity`, or `planning_horizon`.

## Output contract

The model returns exactly one executable action id or `ESCALATE`.

`ESCALATE` is a routing signal, not a world action. v0.1 records it but does not execute a larger System-2 model. Therefore an escalation does not count as local task resolution, and cannot save a hard-deadline projectile case by itself.

The parser deliberately ignores hidden/reasoning text. Only the final executable channel is accepted.

## Deadline semantics

For Bedroom-1 projectile, the world reports 120 ms to contact. The current experimental contract reserves 15 ms for motor/action commit, leaving a 105 ms direct decision deadline. This deadline is derived from the world contract, not from the oracle cognitive-budget controller.

## CLI

```bash
python run.py bedroom-direct-real \
  --base-url http://127.0.0.1:8080 \
  --model local-model \
  --cases normal,roach,projectile,complex_visitor \
  --runs 5 \
  --max-tokens 8 \
  --concurrency 1 \
  --warmup 1 \
  --warmup-case roach
```

`--warmup` requests are discarded from measured results but recorded in run configuration. Choosing a different warmup case helps avoid exact-prompt warmup of the focal case, although runtime prefix/cache reuse may still occur and must be treated as an experimental factor.

## Crossover study

The intended comparison is:

1. engineered/oracle Gate + small model action selection (`bedroom-real`);
2. always-on direct small model (`bedroom-direct-real`);
3. later: learned tiny MetaControl policy + small/large model cascade;
4. hard runtime constraints + direct small model.

Scale concurrency and case diversity before concluding that a separate gate is unnecessary. Report cold/warm behavior, semantic success, deadline violations, and exact prompt hashes rather than only mean latency.
