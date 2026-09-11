# Meta-control competing architectures

GameAgentWorkload should not assume a dedicated learned Gate is the final answer. The current evidence supports evaluating multiple architectures under the same persistent-agent workload.

## A. Engineered/Oracle Gate + SLM

Purpose: historical/scaffold baseline.

```text
engineered features -> interrupt/budget logic -> small model action
```

Useful for isolating model action quality, but not a satisfying final architecture because humans pre-solve much of the cognition-demand problem.

## B. Always-on small SLM direct control

```text
structured control state
        ↓
small System-1 model
        ↓
CONTINUE | local ACTION | ESCALATE
```

Advantages:

- one semantic inference can combine change interpretation and simple action selection;
- removes a separately trained classifier from the single-agent hot path;
- current RTX 6000 Ada evidence shows tens-of-milliseconds hot-path latency for Qwen3.5-0.8B-Q8_0.

Open risks:

- semantic failure on subtle affordance changes;
- over-confidence or over-escalation;
- aggregate throughput under many concurrent agents;
- prompt/prefix/cache dependence;
- GPU contention with rendering.

## C. Learned tiny MetaControl + SLM

Future alternative:

```text
domain state / latent state
        ↓
tiny learned meta-control policy
        ↓
continue | allocate cognition
        ↓
SLM / larger Brain
```

The policy should ideally be supervised by outcomes / counterfactual compute allocation rather than by manually labeling every scenario `FAST`, `URGENT`, or `DELIBERATE`.

This architecture becomes more attractive if always-on SLM calls fail at concurrency, power, latency-tail, or graphics-contention constraints.

## D. Hard runtime admission + direct SLM

Current strong candidate:

```text
exception/delta control state
        ↓
small System-1 model
        ↓
candidate action or ESCALATE
        ↓
deterministic runtime admission
        ↓
execute locally or route to System-2
```

The learned model handles semantic meaning. Deterministic runtime owns facts that should not require learned inference:

- physical/action legality;
- hard deadlines;
- whether escalation can finish before a deadline;
- executor availability;
- hard permission/safety constraints.

This is not a case-specific rule system. It is a general separation between semantic decision making and execution feasibility.

## What to measure later

The key future question is the crossover point, not which architecture sounds cleaner on paper.

Measure across increasing agent count/event rate:

- semantic/local-resolution success;
- escalation quality;
- p50/p95/p99 action-ready latency;
- deadline violation probability;
- throughput and queueing;
- VRAM/RAM;
- energy per decision;
- graphics frame-time impact when applicable;
- cold, warm-uncached, and warm-reuse regimes separately.
