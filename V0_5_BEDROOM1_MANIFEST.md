# GameAgentWorkload v0.5 Bedroom-1 Candidate Manifest

Status: **experimental branch candidate**, not a final benchmark standard.

## Base

This working tree was derived from the conversation's `GameAgentWorkload_public-alpha.zip`,
which itself was derived from AWB0 v0.4. The subsequently reported public-repo packaging
fix is represented locally with explicit setuptools package discovery for `awb*` and
`scenarios*`.

Because the GitHub connector was unavailable in this conversation while this candidate
was built, the tree should be diffed against the current public `main` before committing.
Do not blindly overwrite unrelated repository changes.

## Added

- `awb/bedroom.py`
  - generic `ObservationDelta`;
  - `CognitiveInterruptGate`;
  - `CognitiveBudgetController`;
  - progressive cognition stages;
  - behavioral-envelope evaluation;
  - deterministic reference decision policy explicitly marked as non-normative.
- `scenarios/bedroom_1.py`
  - normal control;
  - roach-like sleep-surface obstruction;
  - imminent projectile with a 120 ms world deadline;
  - consequential ambiguous social visitor.
- `tests/test_bedroom_1.py`
  - core-case regressions;
  - unseen broken-bed extension case;
  - harmless thunder non-overtrigger regression.
- `docs/BEDROOM_1_EXPERIMENTAL_CONTRACT.md`
- `docs/BEDROOM_1_HARDWARE_MATRIX.md`
- canonical Bedroom-1 traces and reports under `traces/bedroom_1/`.
- `bedroom-suite` CLI command.

## Intentional architecture boundary

The core gate/budget runtime never branches on `roach`, `projectile`,
`complex_visitor`, or other case names. Cases supply perception-bounded features and
currently available affordances. Extension cases should be added without case-name
branches in the core runtime.

## Validation performed

- `python -m pytest -q` -> **23 passed**
- `python -m compileall -q awb scenarios run.py` -> pass
- `python -c "import awb; print(awb.__version__)"` -> `0.5.0a0`
- `python run.py --help` -> pass
- `python run.py bedroom-suite --seed 7 --concurrency 1` -> all four core cases pass
- editable install validated with `python -m pip install -e ".[dev]" --no-build-isolation`
  in the network-disabled build environment.

## Important non-claims / next work

This candidate does **not** yet claim to solve semantic LLM cognition. The synthetic
closed loop uses a deterministic reference policy after workload timing completes.
The existing `replay-real` path can already measure the exact Bedroom-1 inference
requests on i5 / RTX hardware, but it currently constructs token-count-controlled
prompts rather than asking the real model to semantically choose the action.

Next implementation target:

1. semantic model-backed `DecisionBackend` with structured action output;
2. measured hardware latency fed back into the Bedroom-1 world clock;
3. open-loop + closed-loop RTX 3090 / 4090 / 5090 comparison;
4. blind extension cases used to attack the architecture without adding core
   case-specific branches.

## 0.5.0a1 semantic-real addendum

`bedroom-real` is a separate real-model experiment. It does not use `ReferenceDecisionPolicy` for
action selection. The existing CognitiveInterruptGate and CognitiveBudgetController remain fixed
oracle-feature controls in this addendum; model-driven interruption and model-driven cognition-depth
selection are explicitly out of scope.
