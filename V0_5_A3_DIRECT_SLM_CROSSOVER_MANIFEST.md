# v0.5.0a3 Direct SLM Crossover Manifest

Base: v0.5.0a2 semantic-gate working tree.

Adds:
- `awb/control_state.py` with domain-neutral `AgentControlState` and a Bedroom adapter;
- `awb/direct_control.py` with the always-on direct-SLM experiment;
- `bedroom-direct-real` CLI;
- direct-controller regression tests;
- optional discarded warmup requests and request concurrency;
- structured JSON prompts with action-id / `ESCALATE` output contract.

Direct path invariants:
- does not call the engineered/oracle `CognitiveInterruptGate`;
- does not call the oracle `CognitiveBudgetController`;
- does not expose Bedroom engineered gate scalars to the model;
- does not use cognition-mode labels as model outputs;
- does not mine reasoning text for an executable decision;
- derives the projectile 105 ms decision deadline only from the 120 ms world deadline minus the existing 15 ms motor-commit reserve.

Preserves:
- v0.5.0a2 semantic gate probe;
- v0.5.0a1 oracle-gated real semantic action baseline;
- synthetic Bedroom-1 regression path;
- historical saloon baselines.

Validation in the build environment:
- `pytest -q`: 42 passed;
- `python -c "import awb; print(awb.__version__)"`: `0.5.0a3`;
- `python run.py --help`: includes `bedroom-direct-real`;
- direct CLI plumbing exercised against a local streaming HTTP stub.

Research status:
- direct SLM crossover path: implemented;
- System-2 escalation execution: not implemented;
- extension-case registry: not implemented;
- learned tiny MetaControl model: intentionally not implemented yet.

Do not overwrite a newer GitHub branch wholesale; reconcile incrementally.
