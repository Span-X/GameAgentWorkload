# v0.5.0a2 Semantic Gate Probe Manifest

This candidate is based on the v0.5.0a1 semantic-real working tree.

Adds:
- `awb/semantic_gate.py`
- `bedroom-gate-real` CLI
- semantic gate probe tests
- semantic gate design note

Preserves:
- Phase-1 oracle-feature `CognitiveInterruptGate`
- Phase-1 `CognitiveBudgetController`
- existing `bedroom-real` action-selection path
- synthetic Bedroom-1 regression path
- historical saloon baselines

Validation in the build environment:
- `pytest -q`: 35 passed
- Python compile check: passed

Research status:
- real model gate classification: implemented
- engineered scalar features exposed to gate model: no
- final low-cost gate: not claimed
- action selection in gate probe: not measured
- case-expansion robustness: next step

Do not overwrite a newer GitHub branch wholesale; reconcile incrementally.
