# v0.5.0a4 Affordance / Skill-Precondition State Manifest

Base: v0.5.0a3 direct-SLM crossover working tree.

Adds:
- `awb/affordance.py` with domain-neutral structural relation and skill-precondition contracts;
- generic affordance/precondition resolution independent of Bedroom case names/entity classes;
- `skill_preconditions` and `observed_relations` in `AgentControlState`;
- runtime-derived precondition evidence in direct-controller reports;
- regression tests for entity-class-independent resource occupancy and trajectory intersection;
- a behavioral-envelope correction accepting either inspection or direct restoration as a reasonable immediate response to an uncertain bed obstruction.

Direct-path invariants retained:
- no engineered/oracle `CognitiveInterruptGate` call before inference;
- no oracle `CognitiveBudgetController` call before inference;
- no engineered Bedroom risk/relevance/complexity scalars exposed to the model;
- no case name exposed as a control label;
- no hidden reasoning parsed as an action;
- hard world deadlines remain deterministic runtime facts.

Important distinction:
- `skill_precondition.status` is an objective affordance/execution fact derived from structural relations;
- it is **not** an engineered cognition-demand score and does not prescribe FAST / EMERGENCY / DELIBERATE.

Validation in the build environment:
- `pytest -q`: 49 passed;
- `python -c "import awb; print(awb.__version__)"`: `0.5.0a4`;
- `python run.py --help`: includes `bedroom-direct-real`;
- direct-controller prompt regression verifies engineered Gate features remain absent.

Research status:
- direct SLM + affordance/precondition representation: implemented;
- System-2 escalation execution: not implemented;
- extension-case registry: not implemented;
- learned tiny MetaControl policy: intentionally deferred pending direct-SLM crossover evidence.

Do not overwrite a newer GitHub branch wholesale; reconcile incrementally.
