# v0.5.0a5 Exception-Oriented Control-State Manifest

Base: v0.5.0a4 affordance / skill-precondition working tree.

Adds/changes:
- retain complete runtime `skill_preconditions` in `AgentControlState`;
- change the model-facing projection from full `skill_preconditions` to `precondition_exceptions`;
- expose only `violated` or `unknown` preconditions to the direct SLM;
- omit stable `satisfied` preconditions from the prompt;
- preserve perceived changes, observed attributes/relations, executable actions, and hard deadlines;
- update direct-control prompt style to `agent_direct_control_v0.3_exception_state`;
- add regressions proving social/normal prompts do not contain satisfied precondition noise while obstruction/projectile exceptions remain visible.

Research intent:
- test the domain-neutral hypothesis that persistent controllers should consume decision-relevant exceptions/deltas rather than a redundant dump of nominal state;
- do not add a visitor-specific prompt rule or a pest-specific branch.

Direct-path invariants retained:
- no engineered/oracle `CognitiveInterruptGate` call before inference;
- no oracle `CognitiveBudgetController` call before inference;
- no engineered Bedroom cognition-demand scalars exposed to the model;
- no hidden reasoning parsed as an executable action;
- hard world deadlines remain deterministic runtime facts.

Do not overwrite a newer GitHub branch wholesale; reconcile incrementally.
