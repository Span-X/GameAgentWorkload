# Changelog

## 0.5.0-alpha (Bedroom-1 experimental candidate)

- add one-agent Bedroom-1 core cases: normal / roach-like obstruction / imminent projectile / complex visitor
- separate Cognitive Interrupt Gate from Cognitive Budget Controller
- add progressive fast -> deliberate cognition depth for ambiguous cases
- add behavioral-envelope evaluation instead of one exact scripted action
- add generic inference-completion listeners for scenario runtimes
- add an unseen broken-bed extension regression proving case expansion without core case-name branches
- preserve v0.4 saloon baselines as historical scalability probes

## 0.5.0a5 - Exception-oriented control-state projection

- Keep full runtime skill-precondition truth, but expose only violated/unknown exceptions to the always-on SLM.
- Stop redundantly injecting stable satisfied preconditions into every direct-control prompt.
- Preserve structural relations and perceived changes so social/non-precondition events remain decision-visible.
- Add regression coverage for normal/social no-noise projection and obstruction/projectile exception visibility.
- Do not add case-specific visitor/pest rules; this is a representation-layer change.

## 0.5.0a4 - Runtime-derived affordance / skill-precondition state

- Add domain-neutral structural relations and skill-precondition state to `AgentControlState`.
- Add a generic affordance resolver for objective `resource_clear` and `trajectory_clear` requirements; it does not inspect case names, entity classes, cognition-mode labels, or engineered risk/relevance scalars.
- Bedroom adapter normalizes perceived relations into runtime facts and derives `sleep_surface_clear` / `body_trajectory_clear` status before direct SLM control.
- Keep the direct SLM path free of the oracle interrupt gate and oracle budget controller.
- Correct the roach behavioral envelope so both `inspect_change` and a direct precondition-restoring action are acceptable immediate next actions; blindly continuing sleep remains a failure.
- Preserve the engineered-feature Gate as a historical/oracle baseline rather than deleting it.

## 0.4.0-alpha — Research preview

- Added layered cognition workload (`reflex`, `reactive`, `cognitive`, `strategic`).
- Added real llama.cpp trace replay with exact token-count prompts.
- Added deadline decomposition into intrinsic service vs queue-induced misses.
- Preserved the first real i5-8250U CPU baseline.
- Preserved historical `saloon_64` while adding `saloon_64_layered`.
- Added persistent structured cognition and interruption/replanning semantics.

## 0.5.0a1 - Bedroom-1 semantic real action path

- Add a real llama.cpp chat-backed Bedroom-1 action-selection experiment (`bedroom-real`).
- Keep the synthetic/reference Bedroom-1 suite unchanged as a deterministic regression path.
- Hide engineered interrupt/budget scalars and case names from the semantic model prompt.
- Record real wall latency / TTFT and latency-aware projectile outcomes.
- Explicitly scope this experiment to action selection; interrupt selection and cognition-depth allocation remain oracle-feature controlled.

## 0.5.0a3 - Direct SLM crossover experiment

- Add domain-neutral `AgentControlState` plus Bedroom-1 adapter.
- Add `bedroom-direct-real`: an always-on small-model controller that returns an executable action id or `ESCALATE`.
- Remove the engineered/oracle interrupt gate and cognitive-budget controller from this experimental path.
- Serialize the direct-controller observation as structured JSON and keep engineered Bedroom gate scalars out of the model input.
- Add concurrency and explicit discarded warmup controls for hardware crossover experiments.
- Keep hard world deadlines separate from cognition-mode labels; projectile retains the 120 ms contact / 105 ms decision contract.
- Do not parse hidden reasoning as an executable action.

## Research snapshot handoff — 2026-09-10

- Preserves the direct-SLM/affordance/exception-state research line through `0.5.0a5`.
- Adds a consolidated research-results note and competing meta-control architecture note.
- Adds the design contract for the next `v0.6` Playable Persistent NPC Loop.
- `v0.6` runtime implementation is intentionally not included in this snapshot commit.
