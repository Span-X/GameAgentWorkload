# Changelog

## 0.5.0-alpha (Bedroom-1 experimental candidate)

- add one-agent Bedroom-1 core cases: normal / roach-like obstruction / imminent projectile / complex visitor
- separate Cognitive Interrupt Gate from Cognitive Budget Controller
- add progressive fast -> deliberate cognition depth for ambiguous cases
- add behavioral-envelope evaluation instead of one exact scripted action
- add generic inference-completion listeners for scenario runtimes
- add an unseen broken-bed extension regression proving case expansion without core case-name branches
- preserve v0.4 saloon baselines as historical scalability probes

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
