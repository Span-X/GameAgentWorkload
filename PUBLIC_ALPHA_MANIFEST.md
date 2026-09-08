# Public alpha manifest

This publication candidate was prepared from the AWB0_v0.4 research baseline.

## Preserved research assets

- `saloon_64` historical scenario
- `saloon_64_layered` layered-cognition scenario
- canonical traces and reports for both scenarios
- i5-8250U + Qwen3.5-0.8B-Q8_0 real CPU baseline
- deterministic replay and real llama.cpp replay paths
- 17 regression tests

## Public-facing additions

- `README.md` reframed around the general GameAgentWorkload research question
- Apache-2.0 `LICENSE`
- `CONTRIBUTING.md`
- `CITATION.cff`
- `CHANGELOG.md`
- `docs/ARCHITECTURE.md`
- `docs/CLOUD_HARDWARE_PLAN.md`
- scenario/hardware GitHub issue templates

## Cleanup

- removed redundant sweep trace directories from the initial publication candidate
- removed duplicate v0.4 trace directory
- moved historical v0.4 design notes under `docs/history/`
- aligned project/package version metadata to `0.4.0a0`
- updated CLI/report branding to `GameAgentWorkload 0.4-alpha`
- excluded local real-replay outputs and Python caches from version control

## Validation

- `python -m pytest -q` -> `17 passed`
- `saloon_64_layered` synthetic smoke completed successfully after cleanup
