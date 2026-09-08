from __future__ import annotations

from awb.backends import FakeBackend
from awb.real_replay import _deadline_flags
from awb.tracing import TraceRecorder
from awb.world import World
from scenarios import saloon_64_layered


def _run_layered(enable_cognitive_loop: bool = False):
    trace = TraceRecorder()
    world = World(
        seed=7,
        backend=FakeBackend(max_concurrency=4),
        trace=trace,
        enable_cognitive_loop=enable_cognitive_loop,
    )
    saloon_64_layered.build(world)
    world.activation_snapshot()
    world.run(saloon_64_layered.handle)
    return world, trace


def test_layered_scenario_has_reflex_and_model_layers():
    _, trace = _run_layered(False)
    reflex = [r for r in trace.records if r.get("event") == "reflex_action"]
    requests = [r for r in trace.records if r.get("event") == "inference_request"]
    layers = {r.get("cognitive_layer") for r in requests}
    assert reflex
    assert {"reactive", "cognitive", "strategic"}.issubset(layers)
    assert all(r.get("cognitive_layer") == "reflex" for r in reflex)


def test_reactive_requests_are_small_structured_work():
    _, trace = _run_layered(False)
    reactive = [
        r for r in trace.records
        if r.get("event") == "inference_request" and r.get("cognitive_layer") == "reactive"
    ]
    assert reactive
    assert max(r["input_tokens"] for r in reactive) <= 128
    assert max(r["expected_output_tokens"] for r in reactive) <= 2


def test_deadline_decomposition():
    assert _deadline_flags(220, 0.0, 100.0, 80.0) == (False, False, False, False)
    assert _deadline_flags(220, 150.0, 100.0, 80.0) == (True, False, True, False)
    assert _deadline_flags(220, 50.0, 300.0, 250.0) == (True, True, False, True)
