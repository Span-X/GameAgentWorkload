from pathlib import Path

from awb.backends import FakeBackend, LlamaCppBackend
from awb.metrics import build_report
from awb.replay import request_digest_from_trace
from awb.tracing import TraceRecorder
from awb.types import ActivationLevel, AgentState, RequestPriority
from awb.world import World
from scenarios import saloon_64


def run_once(seed=7, concurrency=4, cognitive_loop=True):
    trace = TraceRecorder()
    world = World(
        seed=seed,
        backend=FakeBackend(max_concurrency=concurrency),
        trace=trace,
        enable_cognitive_loop=cognitive_loop,
    )
    saloon_64.build(world)
    world.activation_snapshot()
    world.run(saloon_64.handle)
    return world, trace


def test_unobserved_event_is_not_global_knowledge():
    trace = TraceRecorder()
    world = World(seed=1, backend=FakeBackend(), trace=trace, enable_cognitive_loop=False)
    world.add_agent(AgentState(agent_id=1, location_id=1, activation=ActivationLevel.ACTIVE))
    world.add_agent(AgentState(agent_id=2, location_id=2, activation=ActivationLevel.ACTIVE))
    event = world.events.push(0, "secret_event", source_id=99, location_id=1)
    world.clock_ms = 0
    observers = world.observe_local_event(event)
    assert observers == [1]
    assert not world.agents[2].cognition.beliefs
    assert not world.agents[2].cognition.working_memory.observations


def test_observation_enters_working_memory_but_not_automatic_belief():
    trace = TraceRecorder()
    world = World(seed=1, backend=FakeBackend(), trace=trace, enable_cognitive_loop=False)
    world.add_agent(AgentState(agent_id=1, location_id=1, activation=ActivationLevel.ACTIVE))
    event = world.events.push(0, "unknown_noise", source_id=99, location_id=1)
    world.clock_ms = 0
    world.observe_local_event(event)
    agent = world.agents[1]
    assert len(agent.cognition.working_memory.observations) == 1
    assert not agent.cognition.beliefs


def test_threat_promotes_agents_and_updates_structured_cognition():
    world, trace = run_once(concurrency=16)
    changed = [r for r in trace.records if r.get("event") == "activation_changed"]
    requests = [r for r in trace.records if r.get("event") == "inference_request"]
    assert any(r["new"] >= int(ActivationLevel.COGNITIVE) for r in changed)
    assert any(r["task_type"] == "threat_assessment" for r in requests)
    assert any(r["task_type"] == "combat_replan" for r in requests)
    assert any("armed_player_is_dangerous" in a.cognition.beliefs for a in world.agents.values())
    assert any(a.cognition.intent is not None for a in world.agents.values())
    assert any(a.cognition.plan is not None for a in world.agents.values())


def test_target_death_only_invalidates_genuine_target_dependent_work():
    _, trace = run_once(concurrency=8)
    superseded = [r for r in trace.records if r.get("event") == "inference_superseded"]
    interrupted = [r for r in trace.records if r.get("event") == "inference_interrupted"]
    reqs = {r["request_id"]: r for r in trace.records if r.get("event") == "inference_request"}
    affected = superseded + interrupted
    assert affected
    assert all(reqs[r["request_id"]]["task_type"] == "assist_target" for r in affected)
    assert not any(reqs[r["request_id"]]["task_type"] == "threat_assessment" for r in affected)


def test_interruption_preserves_belief_or_working_memory_state():
    _, trace = run_once(concurrency=24)
    revisions = [r for r in trace.records if r.get("event") == "cognitive_state_revised"]
    assert revisions
    assert any("beliefs" in r["preserved_components"] or "working_memory" in r["preserved_components"] for r in revisions)
    assert not any("retained_prefill_tokens" in r for r in trace.records)
    assert not any("discarded_prefill_tokens" in r for r in trace.records)


def test_logical_state_persistence_does_not_fake_kv_cache_reuse():
    _, trace = run_once(concurrency=16)
    reqs = [r for r in trace.records if r.get("event") == "inference_request"]
    assert any(r.get("carried_state_tokens", 0) > 0 for r in reqs)
    assert all(r.get("cache_reusable_tokens", 0) == 0 for r in reqs)
    assert all(r.get("fresh_input_tokens") == r.get("input_tokens") for r in reqs)


def test_priority_deadline_scheduler_starts_critical_before_background():
    trace = TraceRecorder()
    backend = FakeBackend(max_concurrency=1)
    world = World(seed=1, backend=backend, trace=trace, enable_cognitive_loop=False)
    world.add_agent(AgentState(agent_id=1, location_id=1))
    world.create_request(
        agent_id=1,
        task_type="memory",
        priority=RequestPriority.BACKGROUND,
        deadline_ms=8000,
        input_tokens=500,
        output_tokens=20,
        cognitive_domain="memory",
    )
    world.create_request(
        agent_id=1,
        task_type="combat",
        priority=RequestPriority.CRITICAL,
        deadline_ms=100,
        input_tokens=500,
        output_tokens=20,
        cognitive_domain="threat",
    )
    world.scheduler.start_ready(0)
    starts = [r for r in trace.records if r.get("event") == "inference_started"]
    reqs = {r["request_id"]: r for r in trace.records if r.get("event") == "inference_request"}
    assert reqs[starts[0]["request_id"]]["task_type"] == "combat"


def test_request_stream_is_deterministic(tmp_path: Path):
    _, trace1 = run_once(seed=7)
    _, trace2 = run_once(seed=7)
    assert trace1.request_stream_digest() == trace2.request_stream_digest()
    path = trace1.write_jsonl(tmp_path / "trace.jsonl")
    assert request_digest_from_trace(path) == trace1.request_stream_digest()


def test_continuous_cognition_generates_pulses_and_commits_state():
    world, trace = run_once(concurrency=16, cognitive_loop=True)
    pulses = [r for r in trace.records if r.get("event") == "cognitive_pulse"]
    commits = [r for r in trace.records if r.get("event") == "cognitive_state_committed"]
    ambient = [r for r in trace.records if r.get("event") == "inference_request" and r.get("task_type") == "ambient_cognition"]
    assert pulses
    assert ambient
    assert commits
    assert any(agent.cognition.revision > 0 for agent in world.agents.values())


def test_report_has_structured_cognition_metrics():
    world, trace = run_once(concurrency=8)
    report = build_report("saloon_64", len(world.agents), trace.records, trace.request_stream_digest())
    assert report.requests > 0
    assert report.completed > 0
    assert report.belief_updates > 0
    assert report.working_memory_updates > 0
    assert report.intent_transitions > 0
    assert report.carried_state_tokens > 0
    assert report.simulated_cache_reusable_tokens == 0


def test_llamacpp_exact_prompt_builder_hits_target_without_server():
    backend = LlamaCppBackend()
    backend._token_cache["AWB agent=1 task=x domain=general structured state observation belief intent plan. "] = [1, 2, 3]
    backend._token_cache[" persistent world state observation memory belief intent plan context update "] = [4, 5]
    record = {
        "agent_id": 1,
        "task_type": "x",
        "cognitive_domain": "general",
        "input_tokens": 17,
        "carried_state_tokens": 0,
    }
    prompt = backend.make_exact_prompt_tokens(record)
    assert len(prompt) == 17


def test_llamacpp_cache_experiment_uses_previous_prefix_without_claiming_cache_hit():
    backend = LlamaCppBackend()
    backend._token_cache["AWB agent=2 task=x domain=general structured state observation belief intent plan. "] = [1, 2]
    backend._token_cache[" persistent world state observation memory belief intent plan context update "] = [3, 4]
    record = {
        "agent_id": 2,
        "task_type": "x",
        "cognitive_domain": "general",
        "input_tokens": 10,
        "carried_state_tokens": 4,
    }
    previous = list(range(20, 30))
    prompt = backend.make_exact_prompt_tokens(record, previous_prompt_tokens=previous, cache_experiment=True)
    assert prompt[:4] == previous[:4]
    assert len(prompt) == 10
