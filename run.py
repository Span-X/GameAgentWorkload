from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from awb import __version__
from awb.backends import FakeBackend, LlamaCppBackend, LlamaCppSemanticDecisionBackend
from awb.metrics import build_report
from awb.real_replay import run_real_replay
from awb.replay import request_digest_from_trace
from awb.semantic_decision import run_semantic_case, write_semantic_report
from awb.tracing import TraceRecorder, read_trace
from awb.world import World
from scenarios import SCENARIOS
from scenarios.bedroom_1 import core_cases


def run_scenario(
    name: str,
    seed: int,
    out_dir: Path,
    concurrency: int,
    *,
    cognitive_loop: bool = True,
    print_report: bool = True,
):
    scenario = SCENARIOS[name]
    trace = TraceRecorder()
    backend = FakeBackend(max_concurrency=concurrency)
    world = World(
        seed=seed,
        backend=backend,
        trace=trace,
        enable_cognitive_loop=cognitive_loop,
    )
    scenario.build(world)
    world.activation_snapshot()
    world.run(scenario.handle)

    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "" if cognitive_loop else ".event_only"
    trace_path = trace.write_jsonl(out_dir / f"{name}{suffix}.trace.jsonl")
    digest = trace.request_stream_digest()
    report = build_report(name, len(world.agents), trace.records, digest)
    report_path = report.write_json(out_dir / f"{name}{suffix}.report.json")
    if print_report:
        print(report.render_text())
        print(f"Trace: {trace_path}")
        print(f"Report: {report_path}")
    return trace_path, report_path, report


def run_sweep(name: str, seed: int, out_dir: Path, values: list[int], cognitive_loop: bool) -> Path:
    rows = []
    for concurrency in values:
        _, _, report = run_scenario(
            name,
            seed,
            out_dir / f"c{concurrency}",
            concurrency,
            cognitive_loop=cognitive_loop,
            print_report=False,
        )
        rows.append(
            {
                "concurrency": concurrency,
                "requests": report.requests,
                "completed": report.completed,
                "deadline_miss_rate": report.deadline_miss_rate,
                "queue_ms_p95": report.queue_ms_p95,
                "decision_ms_p95": report.decision_ms_p95,
                "superseded_before_start": report.superseded_before_start,
                "interrupted_inflight": report.interrupted_inflight,
                "carried_state_tokens": report.carried_state_tokens,
                "simulated_cache_reusable_tokens": report.simulated_cache_reusable_tokens,
                "belief_updates": report.belief_updates,
                "intent_transitions": report.intent_transitions,
                "plan_transitions": report.plan_transitions,
                "trace_digest": report.trace_digest,
            }
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.sweep.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"GameAgentWorkload {__version__} Sweep")
    print(f"Scenario: {name}")
    print("Concurrency | Miss% | Queue p95 | Decision p95 | Carried state | Sim cache")
    for r in rows:
        print(
            f"{r['concurrency']:>11} | "
            f"{r['deadline_miss_rate'] * 100:>5.1f} | "
            f"{r['queue_ms_p95']:>9.1f} | "
            f"{r['decision_ms_p95']:>12.1f} | "
            f"{r['carried_state_tokens']:>13} | "
            f"{r['simulated_cache_reusable_tokens']:>9}"
        )
    print(f"CSV: {path}")
    return path



def run_bedroom_suite(seed: int, out_dir: Path, concurrency: int) -> Path:
    names = sorted(name for name in SCENARIOS if name.startswith("bedroom_1_"))
    rows = []
    for name in names:
        trace_path, _, report = run_scenario(
            name,
            seed,
            out_dir,
            concurrency,
            cognitive_loop=True,
            print_report=False,
        )
        records = read_trace(trace_path)
        result = next(r for r in records if r.get("event") == "bedroom_case_result")
        rows.append(
            {
                "scenario": name,
                "case": result["case"],
                "passed": result["passed"],
                "interrupt": result["interrupt"],
                "cognitive_mode": result["cognitive_mode"],
                "brain_calls": result["brain_calls"],
                "action": result["action"],
                "action_roles": result["action_roles"],
                "sleep_completed": result["sleep_completed"],
                "survived": result["survived"],
                "social_engagement": result["social_engagement"],
                "requests": report.requests,
                "deadline_misses": report.deadline_misses,
                "decision_ms_p95": report.decision_ms_p95,
                "trace_digest": report.trace_digest,
            }
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "bedroom_1.suite.json"
    path.write_text(json.dumps({"version": __version__, "cases": rows}, indent=2), encoding="utf-8")
    print(f"GameAgentWorkload {__version__} Bedroom-1 suite")
    for row in rows:
        print(
            f"{row['case']:<16} pass={str(row['passed']):<5} "
            f"mode={row['cognitive_mode']:<10} brain_calls={row['brain_calls']} "
            f"p95={row['decision_ms_p95']:.1f}ms"
        )
    print(f"Suite JSON: {path}")
    return path



def run_bedroom_semantic_real(
    *,
    base_url: str,
    model: str,
    out_dir: Path,
    case_names: list[str],
    runs: int,
    max_tokens: int,
    timeout: float,
) -> tuple[Path, Path]:
    client = LlamaCppBackend(
        base_url=base_url,
        model=model,
        max_concurrency=1,
        timeout_seconds=timeout,
    )
    health = client.health()
    backend = LlamaCppSemanticDecisionBackend(client)
    cases = core_cases()

    unknown = [name for name in case_names if name not in cases]
    if unknown:
        raise ValueError(f"Unknown Bedroom-1 case(s): {', '.join(unknown)}")

    results = []
    prompts: dict[str, str] = {}
    for run_index in range(1, runs + 1):
        for name in case_names:
            result, prompt = run_semantic_case(
                cases[name],
                backend,
                run_index=run_index,
                max_tokens=max_tokens,
            )
            results.append(result)
            if prompt is not None:
                prompts[name] = prompt
            print(
                f"{name:<16} run={run_index:<2} mode={result.oracle_mode:<10} "
                f"action={str(result.action):<24} semantic={str(result.semantic_pass):<5} "
                f"realtime={str(result.realtime_pass):<5} "
                f"wall={result.wall_ms:.1f}ms ttft={result.ttft_ms:.1f}ms"
            )

    json_path, csv_path = write_semantic_report(
        results,
        prompts,
        out_dir=out_dir,
        model=model,
        backend_name=f"llama.cpp@{base_url}",
        health=health,
    )
    print(f"Semantic JSON: {json_path}")
    print(f"Semantic CSV:  {csv_path}")
    return json_path, csv_path

def main() -> None:
    parser = argparse.ArgumentParser(description=f"GameAgentWorkload {__version__} research benchmark harness")
    sub = parser.add_subparsers(dest="command")

    run_p = sub.add_parser("run", help="run a deterministic synthetic benchmark scenario")
    run_p.add_argument("--scenario", default="saloon_64", choices=sorted(SCENARIOS))
    run_p.add_argument("--seed", type=int, default=7)
    run_p.add_argument("--concurrency", type=int, default=4)
    run_p.add_argument("--out", type=Path, default=Path("traces"))
    run_p.add_argument("--no-cognitive-loop", action="store_true")

    sweep_p = sub.add_parser("sweep", help="run a deterministic synthetic concurrency sweep")
    sweep_p.add_argument("--scenario", default="saloon_64", choices=sorted(SCENARIOS))
    sweep_p.add_argument("--seed", type=int, default=7)
    sweep_p.add_argument("--concurrency", default="1,2,3,4,8,16,24,32")
    sweep_p.add_argument("--out", type=Path, default=Path("traces/sweep"))
    sweep_p.add_argument("--no-cognitive-loop", action="store_true")

    bedroom_p = sub.add_parser("bedroom-suite", help="run all Bedroom-1 core cases")
    bedroom_p.add_argument("--seed", type=int, default=7)
    bedroom_p.add_argument("--concurrency", type=int, default=1)
    bedroom_p.add_argument("--out", type=Path, default=Path("traces/bedroom_1"))

    semantic_p = sub.add_parser(
        "bedroom-real",
        help="run Bedroom-1 semantic action choices against a real llama.cpp model",
    )
    semantic_p.add_argument("--base-url", default="http://127.0.0.1:8080")
    semantic_p.add_argument("--model", default="local-model")
    semantic_p.add_argument(
        "--cases",
        default="roach,projectile,complex_visitor",
        help="comma-separated Bedroom-1 case names; use normal to include the no-model control",
    )
    semantic_p.add_argument("--runs", type=int, default=1)
    semantic_p.add_argument("--max-tokens", type=int, default=16)
    semantic_p.add_argument("--timeout", type=float, default=120.0)
    semantic_p.add_argument("--out", type=Path, default=Path("traces/bedroom_1_real"))

    replay_p = sub.add_parser("replay", help="inspect the canonical request-stream digest")
    replay_p.add_argument("trace", type=Path)

    probe_p = sub.add_parser("llamacpp-probe", help="probe a real llama.cpp server with an exact-size token prompt")
    probe_p.add_argument("--base-url", default="http://127.0.0.1:8080")
    probe_p.add_argument("--model", default="local-model")
    probe_p.add_argument("--input-tokens", type=int, default=256)
    probe_p.add_argument("--output-tokens", type=int, default=16)
    probe_p.add_argument("--timeout", type=float, default=120.0)

    real_p = sub.add_parser("replay-real", help="replay recorded inference requests against a real llama.cpp server")
    real_p.add_argument("trace", type=Path)
    real_p.add_argument("--base-url", default="http://127.0.0.1:8080")
    real_p.add_argument("--model", default="local-model")
    real_p.add_argument("--concurrency", type=int, default=1)
    real_p.add_argument("--max-requests", type=int, default=None)
    real_p.add_argument("--task-types", default="")
    real_p.add_argument("--cognitive-layers", default="")
    real_p.add_argument("--timing-mode", choices=["trace", "immediate"], default="trace")
    real_p.add_argument("--arrival-scale", type=float, default=1.0)
    real_p.add_argument("--cache-prompt", action="store_true")
    real_p.add_argument("--cache-experiment", action="store_true")
    real_p.add_argument("--server-slots", type=int, default=None)
    real_p.add_argument("--timeout", type=float, default=120.0)
    real_p.add_argument("--out", type=Path, default=Path("traces/real_replay"))

    args = parser.parse_args()
    if args.command in (None, "run"):
        if args.command is None:
            args.scenario = "saloon_64"
            args.seed = 7
            args.concurrency = 4
            args.out = Path("traces")
            args.no_cognitive_loop = False
        run_scenario(
            args.scenario,
            args.seed,
            args.out,
            args.concurrency,
            cognitive_loop=not args.no_cognitive_loop,
        )
    elif args.command == "sweep":
        values = [int(x.strip()) for x in args.concurrency.split(",") if x.strip()]
        run_sweep(
            args.scenario,
            args.seed,
            args.out,
            values,
            cognitive_loop=not args.no_cognitive_loop,
        )
    elif args.command == "bedroom-suite":
        run_bedroom_suite(args.seed, args.out, args.concurrency)
    elif args.command == "bedroom-real":
        case_names = [x.strip() for x in args.cases.split(",") if x.strip()]
        if not case_names:
            raise SystemExit("--cases must contain at least one Bedroom-1 case")
        if args.runs < 1:
            raise SystemExit("--runs must be >= 1")
        if args.max_tokens < 1:
            raise SystemExit("--max-tokens must be >= 1")
        run_bedroom_semantic_real(
            base_url=args.base_url,
            model=args.model,
            out_dir=args.out,
            case_names=case_names,
            runs=args.runs,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
        )
    elif args.command == "replay":
        print(request_digest_from_trace(args.trace))
    elif args.command == "llamacpp-probe":
        backend = LlamaCppBackend(
            base_url=args.base_url,
            model=args.model,
            timeout_seconds=args.timeout,
        )
        record = {
            "agent_id": 1,
            "task_type": "probe",
            "cognitive_domain": "general",
            "input_tokens": args.input_tokens,
            "expected_output_tokens": args.output_tokens,
            "carried_state_tokens": 0,
        }
        print(json.dumps(backend.probe(record), indent=2))
    elif args.command == "replay-real":
        task_types = {x.strip() for x in args.task_types.split(",") if x.strip()} or None
        cognitive_layers = {x.strip() for x in args.cognitive_layers.split(",") if x.strip()} or None
        backend = LlamaCppBackend(
            base_url=args.base_url,
            model=args.model,
            max_concurrency=args.concurrency,
            timeout_seconds=args.timeout,
        )
        csv_path, report_path, report = run_real_replay(
            trace_path=args.trace,
            backend=backend,
            out_dir=args.out,
            concurrency=args.concurrency,
            max_requests=args.max_requests,
            task_types=task_types,
            cognitive_layers=cognitive_layers,
            timing_mode=args.timing_mode,
            arrival_scale=args.arrival_scale,
            cache_prompt=args.cache_prompt,
            cache_experiment=args.cache_experiment,
            server_slots=args.server_slots,
        )
        print(report.render_text())
        print(f"Requests CSV: {csv_path}")
        print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
