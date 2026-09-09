from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from awb.backends import LlamaCppBackend
from awb.real_replay import run_real_replay
from awb.tracing import TraceRecorder
from awb.types import AgentState, RequestPriority
from awb.world import World
from awb.backends import FakeBackend


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        return

    def _read_json(self):
        n = int(self.headers.get("Content-Length", "0"))
        return json.loads(self.rfile.read(n).decode("utf-8")) if n else {}

    def do_GET(self):
        if self.path == "/health":
            data = json.dumps({"status": "ok"}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        self.send_error(404)

    def do_POST(self):
        payload = self._read_json()
        if self.path == "/tokenize":
            # Stable nonempty tokenizer substitute for client-path tests.
            text = str(payload.get("content", ""))
            tokens = [100 + (i % 17) for i, _ in enumerate(text.split() or ["x"])]
            data = json.dumps({"tokens": tokens}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        if self.path == "/completion":
            prompt = payload.get("prompt", [])
            n_predict = int(payload.get("n_predict", 1))
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            first = {"content": "x", "tokens": [7], "stop": False}
            final = {
                "content": "",
                "tokens": [],
                "stop": True,
                "stop_type": "limit",
                "timings": {
                    "cache_n": 0,
                    "prompt_n": len(prompt),
                    "prompt_ms": 10.0,
                    "prompt_per_second": len(prompt) / 0.01,
                    "predicted_n": n_predict,
                    "predicted_ms": 20.0,
                    "predicted_per_second": n_predict / 0.02,
                },
            }
            self.wfile.write(("data: " + json.dumps(first) + "\n\n").encode())
            self.wfile.flush()
            self.wfile.write(("data: " + json.dumps(final) + "\n\n").encode())
            self.wfile.flush()
            return
        self.send_error(404)


def _make_trace(path: Path) -> Path:
    trace = TraceRecorder()
    world = World(seed=1, backend=FakeBackend(max_concurrency=1), trace=trace, enable_cognitive_loop=False)
    world.add_agent(AgentState(agent_id=1, location_id=1))
    for i in range(3):
        world.clock_ms = i * 10
        world.create_request(
            agent_id=1,
            task_type="probe_task",
            priority=RequestPriority.NORMAL,
            deadline_ms=1000,
            input_tokens=64 + i,
            output_tokens=4,
            cognitive_domain="general",
        )
    return trace.write_jsonl(path)


def test_real_replay_client_path_with_mock_llamacpp(tmp_path: Path):
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        trace_path = _make_trace(tmp_path / "trace.jsonl")
        backend = LlamaCppBackend(base_url=f"http://127.0.0.1:{server.server_port}", timeout_seconds=5)
        csv_path, report_path, report = run_real_replay(
            trace_path=trace_path,
            backend=backend,
            out_dir=tmp_path / "real",
            concurrency=2,
            max_requests=3,
            timing_mode="immediate",
        )
        assert report.completed == 3
        assert report.failed == 0
        assert report.prompt_tokens_total == 64 + 65 + 66
        assert csv_path.exists()
        assert report_path.exists()
    finally:
        server.shutdown()
        server.server_close()


def test_loopback_backend_bypasses_environment_proxy(monkeypatch):
    """A local llama.cpp server must not be sent through HTTP_PROXY."""
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:9")
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:9")
    monkeypatch.delenv("NO_PROXY", raising=False)
    monkeypatch.delenv("no_proxy", raising=False)

    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        backend = LlamaCppBackend(
            base_url=f"http://127.0.0.1:{server.server_port}",
            timeout_seconds=5,
        )
        assert backend.health()["status"] == "ok"
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()
