from __future__ import annotations

import json
import time
import urllib.request
from urllib.parse import urlparse
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class LlamaCppBackend:
    """Thin client for a llama.cpp server.

    v0.3 uses llama.cpp's native `/completion` endpoint for real workload
    replay because it accepts raw token-id prompts and returns server timings.
    This lets AWB reproduce trace input lengths exactly instead of guessing from
    text length.
    """

    base_url: str = "http://127.0.0.1:8080"
    model: str = "local-model"
    max_concurrency: int = 1
    timeout_seconds: float = 120.0
    _token_cache: dict[str, list[int]] = field(default_factory=dict, init=False, repr=False)
    _opener: urllib.request.OpenerDirector = field(init=False, repr=False)

    def __post_init__(self) -> None:
        # urllib honors HTTP_PROXY/HTTPS_PROXY (and on Windows, system proxy
        # settings).  Loopback llama.cpp traffic must never leave the host: a
        # configured proxy can otherwise turn a healthy 127.0.0.1 server into
        # an HTTP 502.  Preserve normal proxy behavior for non-loopback URLs.
        host = (urlparse(self.base_url).hostname or "").lower()
        if host in {"127.0.0.1", "localhost", "::1"}:
            self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        else:
            self._opener = urllib.request.build_opener()

    def _url(self, path: str) -> str:
        return f"{self.base_url.rstrip('/')}/{path.lstrip('/')}"

    def _post_json(self, path: str, payload: dict[str, Any], *, timeout: float | None = None) -> dict:
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self._url(path),
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with self._opener.open(req, timeout=timeout or self.timeout_seconds) as response:
            return json.loads(response.read().decode("utf-8"))

    def health(self) -> dict:
        with self._opener.open(self._url("/health"), timeout=min(10.0, self.timeout_seconds)) as response:
            return json.loads(response.read().decode("utf-8"))

    def tokenize(self, text: str) -> list[int]:
        if text in self._token_cache:
            return list(self._token_cache[text])
        payload = self._post_json(
            "/tokenize",
            {"content": text, "add_special": False, "parse_special": True},
        )
        tokens = [int(x) for x in payload.get("tokens", [])]
        if not tokens:
            raise RuntimeError("llama.cpp /tokenize returned no tokens")
        self._token_cache[text] = tokens
        return list(tokens)

    def make_exact_prompt_tokens(
        self,
        record: dict,
        *,
        previous_prompt_tokens: list[int] | None = None,
        cache_experiment: bool = False,
    ) -> list[int]:
        target = max(1, int(record.get("input_tokens", 1)))
        semantic = (
            f"AWB agent={record.get('agent_id')} task={record.get('task_type')} "
            f"domain={record.get('cognitive_domain')} structured state observation belief intent plan. "
        )
        semantic_tokens = self.tokenize(semantic)
        filler_tokens = self.tokenize(
            " persistent world state observation memory belief intent plan context update "
        )

        prefix: list[int] = []
        if cache_experiment and previous_prompt_tokens:
            requested_prefix = int(record.get("carried_state_tokens", 0))
            prefix_len = min(target, requested_prefix, len(previous_prompt_tokens))
            prefix = list(previous_prompt_tokens[:prefix_len])

        if not prefix:
            prefix = semantic_tokens[:target]

        out = list(prefix)
        remaining = target - len(out)
        if remaining > 0:
            source = filler_tokens or semantic_tokens
            repeats, tail = divmod(remaining, len(source))
            out.extend(source * repeats)
            out.extend(source[:tail])
        return out[:target]

    def stream_completion(
        self,
        *,
        prompt_tokens: list[int],
        n_predict: int,
        cache_prompt: bool,
        id_slot: int = -1,
    ) -> dict[str, Any]:
        payload = {
            "prompt": prompt_tokens,
            "n_predict": max(1, int(n_predict)),
            "temperature": 0.0,
            "ignore_eos": True,
            "seed": 1,
            "stream": True,
            "return_tokens": True,
            "cache_prompt": bool(cache_prompt),
            "id_slot": int(id_slot),
        }
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self._url("/completion"),
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        wall_start = time.perf_counter()
        first_token_at: float | None = None
        final_payload: dict[str, Any] = {}
        observed_output_tokens = 0

        with self._opener.open(req, timeout=self.timeout_seconds) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if not data or data == "[DONE]":
                    continue
                try:
                    event = json.loads(data)
                except json.JSONDecodeError:
                    continue

                tokens = event.get("tokens") or []
                content = event.get("content")
                if first_token_at is None and (tokens or content not in (None, "")):
                    first_token_at = time.perf_counter()
                if tokens:
                    observed_output_tokens += len(tokens)
                elif content not in (None, "") and not event.get("stop"):
                    observed_output_tokens += 1

                if event.get("stop") or "timings" in event:
                    final_payload = event

        wall_end = time.perf_counter()
        timings = final_payload.get("timings", {}) if isinstance(final_payload, dict) else {}
        usage = final_payload.get("usage", {}) if isinstance(final_payload, dict) else {}

        prompt_n = int(timings.get("prompt_n", 0) or usage.get("prompt_tokens", 0) or 0)
        cache_n = int(timings.get("cache_n", 0) or 0)
        predicted_n = int(
            timings.get("predicted_n", 0)
            or usage.get("completion_tokens", 0)
            or observed_output_tokens
            or 0
        )
        prompt_ms = float(timings.get("prompt_ms", 0.0) or 0.0)
        predicted_ms = float(timings.get("predicted_ms", 0.0) or 0.0)
        ttft_ms = (
            (first_token_at - wall_start) * 1000.0
            if first_token_at is not None
            else (wall_end - wall_start) * 1000.0
        )
        return {
            "wall_ms": (wall_end - wall_start) * 1000.0,
            "ttft_ms": ttft_ms,
            "prompt_n": prompt_n,
            "cache_n": cache_n,
            "predicted_n": predicted_n,
            "prompt_ms": prompt_ms,
            "predicted_ms": predicted_ms,
            "prompt_per_second": float(timings.get("prompt_per_second", 0.0) or 0.0),
            "predicted_per_second": float(timings.get("predicted_per_second", 0.0) or 0.0),
            "stop_type": final_payload.get("stop_type") if isinstance(final_payload, dict) else None,
        }


    def stream_chat_completion(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 16,
    ) -> dict[str, Any]:
        """Run a real semantic decision through llama.cpp's OpenAI chat API.

        Unlike trace replay, this sends the actual game observation text so the
        model itself must choose an action.  Wall/TTFT are measured client-side
        and therefore include real hardware/runtime behavior.
        """

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.0,
            "max_tokens": max(1, int(max_tokens)),
            "stream": True,
            "seed": 1,
        }
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self._url("/v1/chat/completions"),
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        wall_start = time.perf_counter()
        first_token_at: float | None = None
        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        finish_reason: str | None = None
        usage: dict[str, Any] = {}

        with self._opener.open(req, timeout=self.timeout_seconds) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if not data or data == "[DONE]":
                    continue
                try:
                    event = json.loads(data)
                except json.JSONDecodeError:
                    continue

                if isinstance(event.get("usage"), dict):
                    usage = event["usage"]
                choices = event.get("choices") or []
                if not choices:
                    continue
                choice = choices[0] or {}
                delta = choice.get("delta") or {}
                content = delta.get("content")
                reasoning = (
                    delta.get("reasoning_content")
                    or delta.get("reasoning")
                    or delta.get("analysis")
                )
                if content not in (None, ""):
                    if first_token_at is None:
                        first_token_at = time.perf_counter()
                    content_parts.append(str(content))
                if reasoning not in (None, ""):
                    if first_token_at is None:
                        first_token_at = time.perf_counter()
                    reasoning_parts.append(str(reasoning))
                if choice.get("finish_reason") is not None:
                    finish_reason = str(choice["finish_reason"])

        wall_end = time.perf_counter()
        if first_token_at is None:
            first_token_at = wall_end

        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
        return {
            "text": "".join(content_parts),
            "reasoning_text": "".join(reasoning_parts),
            "wall_ms": (wall_end - wall_start) * 1000.0,
            "ttft_ms": (first_token_at - wall_start) * 1000.0,
            "prompt_tokens": int(prompt_tokens) if prompt_tokens is not None else None,
            "completion_tokens": int(completion_tokens) if completion_tokens is not None else None,
            "finish_reason": finish_reason,
        }

    def probe(self, record: dict | None = None) -> dict[str, Any]:
        record = record or {
            "agent_id": 1,
            "task_type": "threat_assessment",
            "cognitive_domain": "threat",
            "input_tokens": 256,
            "expected_output_tokens": 16,
            "carried_state_tokens": 0,
        }
        prompt_tokens = self.make_exact_prompt_tokens(record)
        return self.stream_completion(
            prompt_tokens=prompt_tokens,
            n_predict=int(record.get("expected_output_tokens", 16)),
            cache_prompt=False,
        )
