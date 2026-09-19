"""Minimal model client with a persistent response cache.

Every call is keyed by (model, messages, sampling options). The seed is part of
the key, so two configurations that send the same prompt in the same trial get
the same sampled output (common random numbers), while a later trial uses a
different seed and therefore a fresh sample. The cache also lets an interrupted
run resume without repeating model calls.

Two backends, chosen with ``CA_PROVIDER``:

``ollama`` (default)
    Local, free, and what the published results in ``results/`` were measured
    with. Its request payload and cache key are deliberately untouched, so
    that cache stays valid and those numbers stay reproducible.

``nvidia``
    NVIDIA NIM's OpenAI-compatible endpoint. Token counts come from the
    response ``usage`` field rather than being estimated, and calls are
    rate-limited and backed off, because the free tier will otherwise return
    429 halfway through a long run and lose the batch.

Write NVIDIA runs to a separate results directory and label the model: the
two are not comparable, and overwriting the qwen numbers would destroy the
only measured baseline.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
import urllib.error
import urllib.request

PROVIDER = os.environ.get("CA_PROVIDER", "ollama").strip().lower()

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
NVIDIA_URL = os.environ.get("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")

_DEFAULT_MODEL = {"ollama": "qwen2.5:7b",
                  "nvidia": "nvidia/nemotron-3-super-120b-a12b"}
MODEL = os.environ.get("CA_MODEL", _DEFAULT_MODEL.get(PROVIDER, "qwen2.5:7b"))

#: Free-tier friendly defaults. Raise them only if your account allows it.
MAX_RPM = int(os.environ.get("CA_MAX_RPM", "40"))
MAX_CONCURRENCY = int(os.environ.get("CA_CONCURRENCY", "2"))


class _RateLimiter:
    """Spaces calls to stay under a requests-per-minute cap.

    A cap is not the same as a retry: backing off after a 429 still wastes the
    call and the latency. Spacing requests means most runs never see one.
    """

    def __init__(self, max_rpm: int, max_concurrent: int) -> None:
        self._interval = 60.0 / max_rpm if max_rpm > 0 else 0.0
        self._slots = threading.Semaphore(max(1, max_concurrent))
        self._lock = threading.Lock()
        self._next_at = 0.0

    def __enter__(self):
        self._slots.acquire()
        if self._interval:
            with self._lock:
                wait = max(0.0, self._next_at - time.monotonic())
                self._next_at = max(self._next_at, time.monotonic()) + self._interval
            if wait:
                time.sleep(wait)
        return self

    def __exit__(self, *exc):
        self._slots.release()
        return False


_limiter = _RateLimiter(MAX_RPM, MAX_CONCURRENCY)


def stable_seed(*parts: str) -> int:
    digest = hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % (2 ** 31)


class LLM:
    def __init__(self, cache_path: str):
        self.cache_path = cache_path
        self.cache: dict[str, dict] = {}
        self.lock = threading.Lock()
        if os.path.exists(cache_path):
            with open(cache_path, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        rec = json.loads(line)
                        self.cache[rec["key"]] = rec

    def chat(self, system: str, user: str, *, temperature: float, seed: int, num_predict: int) -> dict:
        if PROVIDER == "nvidia":
            return self._chat_nvidia(system, user, temperature=temperature,
                                     seed=seed, num_predict=num_predict)
        payload = {
            "model": MODEL,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "stream": False,
            "options": {"temperature": temperature, "seed": seed, "num_predict": num_predict, "num_ctx": 8192},
        }
        key = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
        with self.lock:
            if key in self.cache:
                return self.cache[key]
        for attempt in range(5):
            try:
                req = urllib.request.Request(
                    f"{OLLAMA_URL}/api/chat",
                    data=json.dumps(payload).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                )
                started = time.time()
                with urllib.request.urlopen(req, timeout=600) as resp:
                    body = json.loads(resp.read().decode("utf-8"))
                break
            except Exception:  # transient server errors: back off and retry
                if attempt == 4:
                    raise
                time.sleep(3 * (attempt + 1))
        rec = {
            "key": key,
            "text": body["message"]["content"],
            "prompt_tokens": int(body.get("prompt_eval_count", 0)),
            "completion_tokens": int(body.get("eval_count", 0)),
            "seconds": body.get("total_duration", (time.time() - started) * 1e9) / 1e9,
        }
        with self.lock:
            self.cache[key] = rec
            with open(self.cache_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec) + "\n")
        return rec

    # -- NVIDIA NIM (OpenAI-compatible) ------------------------------------

    def _chat_nvidia(self, system: str, user: str, *, temperature: float,
                     seed: int, num_predict: int) -> dict:
        api_key = os.environ.get("NVIDIA_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError(
                "CA_PROVIDER=nvidia needs NVIDIA_API_KEY. Put it in .env, "
                "never in code or a commit.")

        payload = {
            "model": MODEL,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "temperature": temperature,
            "max_tokens": num_predict,
            "seed": seed,
            "stream": False,
        }
        # Namespaced so NVIDIA and Ollama results cannot collide in one cache
        # file, and so the existing qwen keys keep resolving unchanged.
        key = hashlib.sha256(
            ("nvidia\x1f" + json.dumps(payload, sort_keys=True)).encode("utf-8")
        ).hexdigest()
        with self.lock:
            if key in self.cache:
                return self.cache[key]

        body, started = None, time.time()
        for attempt in range(6):
            try:
                with _limiter:
                    request = urllib.request.Request(
                        f"{NVIDIA_URL}/chat/completions",
                        data=json.dumps(payload).encode("utf-8"),
                        headers={"Content-Type": "application/json",
                                 "Authorization": f"Bearer {api_key}",
                                 "Accept": "application/json"},
                    )
                    started = time.time()
                    with urllib.request.urlopen(request, timeout=600) as response:
                        body = json.loads(response.read().decode("utf-8"))
                break
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", "replace")[:300]
                if exc.code in (401, 403):
                    raise RuntimeError(
                        f"NVIDIA rejected the key ({exc.code}). Check NVIDIA_API_KEY "
                        f"is current and has credit. {detail}") from exc
                if exc.code == 402 or "credit" in detail.lower() or "quota" in detail.lower():
                    raise RuntimeError(
                        f"NVIDIA credits appear to be exhausted ({exc.code}): {detail}"
                    ) from exc
                if exc.code != 429 and exc.code < 500:
                    raise RuntimeError(f"NVIDIA error {exc.code}: {detail}") from exc
                if attempt == 5:
                    raise RuntimeError(
                        f"NVIDIA still returning {exc.code} after 6 attempts: {detail}"
                    ) from exc
                time.sleep(min(60.0, 2.0 * (2 ** attempt)))  # exponential backoff
            except Exception:
                if attempt == 5:
                    raise
                time.sleep(min(60.0, 2.0 * (2 ** attempt)))

        choice = (body.get("choices") or [{}])[0]
        usage = body.get("usage") or {}
        rec = {
            "key": key,
            "text": (choice.get("message") or {}).get("content", ""),
            # Measured, not estimated: the whole point of reading `usage`.
            "prompt_tokens": int(usage.get("prompt_tokens", 0)),
            "completion_tokens": int(usage.get("completion_tokens", 0)),
            "seconds": time.time() - started,
            "provider": "nvidia",
            "model": MODEL,
            # Whether the seed was honoured is not reported by the API, so
            # record that we asked rather than claiming it was respected.
            "seed_requested": seed,
            "seed_honoured": None,
        }
        with self.lock:
            self.cache[key] = rec
            with open(self.cache_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec) + "\n")
        return rec
