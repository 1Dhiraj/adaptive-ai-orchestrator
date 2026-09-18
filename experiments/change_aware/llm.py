"""Minimal Ollama client with a persistent response cache.

Every call is keyed by (model, messages, sampling options). The seed is part of
the key, so two configurations that send the same prompt in the same trial get
the same sampled output (common random numbers), while a later trial uses a
different seed and therefore a fresh sample. The cache also lets an interrupted
run resume without repeating model calls.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
import urllib.request

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
MODEL = os.environ.get("CA_MODEL", "qwen2.5:7b")


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
