"""LLM provider abstraction with token/cost metering.

Two providers ship in the box:

``GeminiProvider``
    Real calls through ``google-genai``, with retry/backoff on transient
    errors and usage pulled from the response metadata.

``StubProvider``
    A deterministic, offline generator. Same prompt -> same output, always.
    This is what makes the test suite and the benchmark harness reproducible
    on a machine with no API key, and it still *varies with the prompt*, so
    hash-based change propagation behaves realistically.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional

from .config import settings
from .models import LLMUsage

# Approximate USD per 1M tokens. Override via Provider(pricing=...) if these
# drift -- they are only used for the cost column in reports.
PRICING: Dict[str, tuple] = {
    "gemini-2.5-flash": (0.30, 2.50),
    "gemini-2.5-flash-lite": (0.10, 0.40),
    "gemini-2.5-pro": (1.25, 10.00),
    "gemini-2.0-flash": (0.10, 0.40),
    "claude-opus-5": (15.00, 75.00),
    "claude-sonnet-5": (3.00, 15.00),
    "claude-haiku-4-5-20251001": (1.00, 5.00),
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    # NVIDIA NIM free tier is credit-based, not per-token billed.
    "meta/llama-3.1-8b-instruct": (0.0, 0.0),
    "nvidia/nemotron-3-super-120b-a12b": (0.0, 0.0),
    "meta/llama-3.3-70b-instruct": (0.0, 0.0),
    "stub": (0.0, 0.0),
}
_DEFAULT_PRICE = (0.30, 2.50)


@dataclass
class LLMResponse:
    text: str
    usage: LLMUsage
    model: str
    latency_s: float


class LLMError(RuntimeError):
    """Raised when a provider fails after exhausting its retries."""


def estimate_tokens(text: str) -> int:
    """Cheap token estimate (~4 characters per token) for the stub provider."""
    return max(1, len(text) // 4)


def price(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    in_rate, out_rate = PRICING.get(model, _DEFAULT_PRICE)
    return round((prompt_tokens * in_rate + completion_tokens * out_rate) / 1_000_000, 8)


class LLMProvider(ABC):
    """Common interface so the orchestrator never imports a vendor SDK."""

    name: str = "abstract"
    model: str = "unknown"

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.total_usage = LLMUsage()

    @abstractmethod
    def _generate(self, prompt: str, system: Optional[str], json_mode: bool,
                  metadata: Optional[Dict[str, str]]) -> LLMResponse:
        ...

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        json_mode: bool = False,
        metadata: Optional[Dict[str, str]] = None,
        images: Optional[List[bytes]] = None,
    ) -> LLMResponse:
        """Generate a completion.

        ``metadata`` carries orchestrator-side hints (currently just the
        caller's agent role). Real providers ignore it; the stub uses it so
        its output matches the role without guessing from the prompt text.

        ``images`` are raw PNG bytes, for looking at a screen rather than
        reading a description of one. A provider that cannot accept them
        raises rather than answering from the text alone -- a vision loop
        that silently went blind would invent coordinates and click them.
        """
        if images:
            response = self._generate_with_images(prompt, system, json_mode,
                                                  metadata, images)
        else:
            response = self._generate(prompt, system, json_mode, metadata)
        with self._lock:
            self.total_usage = self.total_usage + response.usage
        return response

    def _generate_with_images(self, prompt: str, system: Optional[str],
                              json_mode: bool, metadata: Optional[Dict[str, str]],
                              images: List[bytes]) -> LLMResponse:
        """Override in providers that accept images. Refuses by default."""
        raise LLMError(
            f"provider '{self.name}' ({getattr(self, 'model', '?')}) cannot accept "
            "images. Use a vision model -- e.g. LLM_PROVIDER=openai with "
            "OPENAI_BASE_URL=https://openrouter.ai/api/v1 and "
            "OPENAI_MODEL=minimax/minimax-m3.")

    def supports_images(self) -> bool:
        """Whether this provider can be given screenshots."""
        return type(self)._generate_with_images is not LLMProvider._generate_with_images

    def reset_usage(self) -> None:
        with self._lock:
            self.total_usage = LLMUsage()


# ---------------------------------------------------------------------------
# Gemini
# ---------------------------------------------------------------------------

#: Substrings that mark an exception as worth retrying.
_TRANSIENT_MARKERS = (
    "429", "500", "502", "503", "504",
    "rate limit", "resource_exhausted", "unavailable",
    "deadline", "timeout", "internal error", "overloaded",
)


def is_transient(exc: BaseException) -> bool:
    text = f"{type(exc).__name__} {exc}".lower()
    return any(marker in text for marker in _TRANSIENT_MARKERS)


class GeminiProvider(LLMProvider):
    name = "gemini"

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None,
                 max_retries: Optional[int] = None):
        super().__init__()
        from google import genai  # imported lazily so the stub path needs no SDK

        self.model = model or settings.gemini_model
        self.max_retries = settings.llm_max_retries if max_retries is None else max_retries
        self._client = genai.Client(api_key=api_key or settings.gemini_api_key)

    def _generate(self, prompt: str, system: Optional[str], json_mode: bool,
                  metadata: Optional[Dict[str, str]] = None) -> LLMResponse:
        config: Dict[str, object] = {}
        if system:
            config["system_instruction"] = system
        if json_mode:
            config["response_mime_type"] = "application/json"

        last_exc: Optional[BaseException] = None
        for attempt in range(self.max_retries + 1):
            started = time.time()
            try:
                response = self._client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=config or None,
                )
                latency = time.time() - started
                meta = getattr(response, "usage_metadata", None)
                prompt_tokens = getattr(meta, "prompt_token_count", None) or estimate_tokens(prompt)
                completion_tokens = getattr(meta, "candidates_token_count", None) or 0
                text = response.text or ""
                if not completion_tokens:
                    completion_tokens = estimate_tokens(text)
                return LLMResponse(
                    text=text,
                    usage=LLMUsage(
                        calls=1,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        cost_usd=price(self.model, prompt_tokens, completion_tokens),
                    ),
                    model=self.model,
                    latency_s=latency,
                )
            except Exception as exc:  # noqa: BLE001 - provider SDKs raise broadly
                last_exc = exc
                if attempt >= self.max_retries or not is_transient(exc):
                    break
                time.sleep(min(2 ** attempt, 30))

        raise LLMError(f"Gemini call failed after {self.max_retries + 1} attempt(s): {last_exc}") from last_exc


# ---------------------------------------------------------------------------
# Deterministic offline stub
# ---------------------------------------------------------------------------

_STOPWORDS = {
    "the", "a", "an", "and", "or", "for", "with", "that", "this", "your",
    "task", "produce", "output", "context", "from", "prerequisite", "tasks",
    "you", "are", "senior", "step", "using", "into", "then", "will", "should",
    "given", "based", "prior", "their", "have", "each", "must", "when",
}

#: Role -> the shape of a deliverable that role would actually hand over.
_ROLE_TEMPLATES: Dict[str, List[str]] = {
    "frontend": [
        "Built {n} React components: {items}.",
        "Routing wired with react-router; state via Zustand store `use{Cap}Store`.",
        "Accessibility pass done: keyboard nav, ARIA labels, focus traps on modals.",
    ],
    "backend": [
        "Exposed {n} REST endpoints: {endpoints}.",
        "Auth: JWT access tokens (15 min) + rotating refresh tokens.",
        "Validation with Pydantic models; errors returned as RFC 7807 problem details.",
    ],
    "database": [
        "Schema defined with {n} tables: {items}.",
        "Indexes added on foreign keys and lookup columns; FKs are ON DELETE CASCADE.",
        "Migration `0001_{slug}.sql` generated and applied.",
    ],
    "testing": [
        "Wrote {n} test cases covering {items}.",
        "Result: {n2} passed, 0 failed, coverage 87%.",
        "Added a regression test for the edge case found in {slug}.",
    ],
    "devops": [
        "Containerised with a multi-stage Dockerfile; image size {n2} MB.",
        "CI pipeline: lint -> test -> build -> deploy, gated on green tests.",
        "Secrets sourced from the environment, never baked into the image.",
    ],
    "security": [
        "Threat model reviewed: {items}.",
        "Findings: {n} medium, 0 critical. Rate limiting added on auth routes.",
        "All user input parameterised; no string-concatenated SQL remains.",
    ],
    "research": [
        "Reviewed {n} sources on {slug}.",
        "Key finding: {items}.",
        "Recommendation: proceed with the approach above, revisit after load testing.",
    ],
    "writer": [
        "Drafted documentation covering {items}.",
        "Includes a quick-start, an API reference, and {n} worked examples.",
    ],
    "reviewer": [
        "Reviewed the work from upstream steps; {n} issues raised.",
        "Blocking: none. Non-blocking: naming consistency, missing docstrings on {items}.",
    ],
    "planner": [
        "Broke the goal into {n} workstreams: {items}.",
    ],
}
_GENERIC_TEMPLATE = [
    "Completed: {slug}.",
    "Deliverables: {items}.",
    "No blockers; downstream steps can proceed.",
]


# ---------------------------------------------------------------------------
# Anthropic / OpenAI / Ollama -- plain HTTP, no vendor SDK required
# ---------------------------------------------------------------------------


def _post_json(url: str, headers: Dict[str, str], body: Dict[str, object],
               timeout: float, max_retries: int):
    """POST with the project's transient-error retry policy. Returns parsed JSON."""
    import requests  # lazy: only needed by providers that actually call out

    last_exc: Optional[BaseException] = None
    for attempt in range(max_retries + 1):
        try:
            response = requests.post(url, headers=headers, json=body, timeout=timeout)
        except requests.RequestException as exc:
            last_exc = exc
        else:
            if response.status_code < 400:
                return response.json()
            last_exc = RuntimeError(
                f"HTTP {response.status_code} from {url}: {response.text[:400]}")
            if not is_transient(last_exc):
                break
        if attempt < max_retries:
            time.sleep(min(2 ** attempt, 30))
    raise LLMError(f"request to {url} failed after {max_retries + 1} attempt(s): {last_exc}") \
        from last_exc


class AnthropicProvider(LLMProvider):
    """Claude via the Messages API (plain HTTP, no ``anthropic`` package needed)."""

    name = "anthropic"

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None,
                 max_retries: Optional[int] = None):
        super().__init__()
        self.api_key = api_key or settings.anthropic_api_key
        if not self.api_key:
            raise LLMError("ANTHROPIC_API_KEY is not set")
        self.model = model or settings.anthropic_model
        self.max_retries = settings.llm_max_retries if max_retries is None else max_retries

    def _generate(self, prompt: str, system: Optional[str], json_mode: bool,
                  metadata: Optional[Dict[str, str]] = None) -> LLMResponse:
        # Claude has no dedicated JSON mode; asking plainly in the system
        # prompt is the standard workaround and is reliable enough for the
        # structured plans this project asks for.
        if json_mode:
            system = f"{system or ''}\nRespond with ONLY valid JSON. No markdown fences.".strip()

        body: Dict[str, object] = {
            "model": self.model,
            "max_tokens": 4096,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            body["system"] = system

        started = time.time()
        data = _post_json(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            body=body, timeout=settings.llm_timeout_s, max_retries=self.max_retries,
        )
        text = "".join(block.get("text", "") for block in data.get("content", []))
        usage = data.get("usage", {})
        return LLMResponse(
            text=text,
            usage=LLMUsage(
                calls=1,
                prompt_tokens=usage.get("input_tokens", estimate_tokens(prompt)),
                completion_tokens=usage.get("output_tokens", estimate_tokens(text)),
                cost_usd=price(self.model, usage.get("input_tokens", 0),
                              usage.get("output_tokens", 0)),
            ),
            model=self.model, latency_s=time.time() - started,
        )


class OpenAIProvider(LLMProvider):
    """GPT via the Chat Completions API. Also fits any OpenAI-compatible
    endpoint (Azure OpenAI, OpenRouter, vLLM, ...) via ``base_url``."""

    name = "openai"

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None,
                 base_url: Optional[str] = None, max_retries: Optional[int] = None):
        super().__init__()
        self.api_key = api_key or settings.openai_api_key
        if not self.api_key:
            raise LLMError("OPENAI_API_KEY is not set")
        self.model = model or settings.openai_model
        self.base_url = (base_url or settings.openai_base_url).rstrip("/")
        self.max_retries = settings.llm_max_retries if max_retries is None else max_retries

    def _generate_with_images(self, prompt: str, system: Optional[str],
                              json_mode: bool, metadata: Optional[Dict[str, str]],
                              images: List[bytes]) -> LLMResponse:
        """Same endpoint, with the image parts OpenAI-compatible APIs expect.

        Written against the shared format rather than one vendor's, so the
        same path serves OpenAI, OpenRouter (minimax, qwen-vl, gemini) and any
        compatible gateway -- whichever model is configured.
        """
        import base64

        parts: List[Dict[str, object]] = [{"type": "text", "text": prompt}]
        for raw in images:
            encoded = base64.b64encode(raw).decode("ascii")
            parts.append({"type": "image_url",
                          "image_url": {"url": f"data:image/png;base64,{encoded}"}})
        return self._generate(prompt, system, json_mode, metadata, _content=parts)

    def _generate(self, prompt: str, system: Optional[str], json_mode: bool,
                  metadata: Optional[Dict[str, str]] = None,
                  _content: Optional[List[Dict[str, object]]] = None) -> LLMResponse:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": _content if _content else prompt})

        body: Dict[str, object] = {"model": self.model, "messages": messages}
        if json_mode:
            body["response_format"] = {"type": "json_object"}

        started = time.time()
        data = _post_json(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}",
                     "Content-Type": "application/json"},
            body=body, timeout=settings.llm_timeout_s, max_retries=self.max_retries,
        )
        text = data["choices"][0]["message"].get("content", "")
        usage = data.get("usage", {})
        return LLMResponse(
            text=text,
            usage=LLMUsage(
                calls=1,
                prompt_tokens=usage.get("prompt_tokens", estimate_tokens(prompt)),
                completion_tokens=usage.get("completion_tokens", estimate_tokens(text)),
                cost_usd=price(self.model, usage.get("prompt_tokens", 0),
                              usage.get("completion_tokens", 0)),
            ),
            model=self.model, latency_s=time.time() - started,
        )


class NvidiaProvider(OpenAIProvider):
    """NVIDIA NIM -- 100+ hosted open models (Llama, Mistral, Nemotron, Qwen).

    Their API is OpenAI-compatible, so this is a thin subclass rather than a
    reimplementation: only the base URL, key and default model differ. A
    generous free tier makes it a practical alternative when a metered
    provider runs out of quota.
    """

    name = "nvidia"

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None,
                 base_url: Optional[str] = None, max_retries: Optional[int] = None):
        # Bypass OpenAIProvider.__init__, which insists on OPENAI_API_KEY.
        LLMProvider.__init__(self)
        self.api_key = api_key or settings.nvidia_api_key
        if not self.api_key:
            raise LLMError("NVIDIA_API_KEY is not set")
        self.model = model or settings.nvidia_model
        self.base_url = (base_url or settings.nvidia_base_url).rstrip("/")
        self.max_retries = settings.llm_max_retries if max_retries is None else max_retries


class OllamaProvider(LLMProvider):
    """A local model via Ollama's REST API. No API key, free, offline.

    The natural fix for a rate-limited or unaffordable hosted provider:
    ``pip install`` nothing extra, install Ollama, ``ollama pull llama3.1``,
    then set ``LLM_PROVIDER=ollama``.
    """

    name = "ollama"

    def __init__(self, base_url: Optional[str] = None, model: Optional[str] = None,
                 max_retries: Optional[int] = None):
        super().__init__()
        self.base_url = (base_url or settings.ollama_base_url).rstrip("/")
        self.model = model or settings.ollama_model
        self.max_retries = settings.llm_max_retries if max_retries is None else max_retries

    def _generate(self, prompt: str, system: Optional[str], json_mode: bool,
                  metadata: Optional[Dict[str, str]] = None) -> LLMResponse:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        body: Dict[str, object] = {"model": self.model, "messages": messages, "stream": False}
        if json_mode:
            body["format"] = "json"

        started = time.time()
        try:
            data = _post_json(f"{self.base_url}/api/chat", headers={}, body=body,
                              timeout=settings.llm_timeout_s, max_retries=self.max_retries)
        except LLMError as exc:
            raise LLMError(
                f"{exc} (is Ollama running? try: ollama serve -- and 'ollama pull {self.model}')"
            ) from exc

        text = data.get("message", {}).get("content", "")
        return LLMResponse(
            text=text,
            usage=LLMUsage(
                calls=1,
                prompt_tokens=data.get("prompt_eval_count", estimate_tokens(prompt)),
                completion_tokens=data.get("eval_count", estimate_tokens(text)),
                cost_usd=0.0,  # local inference, no metered cost
            ),
            model=self.model, latency_s=time.time() - started,
        )


def _keywords(text: str, limit: int = 6) -> List[str]:
    words = re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", text.lower())
    seen: List[str] = []
    for w in words:
        if w in _STOPWORDS or w in seen:
            continue
        seen.append(w)
        if len(seen) >= limit:
            break
    return seen or ["work"]


class StubProvider(LLMProvider):
    """Deterministic pseudo-LLM: no network, no key, reproducible output.

    Output is derived from a hash of the prompt, so it is stable across runs
    but genuinely different when the prompt changes -- exactly the property
    the adaptive-re-execution logic needs in order to be tested honestly.
    """

    name = "stub"
    model = "stub"

    def __init__(self, latency_s: float = 0.0):
        super().__init__()
        #: Simulated per-call latency, so benchmarks can model API wall time.
        self.latency_s = latency_s

    def _generate(self, prompt: str, system: Optional[str], json_mode: bool,
                  metadata: Optional[Dict[str, str]] = None) -> LLMResponse:
        started = time.time()
        if self.latency_s:
            time.sleep(self.latency_s)

        seed = int(hashlib.sha256(f"{system}\x00{prompt}".encode()).hexdigest()[:16], 16)
        rng = random.Random(seed)

        if json_mode:
            text = self._plan_json(prompt, rng)
        else:
            text = self._prose(prompt, (metadata or {}).get("role", ""), rng)

        prompt_tokens = estimate_tokens(prompt + (system or ""))
        completion_tokens = estimate_tokens(text)
        return LLMResponse(
            text=text,
            usage=LLMUsage(
                calls=1,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                cost_usd=0.0,
            ),
            model=self.model,
            latency_s=time.time() - started,
        )

    # -- generators -------------------------------------------------------

    def _prose(self, prompt: str, role: str, rng: random.Random) -> str:
        # Custom roles have no canned template, but the label should still be
        # theirs -- otherwise a user-registered role looks like it was ignored.
        label = role or "generic"
        # Keywords come from the task line only; the context block below it
        # belongs to upstream steps and would otherwise dominate.
        task_text = prompt.split("## Context", 1)[0]
        kws = _keywords(task_text)
        fields = {
            "n": rng.randint(3, 9),
            "n2": rng.randint(20, 60),
            "items": ", ".join(kws[:3]),
            "endpoints": ", ".join(
                f"{rng.choice(['GET', 'POST', 'PUT', 'DELETE'])} /{k}" for k in kws[:3]
            ),
            "slug": kws[0],
            "Cap": kws[0].capitalize(),
        }
        template = _ROLE_TEMPLATES.get(role, _GENERIC_TEMPLATE)
        body = "\n".join(f"- {line.format(**fields)}" for line in template)
        # A real model's output always differs when its input differs. The
        # templates alone can collide on similar prompts, so a signature of
        # the exact prompt is folded in -- this keeps the stub deterministic
        # while making change propagation behave the way it would in practice.
        ref = f"{rng.getrandbits(32):08x}"
        result = f"[{label}] {body}\n- Artifact reference: {ref}."

        # The offline provider must follow the same tool contract as a live
        # model. This keeps demos executable without pretending the stub wrote
        # a full application: it leaves a concrete hand-off file for coding
        # steps and performs a harmless workspace check for terminal steps.
        tool_match = re.search(
            r"deliverable will be passed to the `([^`]+)` tool", prompt,
            re.IGNORECASE,
        )
        tool_name = tool_match.group(1).lower() if tool_match else ""
        step_match = re.search(r"## Task \(([^)]+)\)", prompt)
        step_id = re.sub(r"[^a-z0-9_-]+", "_",
                         (step_match.group(1) if step_match else role or "output").lower())
        if tool_name == "workspace":
            directive = {
                "action": "write",
                "path": f"deliverables/{step_id}.md",
                "content": result,
            }
            result += "\nTOOL_DIRECTIVE: " + json.dumps(directive)
        elif tool_name == "terminal":
            result += ("\nTOOL_DIRECTIVE: " + json.dumps({
                "command": "python -c \"print('workspace verification complete')\""
            }))
        return result

    def _plan_json(self, prompt: str, rng: random.Random) -> str:
        """Create a task-specific deterministic plan when no live model exists.

        This is deliberately modest pattern matching, not pretend intelligence.
        Its job is to keep the product useful and honest offline without showing
        a frontend/backend/database workflow for every unrelated request.
        """
        task_text = prompt.split("\n\n", 1)[0].removeprefix("Task:").strip()
        text = task_text.lower()
        topic = task_text.rstrip(".") or "the requested work"

        role_prompts = {
            "requirements analyst": "Clarify scope, constraints, supplied facts and acceptance criteria.",
            "research": "Find and organize relevant evidence without inventing facts.",
            "data analyst": "Validate data, calculate carefully and explain the result.",
            "writer": "Produce clear final content from verified upstream material.",
            "desktop operator": "Complete the requested outcome in desktop applications and report evidence.",
            "database": "Design a consistent data model with constraints and migrations.",
            "backend": "Implement working services and explicit interfaces.",
            "frontend": "Implement an accessible user interface connected to real behavior.",
            "testing": "Verify acceptance criteria and report concrete pass or fail evidence.",
            "workflow operator": "Execute the requested digital process carefully and report the outcome.",
        }

        def step(step_id: str, role: str, description: str, deps=None, tool=None):
            return {"id": step_id, "role": role, "description": description,
                    "depends_on": deps or [], "tool": tool}

        software = any(word in text for word in (
            "build an app", "build a website", "web app", "mobile app", "software",
            "api", "frontend", "backend", "code", "application"))
        desktop = any(word in text for word in (
            "microsoft word", "powerpoint", "excel", "desktop", "open the", ".docx", ".pptx"))
        collect_data = any(word in text for word in (
            "scrape", "weather", "collect data", "fetch data", "monitor", "extract"))
        deliver = any(word in text for word in ("email", "send mail", "notify", "send it"))
        research = any(word in text for word in (
            "research", "literature", "paper", "compare", "report", "analyse", "analyze"))

        if desktop:
            tasks = [
                step("prepare", "writer", f"Prepare the verified content and acceptance checklist for: {topic}."),
                step("desktop_work", "desktop operator",
                     f"Complete this outcome in the required desktop application: {topic}.",
                     ["prepare"], "hermes_desktop"),
                step("verify", "testing", "Verify the expected file or desktop result exists and is usable.",
                     ["desktop_work"]),
            ]
            shape = "desktop outcome"
        elif collect_data and deliver:
            tasks = [
                step("collect", "research", f"Collect the requested current data for: {topic}.", tool="rest_api"),
                step("validate", "data analyst", "Check the collected values, units, source and completeness.",
                     ["collect"]),
                step("summarise", "writer", "Turn the validated data into a concise human-readable update.",
                     ["validate"]),
                step("deliver", "workflow operator", f"Deliver the completed update as requested: {topic}.",
                     ["summarise"], "gmail"),
            ]
            shape = "data collection and delivery"
        elif collect_data:
            tasks = [
                step("collect", "research", f"Collect the requested data for: {topic}.", tool="rest_api"),
                step("validate", "data analyst", "Validate source, structure, units and missing values.", ["collect"]),
                step("present", "writer", "Present the validated result in the requested form.", ["validate"]),
            ]
            shape = "data collection"
        elif software:
            tasks = [
                step("requirements", "requirements analyst", f"Define scope and acceptance criteria for: {topic}."),
                step("database", "database", f"Create the database schema and migration files for: {topic}.",
                     ["requirements"], "workspace"),
                step("backend", "backend", f"Create the backend source files and business logic for: {topic}.",
                     ["database"], "workspace"),
                step("frontend", "frontend", f"Implement the user-facing experience for: {topic}.",
                     ["requirements"], "workspace"),
                step("testing", "testing", f"Run acceptance and integration tests for: {topic}.",
                     ["backend", "frontend"], "terminal"),
            ]
            if "github" in text:
                tasks.append(step(
                    "publish", "workflow operator",
                    f"Publish the completed project to GitHub as requested: {topic}.",
                    ["testing"], "github"))
            shape = "software delivery"
        elif research:
            tasks = [
                step("scope", "requirements analyst", f"Define the question and evidence criteria for: {topic}."),
                step("research", "research", f"Gather and compare relevant evidence for: {topic}.", ["scope"]),
                step("analyse", "data analyst", "Identify supported findings, limitations and open questions.",
                     ["research"]),
                step("write", "writer", "Produce the requested report using only supported findings.", ["analyse"]),
                step("verify", "testing", "Check citations, consistency and coverage against the request.", ["write"]),
            ]
            shape = "research and reporting"
        else:
            tasks = [
                step("understand", "requirements analyst", f"Define the desired outcome for: {topic}."),
                step("execute", "workflow operator", f"Complete the digital work required for: {topic}.", ["understand"]),
                step("verify", "testing", "Check the result against the requested outcome.", ["execute"]),
            ]
            shape = "general digital task"

        used_roles = list(dict.fromkeys(task["role"] for task in tasks))
        agents = [{
            "role": role,
            "why": f"responsible for the {shape} workflow",
            "system_prompt": f"You are the {role}. {role_prompts[role]}",
            "handles": [task["id"] for task in tasks if task["role"] == role],
        } for role in used_roles]

        return json.dumps({
            "agents": agents,
            "requirements": [],  # TaskPlanner infers these from actual tool use.
            "tasks": tasks,
            "notes": [f"Offline deterministic planner selected a {shape} workflow. Review before running."],
        })


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_default_provider: Optional[LLMProvider] = None
_factory_lock = threading.Lock()

_PROVIDER_CLASSES: Dict[str, type] = {
    "gemini": GeminiProvider,
    "anthropic": AnthropicProvider,
    "openai": OpenAIProvider,
    "nvidia": NvidiaProvider,
    "ollama": OllamaProvider,
}


def build_provider(name: str) -> LLMProvider:
    """Construct one named provider directly, bypassing auto-detection.

    Raises :class:`LLMError` if that provider's prerequisites (an API key,
    mainly) are not met -- callers that want graceful degradation should
    catch it, as :func:`get_provider` does.
    """
    if name == "stub":
        return StubProvider()
    cls = _PROVIDER_CLASSES.get(name)
    if cls is None:
        raise LLMError(f"unknown provider '{name}'; choose from: "
                       f"{', '.join(sorted(_PROVIDER_CLASSES) + ['stub'])}")
    return cls()


def get_provider(force_stub: bool = False) -> LLMProvider:
    """Return the process-wide provider, building it on first use.

    Selection follows :meth:`Settings.resolve_provider`: an explicit
    ``LLM_PROVIDER``, else NVIDIA when its key is present, else the offline
    stub. A provider that fails to
    construct (bad key, missing package) degrades to the stub rather than
    crashing the whole run.
    """
    global _default_provider
    with _factory_lock:
        if _default_provider is None:
            name = "stub" if force_stub else settings.resolve_provider()
            try:
                _default_provider = build_provider(name)
            except Exception as exc:  # noqa: BLE001 - degrade instead of crashing
                print(f"[llm] '{name}' unavailable ({exc}); falling back to stub provider.")
                _default_provider = StubProvider()
        return _default_provider


def provider_for_role(provider: LLMProvider, role: str) -> LLMProvider:
    """Role-specific NVIDIA model without changing explicit/offline providers."""
    if not isinstance(provider, NvidiaProvider):
        return provider
    env_role = re.sub(r"[^A-Z0-9]+", "_", role.upper())
    model = os.environ.get(f"NVIDIA_MODEL_{env_role}", "").strip()
    if not model or model == provider.model:
        return provider
    with provider._lock:
        cache = getattr(provider, "_role_providers", {})
        if model not in cache:
            cache[model] = NvidiaProvider(api_key=provider.api_key, model=model,
                                         base_url=provider.base_url, max_retries=provider.max_retries)
        provider._role_providers = cache
        return cache[model]


def set_provider(provider: Optional[LLMProvider]) -> None:
    """Inject a provider (tests, benchmarks). ``None`` resets the factory."""
    global _default_provider
    with _factory_lock:
        _default_provider = provider
