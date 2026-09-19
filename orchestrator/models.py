"""Core data types shared by every layer of the orchestrator."""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional

from .inputs import InputRequest


class StepStatus(str, Enum):
    """Lifecycle of a single workflow step."""

    PENDING = "pending"      # never run
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    STALE = "stale"          # ran before, but an input changed -> must re-run
    SKIPPED = "skipped"      # reused from cache during an adaptive re-run
    PAUSED = "paused"        # human-in-the-loop gate, waiting for approval
    CANCELLED = "cancelled"  # an upstream step failed, so this never ran
    AWAITING_INPUT = "awaiting_input"    # needs a value only a human can give
    AWAITING_ACTION = "awaiting_action"  # about to do something irreversible
    NOT_APPLICABLE = "not_applicable"    # a structured branch condition was false


#: Statuses that mean "this step is waiting on a human, not on the machine".
BLOCKED_ON_HUMAN = {
    StepStatus.PAUSED, StepStatus.AWAITING_INPUT, StepStatus.AWAITING_ACTION,
}


#: Statuses that mean "there is a usable output cached for this step".
USABLE = {StepStatus.DONE, StepStatus.SKIPPED, StepStatus.NOT_APPLICABLE}


class ErrorClass(str, Enum):
    """How the orchestrator should react to a failure."""

    RETRIABLE = "retriable"  # transient: network blip, rate limit, timeout
    FATAL = "fatal"          # deterministic: bad prompt, missing dependency


@dataclass
class Step:
    """One unit of work, executed by one agent, possibly using one tool."""

    id: str
    description: str
    agent_role: str
    name: str = ""
    requires_tool: Optional[str] = None
    depends_on: List[str] = field(default_factory=list)

    # Reliability knobs
    max_retries: int = 2
    retry_backoff_s: float = 1.0
    timeout_s: Optional[float] = None

    # Human-in-the-loop: orchestrator stops before running this step until
    # Workflow.approve(step_id) is called.
    requires_approval: bool = False

    #: Values only a human can supply -- a recipient address, which repo to
    #: push to, a deadline. The run pauses here until they are answered.
    inputs: List["InputRequest"] = field(default_factory=list)

    metadata: Dict[str, Any] = field(default_factory=dict)

    #: Optional safe branch rule, for example
    #: {"source":"review", "operator":"contains", "value":"failed"}.
    #: ``source":"trigger.status"`` reads incoming webhook data.
    condition: Optional[Dict[str, Any]] = None

    output_type: str = "text"
    accepts: List[str] = field(default_factory=lambda: ["code", "json", "text"])
    assumes: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not isinstance(self.output_type, str):
            raise ValueError("output_type must be a string")
        for name in ("accepts", "assumes"):
            values = getattr(self, name)
            if not isinstance(values, list) or any(not isinstance(v, str) for v in values):
                raise ValueError(f"{name} must be a list of strings")
        if not self.name:
            self.name = self.id.replace("_", " ").title()
        # A step depending on itself is always a bug; drop it rather than
        # letting it deadlock the scheduler.
        self.depends_on = [d for d in self.depends_on if d != self.id]
        # Accept plain dicts so Step(**json) works.
        self.inputs = [
            i if isinstance(i, InputRequest) else InputRequest.from_dict(i)
            for i in self.inputs
        ]
        if self.condition is not None:
            if not isinstance(self.condition, dict):
                raise ValueError("step condition must be an object")
            operator = str(self.condition.get("operator", "truthy"))
            allowed = {"contains", "not_contains", "equals", "not_equals", "exists", "truthy"}
            if not self.condition.get("source") or operator not in allowed:
                raise ValueError("invalid step condition")

    # -- human input -------------------------------------------------------

    @property
    def pending_inputs(self) -> List["InputRequest"]:
        return [i for i in self.inputs if not i.satisfied]

    def input_values(self) -> Dict[str, Any]:
        from .inputs import values_of

        return values_of(self.inputs)

    def provide_input(self, name: str, value: Any) -> Any:
        for request in self.inputs:
            if request.name == name:
                return request.provide(value)
        raise KeyError(f"step '{self.id}' has no input named '{name}'")

    def definition_hash(self) -> str:
        """Signature of everything about this step that affects its output.

        Excludes scheduling knobs (retries, timeouts) because changing those
        does not change what the step produces. **Includes** the answers to
        this step's human inputs -- change the recipient address and the step
        must re-run, exactly as if its description had changed.
        """
        payload = json.dumps(
            {
                "id": self.id,
                "description": self.description,
                "agent_role": self.agent_role,
                "requires_tool": self.requires_tool,
                "depends_on": sorted(self.depends_on),
                "inputs": {k: str(v) for k, v in sorted(self.input_values().items())},
                "condition": self.condition,
                "output_type": self.output_type,
                "accepts": sorted(self.accepts),
                "assumes": sorted(self.assumes),
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def to_dict(self, redact_secrets: bool = True) -> Dict[str, Any]:
        data = asdict(self)
        # asdict leaves the InputType enums intact, which is not JSON-safe.
        data["inputs"] = [i.to_dict(redact_secrets) for i in self.inputs]
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Step":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class LLMUsage:
    """Token/cost accounting for one or more LLM calls."""

    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def __add__(self, other: "LLMUsage") -> "LLMUsage":
        return LLMUsage(
            calls=self.calls + other.calls,
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            cost_usd=round(self.cost_usd + other.cost_usd, 8),
        )

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["total_tokens"] = self.total_tokens
        return d


@dataclass
class StepResult:
    """Everything we know about one execution of one step."""

    step_id: str
    status: StepStatus = StepStatus.PENDING
    output: str = ""
    error: Optional[str] = None
    error_class: Optional[ErrorClass] = None

    started_at: Optional[float] = None
    ended_at: Optional[float] = None
    attempts: int = 0

    tool_requested: Optional[str] = None
    tool_used: Optional[str] = None
    used_fallback: bool = False

    usage: LLMUsage = field(default_factory=LLMUsage)

    #: Signature of (step definition + upstream outputs). If this is
    #: unchanged since the last successful run, the cached output is still
    #: valid -- this is what makes re-execution adaptive rather than blind.
    input_hash: str = ""
    #: Signature of the produced output, used to stop change propagation
    #: when a re-run produces a byte-identical result.
    output_hash: str = ""
    raw_output: str = ""
    declared_assumptions: List[str] = field(default_factory=list)
    detected_assumptions: List[str] = field(default_factory=list)
    assumed_facts: Dict[str, str] = field(default_factory=dict)
    change_status: str = ""
    invalidation_reasons: List[str] = field(default_factory=list)

    @property
    def duration_s(self) -> float:
        if self.started_at is None:
            return 0.0
        return (self.ended_at or time.time()) - self.started_at

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_id": self.step_id,
            "status": self.status.value,
            "output": self.output,
            "error": self.error,
            "error_class": self.error_class.value if self.error_class else None,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "duration_s": round(self.duration_s, 4),
            "attempts": self.attempts,
            "tool_requested": self.tool_requested,
            "tool_used": self.tool_used,
            "used_fallback": self.used_fallback,
            "usage": self.usage.to_dict(),
            "input_hash": self.input_hash,
            "output_hash": self.output_hash,
            "raw_output": self.raw_output,
            "declared_assumptions": self.declared_assumptions,
            "detected_assumptions": self.detected_assumptions,
            "assumed_facts": self.assumed_facts,
            "change_status": self.change_status,
            "invalidation_reasons": self.invalidation_reasons,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StepResult":
        usage = data.get("usage") or {}
        return cls(
            step_id=data["step_id"],
            status=StepStatus(data.get("status", "pending")),
            output=data.get("output", ""),
            error=data.get("error"),
            error_class=ErrorClass(data["error_class"]) if data.get("error_class") else None,
            started_at=data.get("started_at"),
            ended_at=data.get("ended_at"),
            attempts=data.get("attempts", 0),
            tool_requested=data.get("tool_requested"),
            tool_used=data.get("tool_used"),
            used_fallback=data.get("used_fallback", False),
            usage=LLMUsage(
                calls=usage.get("calls", 0),
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
                cost_usd=usage.get("cost_usd", 0.0),
            ),
            input_hash=data.get("input_hash", ""),
            output_hash=data.get("output_hash", ""),
            raw_output=data.get("raw_output", data.get("output", "")),
            declared_assumptions=list(data.get("declared_assumptions", [])),
            detected_assumptions=list(data.get("detected_assumptions", [])),
            assumed_facts=dict(data.get("assumed_facts", {})),
            change_status=data.get("change_status", ""),
            invalidation_reasons=list(data.get("invalidation_reasons", [])),
        )


@dataclass
class PendingAction:
    """A real-world action held back, waiting for a human to approve it.

    The agent has already produced its output; only the side-effecting tool
    call is outstanding. Keeping the payload here means approving costs
    nothing extra -- the LLM is not asked to regenerate the work.
    """

    step_id: str
    tool: str
    payload: str
    #: Short human-readable summary of what is about to happen.
    preview: str = ""
    irreversible: bool = True
    #: The agent's own output and usage, so the step can be completed as if
    #: it had never paused.
    agent_output: str = ""
    usage: LLMUsage = field(default_factory=LLMUsage)
    created_at: float = field(default_factory=time.time)

    def to_dict(self, include_step_id: bool = True) -> Dict[str, Any]:
        """Serialise. ``include_step_id=False`` when spreading into an event,
        whose envelope already carries the step id."""
        data: Dict[str, Any] = {
            "tool": self.tool,
            "preview": self.preview,
            "irreversible": self.irreversible,
            "payload_preview": self.payload[:600],
            "created_at": self.created_at,
        }
        if include_step_id:
            data["step_id"] = self.step_id
        return data

    def to_storage_dict(self) -> Dict[str, Any]:
        """Full durable form used only by the local state database.

        API and event payloads intentionally expose only ``payload_preview``;
        crash recovery needs the exact payload and prior model usage so an
        approved action can continue without generating it again.
        """
        return {
            "step_id": self.step_id,
            "tool": self.tool,
            "payload": self.payload,
            "preview": self.preview,
            "irreversible": self.irreversible,
            "agent_output": self.agent_output,
            "usage": self.usage.to_dict(),
            "created_at": self.created_at,
        }

    @classmethod
    def from_storage_dict(cls, data: Dict[str, Any]) -> "PendingAction":
        usage = data.get("usage") or {}
        return cls(
            step_id=str(data["step_id"]),
            tool=str(data["tool"]),
            payload=str(data.get("payload", "")),
            preview=str(data.get("preview", "")),
            irreversible=bool(data.get("irreversible", True)),
            agent_output=str(data.get("agent_output", "")),
            usage=LLMUsage(
                calls=int(usage.get("calls", 0)),
                prompt_tokens=int(usage.get("prompt_tokens", 0)),
                completion_tokens=int(usage.get("completion_tokens", 0)),
                cost_usd=float(usage.get("cost_usd", 0.0)),
            ),
            created_at=float(data.get("created_at", time.time())),
        )


def content_hash(*parts: str) -> str:
    """Stable short hash over an ordered list of strings."""
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode("utf-8", errors="replace"))
        h.update(b"\x00")
    return h.hexdigest()[:16]


def new_run_id() -> str:
    return uuid.uuid4().hex[:12]
