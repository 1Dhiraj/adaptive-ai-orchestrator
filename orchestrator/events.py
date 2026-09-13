"""A tiny synchronous event bus.

Everything observable in the system -- step transitions, tool fallbacks,
approvals -- flows through here. The CLI printer, the SQLite recorder and the
dashboard's WebSocket feed are all just subscribers, so none of them are
wired into the orchestrator directly.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional


class EventType(str, Enum):
    RUN_STARTED = "run_started"
    RUN_FINISHED = "run_finished"
    RUN_CANCEL_REQUESTED = "run_cancel_requested"
    PLAN_CREATED = "plan_created"
    JOB_QUEUED = "job_queued"
    JOB_FINISHED = "job_finished"

    STEP_STARTED = "step_started"
    STEP_FINISHED = "step_finished"
    STEP_FAILED = "step_failed"
    STEP_SKIPPED = "step_skipped"
    STEP_RETRY = "step_retry"
    STEP_CANCELLED = "step_cancelled"
    AGENT_MESSAGE = "agent_message"

    STEP_AWAITING_APPROVAL = "step_awaiting_approval"
    STEP_APPROVED = "step_approved"
    STEP_AWAITING_INPUT = "step_awaiting_input"
    STEP_INPUT_PROVIDED = "step_input_provided"
    ACTION_AWAITING_APPROVAL = "action_awaiting_approval"
    ACTION_APPROVED = "action_approved"
    ACTION_REJECTED = "action_rejected"
    ACTION_EXECUTED = "action_executed"

    TOOL_CALLED = "tool_called"
    TOOL_FALLBACK = "tool_fallback"
    TOOL_BROKEN = "tool_broken"
    TOOL_REPAIRED = "tool_repaired"

    CHANGE_REQUESTED = "change_requested"
    IMPACT_ANALYSED = "impact_analysed"

    LOG = "log"


@dataclass
class Event:
    type: EventType
    run_id: str = ""
    step_id: Optional[str] = None
    message: str = ""
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type.value,
            "run_id": self.run_id,
            "step_id": self.step_id,
            "message": self.message,
            "data": self.data,
            "timestamp": self.timestamp,
        }


Subscriber = Callable[[Event], None]


class EventBus:
    """Thread-safe fan-out. A misbehaving subscriber can never break a run."""

    def __init__(self, keep_history: int = 2000):
        self._subscribers: List[Subscriber] = []
        self._lock = threading.RLock()
        self.history: List[Event] = []
        self.keep_history = keep_history

    def subscribe(self, fn: Subscriber) -> Subscriber:
        with self._lock:
            self._subscribers.append(fn)
        return fn

    def unsubscribe(self, fn: Subscriber) -> None:
        with self._lock:
            if fn in self._subscribers:
                self._subscribers.remove(fn)

    def emit(self, event: Event) -> Event:
        with self._lock:
            self.history.append(event)
            if len(self.history) > self.keep_history:
                del self.history[: len(self.history) - self.keep_history]
            subscribers = list(self._subscribers)

        for fn in subscribers:
            try:
                fn(event)
            except Exception as exc:  # noqa: BLE001 - observers must never break execution
                print(f"[events] subscriber {getattr(fn, '__name__', fn)} raised: {exc}")
        return event

    def publish(self, type: EventType, run_id: str = "", step_id: Optional[str] = None,
                message: str = "", **data: Any) -> Event:
        return self.emit(Event(type=type, run_id=run_id, step_id=step_id,
                               message=message, data=data))

    def clear_history(self) -> None:
        with self._lock:
            self.history.clear()


# ---------------------------------------------------------------------------
# Ready-made subscribers
# ---------------------------------------------------------------------------

_ICONS = {
    EventType.STEP_STARTED: "->",
    EventType.STEP_FINISHED: "ok",
    EventType.STEP_FAILED: "XX",
    EventType.STEP_SKIPPED: "..",
    EventType.STEP_RETRY: "~~",
    EventType.STEP_CANCELLED: "--",
    EventType.TOOL_FALLBACK: "!!",
    EventType.TOOL_BROKEN: "!!",
    EventType.TOOL_REPAIRED: "++",
    EventType.STEP_AWAITING_APPROVAL: "??",
    EventType.STEP_APPROVED: "ok",
    EventType.STEP_AWAITING_INPUT: "??",
    EventType.STEP_INPUT_PROVIDED: "ok",
    EventType.ACTION_AWAITING_APPROVAL: "!?",
    EventType.ACTION_APPROVED: "ok",
    EventType.ACTION_REJECTED: "no",
    EventType.ACTION_EXECUTED: "->",
    EventType.CHANGE_REQUESTED: "**",
    EventType.IMPACT_ANALYSED: "**",
    EventType.PLAN_CREATED: "**",
    EventType.LOG: "..",
}

#: Events that head a section rather than describing one step.
_BANNER = {EventType.RUN_STARTED, EventType.RUN_FINISHED}


def console_printer(verbose: bool = False) -> Subscriber:
    """Human-readable progress on stdout."""
    quiet_types = {EventType.LOG, EventType.TOOL_CALLED}

    def printer(event: Event) -> None:
        if not verbose and event.type in quiet_types:
            return
        if event.type in _BANNER:
            arrow = "===>" if event.type is EventType.RUN_STARTED else "<==="
            print(f"\n{arrow} {event.message}")
            return
        icon = _ICONS.get(event.type, "  ")
        where = f" {event.step_id}" if event.step_id else ""
        print(f"  [{icon}]{where}: {event.message}")

    return printer
