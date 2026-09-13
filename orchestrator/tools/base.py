"""Layer 3 -- the tool ecosystem: tools, capabilities and fallback routing."""

from __future__ import annotations

import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional


class ToolError(RuntimeError):
    """A tool tried to run and failed."""


class ToolUnavailableError(ToolError):
    """The requested tool and every fallback for it are unusable."""

    def __init__(self, tool: str, attempted: List[str], reasons: Dict[str, str]):
        self.tool = tool
        self.attempted = attempted
        self.reasons = reasons
        detail = "; ".join(f"{name}: {why}" for name, why in reasons.items()) or "no candidates"
        super().__init__(f"'{tool}' and all fallbacks are unavailable ({detail})")


@dataclass
class ToolInvocation:
    """The record of one ``ToolManager.use`` call, including failed attempts."""

    requested: str
    tool_used: Optional[str] = None
    used_fallback: bool = False
    ok: bool = False
    output: str = ""
    simulated: bool = True
    attempts: List[str] = field(default_factory=list)
    errors: Dict[str, str] = field(default_factory=dict)
    duration_s: float = 0.0

    def to_dict(self) -> dict:
        return {
            "requested": self.requested,
            "tool_used": self.tool_used,
            "used_fallback": self.used_fallback,
            "ok": self.ok,
            "simulated": self.simulated,
            "attempts": self.attempts,
            "errors": self.errors,
            "duration_s": round(self.duration_s, 4),
        }


class Tool(ABC):
    """One external capability the agents can reach for.

    A tool is *live* when it has the credentials it needs, and *simulated*
    otherwise. Simulated tools still exercise the whole routing path -- which
    is the point: you can demo fallback behaviour with no accounts set up.
    """

    #: Unique registry key, e.g. ``"github"``.
    name: str = "tool"
    #: Tools sharing a capability are interchangeable, so any of them can act
    #: as a fallback for any other, e.g. ``"vcs"`` for github / github_cli.
    capability: str = "generic"
    description: str = ""
    #: Preferred fallbacks, in order, tried before other same-capability tools.
    fallbacks: List[str] = []

    #: Does calling this change something outside this process?
    side_effect: bool = False
    #: Can that change be undone easily? An email cannot be unsent; a GitHub
    #: issue can be closed. Irreversible actions are what get gated behind
    #: human approval.
    irreversible: bool = False

    def __init__(self) -> None:
        self.is_broken = False
        self.call_count = 0

    # -- failure injection (for the tool-failure scenario / chaos testing) --

    def break_it(self) -> None:
        self.is_broken = True

    def repair(self) -> None:
        self.is_broken = False

    # -- availability ------------------------------------------------------

    def is_live(self) -> bool:
        """True when real credentials are configured. Override in subclasses."""
        return False

    def unavailable_reason(self) -> Optional[str]:
        if self.is_broken:
            return "marked broken"
        return None

    # -- prompting ---------------------------------------------------------

    def prompt_hint(self) -> str:
        """What an agent needs to know to call this tool correctly.

        Injected into the agent's prompt. Tools with a machine-readable
        argument schema (notably MCP tools) should override this to describe
        their parameters -- otherwise the model has no way to know what a
        valid call looks like.
        """
        return self.description or f"Tool '{self.name}'."

    def is_irreversible(self, task: str, context: Optional[dict] = None) -> bool:
        """Whether *this particular* call cannot be undone.

        Defaults to the class-level flag. Tools whose risk depends on the
        payload override it -- an HTTP connection is harmless for GET and
        unundoable for DELETE, and gating both identically would be wrong in
        one direction or the other.
        """
        return self.irreversible

    def preview(self, task: str, context: Optional[dict] = None) -> str:
        """One line describing what calling this *would* do, without doing it.

        Shown to a human before an irreversible action is approved, so the
        decision is made on what will actually happen rather than on the
        tool's name. Override for anything whose effect is not obvious.
        """
        first_line = (task.strip().splitlines() or ["(empty)"])[0][:120]
        mode = "REAL" if self.is_live() else "simulated"
        return f"{self.name} [{mode}]: {first_line}"

    # -- execution ---------------------------------------------------------

    @abstractmethod
    def _run(self, task: str, context: Optional[dict] = None) -> str:
        """Do the work. Raise :class:`ToolError` on failure."""

    def execute(self, task: str, context: Optional[dict] = None) -> str:
        reason = self.unavailable_reason()
        if reason:
            raise ToolError(f"{self.name} unavailable: {reason}")
        self.call_count += 1
        return self._run(task, context)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        state = "broken" if self.is_broken else ("live" if self.is_live() else "simulated")
        return f"<Tool {self.name} [{self.capability}] {state}>"


class SimulatedTool(Tool):
    """A tool with no real backend, used when credentials are absent.

    Kept deliberately honest: the output is prefixed so nobody mistakes a
    simulated result for a real one in a report.
    """

    def __init__(self, name: str, capability: str, description: str = "",
                 fallbacks: Optional[List[str]] = None):
        super().__init__()
        self.name = name
        self.capability = capability
        self.description = description or f"Simulated {name}"
        self.fallbacks = list(fallbacks or [])

    def _run(self, task: str, context: Optional[dict] = None) -> str:
        summary = task.strip().splitlines()[0][:160] if task.strip() else "(no payload)"
        return f"[simulated:{self.name}] {summary}"


class ToolManager:
    """Registry + fallback router.

    ``use()`` tries the requested tool, then its declared fallbacks, then any
    other registered tool with the same capability. It never silently gives
    up: if nothing works it raises :class:`ToolUnavailableError` listing every
    candidate it tried and why each one failed.
    """

    def __init__(self, tools: Optional[List[Tool]] = None):
        self._tools: Dict[str, Tool] = {}
        self._lock = threading.RLock()
        self.history: List[ToolInvocation] = []
        for tool in tools or []:
            self.register(tool)

    # -- registry ---------------------------------------------------------

    def register(self, tool: Tool) -> "ToolManager":
        with self._lock:
            self._tools[tool.name] = tool
        return self

    def unregister(self, name: str) -> None:
        with self._lock:
            self._tools.pop(name, None)

    def get(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    @property
    def names(self) -> List[str]:
        return sorted(self._tools)

    # -- failure injection -------------------------------------------------

    def break_tool(self, name: str) -> None:
        tool = self._tools.get(name)
        if tool is None:
            raise KeyError(f"no such tool: '{name}'")
        tool.break_it()

    def repair_tool(self, name: str) -> None:
        tool = self._tools.get(name)
        if tool is None:
            raise KeyError(f"no such tool: '{name}'")
        tool.repair()

    def broken_tools(self) -> List[str]:
        return sorted(n for n, t in self._tools.items() if t.is_broken)

    # -- routing ----------------------------------------------------------

    def candidates_for(self, name: str) -> List[str]:
        """Ordered candidate list: the tool, its fallbacks, then same-capability peers."""
        primary = self._tools.get(name)
        ordered: List[str] = [name]
        if primary is not None:
            for fb in primary.fallbacks:
                if fb in self._tools and fb not in ordered:
                    ordered.append(fb)
            for other_name, other in sorted(self._tools.items()):
                if other_name in ordered:
                    continue
                if other.capability == primary.capability:
                    ordered.append(other_name)
        return ordered

    def use(self, name: str, task: str, context: Optional[dict] = None) -> ToolInvocation:
        started = time.time()
        invocation = ToolInvocation(requested=name)

        for candidate_name in self.candidates_for(name):
            tool = self._tools.get(candidate_name)
            if tool is None:
                invocation.errors[candidate_name] = "not registered"
                continue

            invocation.attempts.append(candidate_name)
            try:
                output = tool.execute(task, context)
            except Exception as exc:  # noqa: BLE001 - any tool failure is routable
                invocation.errors[candidate_name] = str(exc)
                continue

            invocation.ok = True
            invocation.tool_used = candidate_name
            invocation.used_fallback = candidate_name != name
            invocation.output = output
            invocation.simulated = not tool.is_live()
            invocation.duration_s = time.time() - started
            with self._lock:
                self.history.append(invocation)
            return invocation

        invocation.duration_s = time.time() - started
        with self._lock:
            self.history.append(invocation)
        raise ToolUnavailableError(name, invocation.attempts, invocation.errors)

    # -- reporting ---------------------------------------------------------

    def stats(self) -> dict:
        total = len(self.history)
        succeeded = sum(1 for i in self.history if i.ok)
        with_fallback = sum(1 for i in self.history if i.used_fallback)
        return {
            "invocations": total,
            "succeeded": succeeded,
            "failed": total - succeeded,
            "fallbacks_used": with_fallback,
            "success_rate": round(succeeded / total, 4) if total else None,
            "broken": self.broken_tools(),
        }

    def describe(self) -> List[dict]:
        return [
            {
                "name": t.name,
                "capability": t.capability,
                "description": t.description,
                "live": t.is_live(),
                "broken": t.is_broken,
                "fallbacks": self.candidates_for(t.name)[1:],
                "calls": t.call_count,
                "side_effect": t.side_effect,
                "irreversible": t.irreversible,
            }
            for t in sorted(self._tools.values(), key=lambda x: x.name)
        ]
