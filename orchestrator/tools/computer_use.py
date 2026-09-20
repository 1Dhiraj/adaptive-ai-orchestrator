"""One capability for "do this on the screen", over several backends.

A step that needs to operate a real interface should not have to know whether
that means a browser or a desktop application. It declares the
``computer_use`` capability; this tool picks a backend:

    browser (Playwright MCP)  ->  desktop (Hermes)  ->  nothing

Browser first on purpose. Anything inside a website is cheaper, more
observable and easier to bound in a browser than by driving the whole
desktop, and a browser session can be restricted to an allow-list of domains
in a way that "control the computer" cannot.

Safety is not a layer on top of this tool -- it is the reason the tool exists
rather than letting agents call the raw backends:

* off unless ``ORCHESTRATOR_ALLOW_DESKTOP`` is set, and irreversible, so the
  approval gate always runs first and shows the plan before anything moves;
* the plan, the backend, and the sites or apps it will touch are shown in the
  preview a person approves, not discovered afterwards;
* an allow-list bounds where it may act -- an empty list means nothing is
  allowed, so forgetting to configure it fails closed;
* anything that looks like a credential is refused rather than typed;
* step and time limits, and every action appended to a run log.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import Tool, ToolError
from .workspace import DEFAULT_WORKSPACE_ROOT, workspace_for

__all__ = ["ComputerUseTool", "ComputerUsePlan", "check_secrets", "allowed_targets"]

#: Patterns that mean "this text contains a secret". Deliberately broad: a
#: false positive costs one refusal, a false negative types a password into a
#: web page and writes it to the action log.
_SECRET_PATTERNS = [
    (re.compile(r"\b(?:\d[ -]*?){13,19}\b"), "what looks like a card number"),
    (re.compile(r"(?i)\b(?:password|passwd|pwd|passphrase|otp|2fa|cvv|pin)\b"
                r"\s*[:=]\s*\S+"), "a password or one-time code"),
    (re.compile(r"(?i)\b(?:api[_-]?key|secret|token|bearer)\b\s*[:=]\s*\S+"), "an API key or token"),
    (re.compile(r"\b(?:sk|pk|ghp|gho|nvapi|xox[baprs])[-_][A-Za-z0-9]{12,}\b"), "a credential-shaped string"),
]

_DEFAULT_MAX_STEPS = 25
_DEFAULT_TIME_LIMIT_S = 300.0


def _enabled() -> bool:
    return os.environ.get("ORCHESTRATOR_ALLOW_DESKTOP", "").strip().lower() in {
        "1", "true", "yes", "on"}


def allowed_targets() -> List[str]:
    """Domains and application names this run may touch.

    Empty means nothing is allowed. Failing closed matters more here than
    convenience: an unset allow-list should not silently grant the whole
    desktop and the whole internet.
    """
    raw = os.environ.get("ORCHESTRATOR_COMPUTER_USE_ALLOW", "")
    return [item.strip().lower() for item in re.split(r"[,\s]+", raw) if item.strip()]


def check_secrets(text: str) -> Optional[str]:
    """Describe the secret this text appears to contain, or None."""
    for pattern, reason in _SECRET_PATTERNS:
        if pattern.search(text or ""):
            return reason
    return None


def _target_allowed(target: str, allow: List[str]) -> bool:
    """True when a domain or app name is covered by the allow-list.

    Domains match on suffix so ``example.com`` covers ``www.example.com``,
    but ``notexample.com`` is not covered by ``example.com``.
    """
    candidate = (target or "").strip().lower()
    if not candidate:
        return False
    for entry in allow:
        if candidate == entry or candidate.endswith("." + entry):
            return True
        if entry in candidate and "." not in entry:
            return True  # application name, e.g. "notepad"
    return False


@dataclass
class ComputerUsePlan:
    """What the agent intends to do, in the words a person will approve."""

    goal: str = ""
    backend: str = "none"
    targets: List[str] = field(default_factory=list)
    actions: List[str] = field(default_factory=list)
    max_steps: int = _DEFAULT_MAX_STEPS
    time_limit_s: float = _DEFAULT_TIME_LIMIT_S

    def describe(self) -> str:
        where = ", ".join(self.targets) or "(no target named)"
        lines = [f"goal   : {self.goal or '(none given)'}",
                 f"backend: {self.backend}",
                 f"touches: {where}",
                 f"limits : {self.max_steps} actions, {self.time_limit_s:.0f}s"]
        for i, action in enumerate(self.actions, 1):
            lines.append(f"  {i}. {action}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {"goal": self.goal, "backend": self.backend, "targets": self.targets,
                "actions": self.actions, "max_steps": self.max_steps,
                "time_limit_s": self.time_limit_s}


def _parse_plan(task: str) -> ComputerUsePlan:
    """Read the agent's TOOL_DIRECTIVE into a plan, tolerating a missing one."""
    match = re.search(r"TOOL_DIRECTIVE:\s*(\{.*\})\s*$", task or "", re.S | re.M)
    payload: Dict[str, Any] = {}
    if match:
        try:
            parsed = json.loads(match.group(1))
            payload = parsed.get("arguments", parsed) if isinstance(parsed, dict) else {}
        except ValueError:
            payload = {}
    goal = str(payload.get("goal") or (task or "").strip().splitlines()[:1][0] if task else "")
    targets = payload.get("targets") or payload.get("domains") or []
    if isinstance(targets, str):
        targets = [targets]
    actions = payload.get("actions") or []
    if isinstance(actions, str):
        actions = [actions]
    return ComputerUsePlan(
        goal=goal[:300],
        targets=[str(t).strip().lower() for t in targets if str(t).strip()],
        actions=[str(a).strip() for a in actions if str(a).strip()],
        max_steps=int(payload.get("max_steps") or _DEFAULT_MAX_STEPS),
        time_limit_s=float(payload.get("time_limit_s") or _DEFAULT_TIME_LIMIT_S),
    )


class ComputerUseTool(Tool):
    """Operate a browser or the desktop, under an approved, bounded plan."""

    name = "computer_use"
    capability = "computer_use"
    description = ("Carry out a task in a browser or desktop application, "
                   "within an approved plan and an allow-list of sites/apps")
    fallbacks: List[str] = []
    side_effect = True
    #: Clicking things on a real screen cannot be undone, so this always
    #: reaches the approval gate before it runs.
    irreversible = True

    def __init__(self, run_id: str = "default", tools: Any = None,
                 root: Optional[Path] = None) -> None:
        super().__init__()
        self.run_id = run_id
        #: The ToolManager, so browser backends registered from MCP can be found.
        self.tools = tools
        self.workspace = workspace_for(run_id, root or DEFAULT_WORKSPACE_ROOT)

    # -- backend selection -------------------------------------------------

    def _browser_backend(self) -> Optional[str]:
        """Name of a registered Playwright-style navigate tool, if any."""
        if self.tools is None:
            return None
        for candidate in ("web_browser_navigate", "browser_navigate"):
            tool = self.tools.get(candidate)
            if tool is not None and not tool.is_broken and tool.is_live():
                return candidate
        return None

    def _desktop_backend(self) -> Optional[str]:
        """Hermes first, then native control.

        Hermes is given a whole outcome and works out the steps itself, so
        when it is installed it is the better instrument. Native control is
        the fallback that always exists: it needs no second agent and no
        extra account, at the cost of the caller having to drive it click by
        click.
        """
        if self.tools is None:
            return None
        for candidate in ("hermes_desktop", "desktop_native"):
            tool = self.tools.get(candidate)
            if tool is not None and not tool.is_broken and tool.is_live():
                return candidate
        return None

    def backend(self) -> str:
        """Which backend would run this, browser preferred."""
        return self._browser_backend() or self._desktop_backend() or "none"

    def is_live(self) -> bool:
        return _enabled() and self.backend() != "none"

    # -- agent-facing ------------------------------------------------------

    def prompt_hint(self) -> str:
        allow = allowed_targets()
        where = ", ".join(allow) if allow else "(none configured -- nothing is allowed yet)"
        return (
            "Operate a browser or desktop app. Say what you intend to do first:\n"
            'TOOL_DIRECTIVE: {"arguments": {"goal": "...", '
            '"targets": ["example.com"], "actions": ["open the page", "read the heading"]}}\n'
            f"You may only touch: {where}\n"
            "Never include a password, card number, API key or one-time code -- "
            "ask the person to enter it themselves.")

    def preview(self, task: str, context: Optional[dict] = None) -> str:
        plan = _parse_plan(task)
        plan.backend = self.backend()
        mode = "REAL" if self.is_live() else "simulated"
        return f"computer_use [{mode}]\n{plan.describe()}"

    # -- execution ---------------------------------------------------------

    def _log(self, entry: Dict[str, Any]) -> None:
        """Append one action to the run's log. Never fatal."""
        try:
            self.workspace.mkdir(parents=True, exist_ok=True)
            path = self.workspace / "computer_use.log.jsonl"
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"at": round(time.time(), 3), **entry}) + "\n")
        except OSError:
            pass

    def _run(self, task: str, context: Optional[dict] = None) -> str:
        plan = _parse_plan(task)
        plan.backend = self.backend()

        secret = check_secrets(task)
        if secret:
            # Refuse before anything is typed *or* written to the log.
            raise ToolError(
                f"refused: the instructions contain {secret}. Ask the person to "
                "enter it themselves, or read it from the vault at use time.")

        allow = allowed_targets()
        if not plan.targets:
            raise ToolError(
                "refused: name the sites or apps this will touch in "
                '"targets", so they can be checked against the allow-list.')
        blocked = [t for t in plan.targets if not _target_allowed(t, allow)]
        if blocked:
            raise ToolError(
                f"refused: {', '.join(blocked)} is not in the allow-list. "
                "Set ORCHESTRATOR_COMPUTER_USE_ALLOW to the domains and apps "
                "this run may touch.")

        if not _enabled():
            return (f"[simulated:computer_use] would use {plan.backend} for "
                    f"{', '.join(plan.targets)}\n{plan.describe()}\n"
                    "  (set ORCHESTRATOR_ALLOW_DESKTOP=1 to act for real)")

        if plan.backend == "none":
            raise ToolError(
                "no computer-use backend available: attach a Playwright MCP "
                "server for browser work, or install the Hermes agent and set "
                "ORCHESTRATOR_ALLOW_DESKTOP=1 for desktop work.")

        self._log({"event": "start", "run_id": self.run_id, "plan": plan.to_dict()})
        started, performed = time.monotonic(), 0
        backend_tool = self.tools.get(plan.backend) if self.tools else None
        if backend_tool is None:
            raise ToolError(f"backend '{plan.backend}' disappeared before it could run")

        outputs: List[str] = []
        for action in plan.actions or [plan.goal]:
            if performed >= plan.max_steps:
                outputs.append(f"stopped: reached the {plan.max_steps}-action limit")
                break
            elapsed = time.monotonic() - started
            if elapsed >= plan.time_limit_s:
                outputs.append(f"stopped: reached the {plan.time_limit_s:.0f}s time limit")
                break
            try:
                result = backend_tool.execute(action)
            except Exception as exc:  # noqa: BLE001 - report, do not crash the run
                self._log({"event": "error", "action": action, "error": str(exc)[:500]})
                raise ToolError(f"{plan.backend} failed on '{action[:80]}': {exc}") from exc
            performed += 1
            self._log({"event": "action", "action": action, "result": str(result)[:500]})
            outputs.append(f"{performed}. {action} -> {str(result)[:300]}")

        self._log({"event": "finish", "actions": performed,
                   "seconds": round(time.monotonic() - started, 2)})
        body = "\n".join(outputs) or "(no actions performed)"
        return (f"[computer_use] {plan.backend}: {performed} action(s) on "
                f"{', '.join(plan.targets)}\n{body}")
