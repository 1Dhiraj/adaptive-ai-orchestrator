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
from urllib.parse import quote_plus, urlparse

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


def _browser_url(target: str, actions: List[str], goal: str = "") -> str:
    """Compile an approved browser target and common search instruction.

    ``browser_navigate`` accepts a URL, while the computer-use plan deliberately
    contains human-readable actions. Passing strings such as ``open the page``
    through as URLs produces convincing but invalid requests. Search boxes are
    equivalent to a GET URL, so compile that common case without needing brittle
    element selectors.
    """
    base = target if re.match(r"^https?://", target, re.I) else f"https://{target}"
    parsed = urlparse(base)
    if not parsed.hostname or any(ch.isspace() for ch in base):
        raise ToolError(f"refused: '{target}' is not a valid browser target")

    approved_host = parsed.hostname.lower()
    query = None
    for action in actions:
        match = re.search(r"(?i)\b(?:enter\s+)?(?:search\s+)?query\b[^'\"]*['\"]([^'\"]+)['\"]", action)
        if match:
            query = match.group(1).strip()
            break
    if not query:
        goal_match = re.search(
            r"(?i)\bsearch(?:\s+(?:google(?:\.com)?|the web))?\s+for\s+(.+?)(?:;|$)",
            goal,
        )
        if goal_match:
            query = goal_match.group(1).strip(" .")
    # A plan commonly says both "open Google" and then "enter query". The
    # search URL is the final intended state and must win over the earlier,
    # explicit Google home-page URL.
    if query and parsed.hostname.lower().removeprefix("www.") == "google.com":
        return f"{parsed.scheme}://{parsed.netloc}/search?q={quote_plus(query)}"

    for action in actions:
        match = re.search(r"https?://[^\s'\"<>]+", action, re.I)
        if not match:
            continue
        candidate = match.group(0).rstrip(".,);]")
        candidate_host = (urlparse(candidate).hostname or "").lower()
        if candidate_host == approved_host or candidate_host.endswith("." + approved_host):
            return candidate

    return base


def _snapshot_ref(snapshot: str, label: str) -> Optional[str]:
    """Find the accessibility ref for a plainly named control."""
    wanted = re.sub(r"\s+", " ", label).strip(" '\".").lower()
    if not wanted:
        return None
    exact, partial = [], []
    for line in str(snapshot).splitlines():
        match = re.search(r"\[ref=([^\]\s]+)", line)
        if not match:
            continue
        normal = re.sub(r"\s+", " ", line).lower()
        if f'"{wanted}"' in normal or f"'{wanted}'" in normal:
            exact.append(match.group(1))
        elif wanted in normal:
            partial.append(match.group(1))
    found = exact or partial
    return found[0] if len(found) == 1 else None


def _browser_interaction(action: str, snapshot: str) -> Optional[tuple[List[str], dict]]:
    """Compile common human browser instructions into typed MCP arguments."""
    text = action.strip()
    # Navigation, search and read/extract instructions are handled by the URL
    # compiler and snapshots, not replayed as bogus navigate calls.
    if re.search(r"(?i)^\s*(?:open|navigate|go to|read|extract|collect|summarize|search\b)", text):
        return None
    if re.search(r"(?i)\bquery\b", text):
        return None
    if re.fullmatch(r"(?i)\s*submit\s*", text):
        return ["web_browser_press_key", "browser_press_key"], {"key": "Enter"}

    typed = re.search(
        r"(?i)^\s*(?:type|enter|fill)\s+(['\"])(.*?)\1\s+(?:into|in)\s+(?:the\s+)?(.+?)\s*$",
        text,
    )
    if typed:
        value, label = typed.group(2), typed.group(3).strip(" '\".")
        ref = _snapshot_ref(snapshot, label)
        if not ref:
            raise ToolError(f"could not uniquely find {label!r} in the browser snapshot")
        return ["web_browser_type", "browser_type"], {
            "element": label, "ref": ref, "text": value,
        }

    clicked = re.search(r"(?i)^\s*click\s+(?:the\s+)?(.+?)\s*$", text)
    if clicked:
        label = re.sub(r"(?i)\s+(?:button|link)$", "", clicked.group(1)).strip(" '\".")
        ref = _snapshot_ref(snapshot, label)
        if not ref:
            raise ToolError(f"could not uniquely find {label!r} in the browser snapshot")
        return ["web_browser_click", "browser_click"], {"element": label, "ref": ref}

    pressed = re.search(r"(?i)^\s*press\s+(.+?)\s*$", text)
    if pressed:
        return ["web_browser_press_key", "browser_press_key"], {"key": pressed.group(1)}
    raise ToolError(
        f"unsupported browser action {text!r}; use open/read, click <label>, "
        "type 'text' into <label>, or press <key>")


def _native_action_spec(action: str, target: str) -> dict:
    """Compile a human desktop action into ``desktop_native`` arguments."""
    text = action.strip()
    window = target.strip()
    if re.search(r"(?i)\b(?:screenshot|capture (?:the )?screen)\b", text):
        return {"action": "screenshot"}
    if re.search(r"(?i)\b(?:list|show) (?:the )?windows\b", text):
        return {"action": "windows"}
    if re.search(r"(?i)^\s*(?:launch|open|start)\b", text):
        named = re.sub(r"(?i)^\s*(?:launch|open|start)\s+(?:the\s+)?", "", text).strip(" .")
        app = window if named.lower() in {"", "app", "application", "program", "window"} else named
        return {"action": "launch", "app": app}
    focused = re.search(r"(?i)^\s*(?:focus|activate|bring forward)\s*(.*)$", text)
    if focused:
        return {"action": "focus", "window": focused.group(1).strip(" .") or window}
    typed = re.search(r"(?i)^\s*(?:type|write|enter)\s+(['\"])(.*?)\1", text)
    if typed:
        return {"action": "type", "text": typed.group(2), "window": window}
    clicked = re.search(r"(?i)^\s*click\s+(?:at\s*)?\(?\s*(\d+)\s*[,x]\s*(\d+)\s*\)?", text)
    if clicked:
        return {"action": "click", "x": int(clicked.group(1)),
                "y": int(clicked.group(2)), "window": window}
    located = re.search(r"(?i)^\s*(?:locate|find)\s+(?:the\s+)?(.+?)\s*$", text)
    if located:
        return {"action": "locate", "target": located.group(1).strip(" ."),
                "window": window}
    pressed = re.search(r"(?i)^\s*press\s+(.+?)\s*$", text)
    if pressed:
        return {"action": "key", "keys": pressed.group(1), "window": window}
    if re.search(r"(?i)^\s*save(?: the)? file\s*$", text):
        return {"action": "key", "keys": "ctrl+s", "window": window}
    raise ToolError(
        f"unsupported desktop action {text!r}; use screenshot, launch/focus, "
        "locate, click x,y, type 'text', press <keys>, or save file")


def _snapshot_highlights(snapshot: str, limit: int = 700) -> str:
    """Compress an accessibility snapshot into evidence that fits agent context.

    Raw browser snapshots begin with navigation chrome and can be tens of
    thousands of characters. Downstream agents have a deliberately small
    shared-context budget, so keep semantic titles, URLs, headings, and text
    snippets while dropping refs and common browser/search controls.
    """
    skip = re.compile(
        r"(?i)skip to|accessibility help|go to google home|search by (?:voice|image)|"
        r"google apps|sign in|settings|\blink \"(?:all|images|videos|news)\""
    )
    interesting = re.compile(
        r"(?i)(?:page title:|^- (?:heading|strong|text:|/url:\s*https?://))"
    )
    titles: List[str] = []
    urls: List[str] = []
    content: List[str] = []
    seen = set()
    for raw_line in str(snapshot).splitlines():
        line = raw_line.strip()
        if not interesting.search(line) or skip.search(line):
            continue
        if re.search(r"(?i)^-?\s*/url:\s*https?://(?:[^/]*\.)?google\.", line):
            continue
        line = re.sub(r"\s*\[(?:ref|cursor|level|disabled|active)[^\]]*\]", "", line)
        line = re.sub(r"^-\s*", "", line).strip()
        if not line or line in seen:
            continue
        if re.match(r"(?i)/url:\s*https?://", line):
            urls.append(line)
        elif line.lower().startswith("page title:"):
            titles.append(line)
        else:
            content.append(line)
        seen.add(line)

    # Direct source URLs are the scarcest and most valuable evidence. Search
    # pages often put a long generated overview before their organic results,
    # so source URLs must not be pushed beyond the context limit.
    highlights: List[str] = []
    used = 0
    for line in titles + urls[:8] + content:
        addition = len(line) + (1 if highlights else 0)
        if used + addition > limit:
            continue
        highlights.append(line)
        used += addition
    return "\n".join(highlights) or str(snapshot)[:limit]


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

        if plan.backend in {"web_browser_navigate", "browser_navigate"}:
            return self._run_browser_plan(plan, backend_tool, started)
        if plan.backend == "desktop_native":
            return self._run_native_desktop_plan(plan, backend_tool, started)

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

    def _run_native_desktop_plan(self, plan: ComputerUsePlan, desktop: Tool,
                                 started: float) -> str:
        """Drive the native backend with typed directives, never prose."""
        outputs: List[str] = []
        performed = 0
        target = plan.targets[0]
        for action in plan.actions or [plan.goal]:
            if performed >= plan.max_steps:
                outputs.append(f"stopped: reached the {plan.max_steps}-action limit")
                break
            if time.monotonic() - started >= plan.time_limit_s:
                outputs.append(f"stopped: reached the {plan.time_limit_s:.0f}s time limit")
                break
            spec = _native_action_spec(action, target)
            payload = "TOOL_DIRECTIVE: " + json.dumps({"arguments": spec})
            try:
                result = desktop.execute(payload)
            except Exception as exc:  # noqa: BLE001
                self._log({"event": "error", "action": action, "error": str(exc)[:500]})
                raise ToolError(f"desktop_native failed on {action!r}: {exc}") from exc
            performed += 1
            self._log({"event": "action", "action": action, "result": str(result)[:500]})
            outputs.append(f"{performed}. {action} -> {str(result)[:500]}")
        self._log({"event": "finish", "actions": performed,
                   "seconds": round(time.monotonic() - started, 2)})
        return (f"[computer_use] desktop_native: {performed} action(s) on "
                f"{', '.join(plan.targets)}\n" + ("\n".join(outputs) or "(no actions performed)"))

    def _run_browser_plan(self, plan: ComputerUsePlan, navigate: Tool,
                          started: float) -> str:
        """Run a descriptive browser plan using typed MCP calls.

        Navigation receives a real URL derived only from an already-approved
        target. A snapshot is appended when available so read/research steps
        return page content instead of merely reporting that navigation worked.
        """
        navigation_outputs: List[str] = []
        performed = 0
        urls = [_browser_url(target, plan.actions, plan.goal) for target in plan.targets]

        # ``targets`` is an allow-list, not a list of pages that must all be
        # opened.  If the plan asks Google for results from github.com, opening
        # the bare GitHub home page afterwards destroys the useful search-page
        # state. Keep explicit/search URLs and omit bare domains that merely
        # bound where later interactions are allowed.
        has_search = any("/search?" in url for url in urls)
        if has_search:
            urls = [url for url in urls if "/search?" in url or any(
                re.search(rf"https?://(?:[^/]+\.)?{re.escape((urlparse(url).hostname or '').lower())}(?:/[^\s'\"<>]*)?",
                          action, re.I)
                for action in plan.actions)]

        snapshot = None
        if self.tools is not None:
            for name in ("web_browser_snapshot", "browser_snapshot"):
                candidate = self.tools.get(name)
                if candidate is not None and not candidate.is_broken and candidate.is_live():
                    snapshot = candidate
                    break

        last_snapshot = ""
        for url in urls:
            if performed >= plan.max_steps or time.monotonic() - started >= plan.time_limit_s:
                break
            payload = "TOOL_DIRECTIVE: " + json.dumps({"arguments": {"url": url}})
            try:
                result = navigate.execute(payload)
            except Exception as exc:  # noqa: BLE001 - report, do not crash the run
                self._log({"event": "error", "action": f"navigate {url}",
                           "error": str(exc)[:500]})
                raise ToolError(f"{plan.backend} failed while opening '{url}': {exc}") from exc
            performed += 1
            self._log({"event": "action", "action": f"navigate {url}",
                       "result": str(result)[:500]})
            navigation_outputs.append(f"{performed}. opened {url} -> {str(result)[:1000]}")
            if (snapshot is not None and performed < plan.max_steps
                    and time.monotonic() - started < plan.time_limit_s):
                try:
                    snapshot_result = snapshot.execute("")
                except Exception as exc:  # noqa: BLE001 - navigation is still useful
                    self._log({"event": "error", "action": "snapshot",
                               "error": str(exc)[:500]})
                    navigation_outputs.append(f"snapshot unavailable: {exc}")
                else:
                    performed += 1
                    last_snapshot = str(snapshot_result)
                    self._log({"event": "action", "action": "snapshot",
                               "result": str(snapshot_result)[:500]})
                    # Replace navigation chatter with compact evidence while
                    # this exact page is still open.
                    navigation_outputs[-1] = (
                        f"page highlights from {url}:\n"
                        f"{_snapshot_highlights(str(snapshot_result))}")

        for action in plan.actions:
            interaction = _browser_interaction(action, last_snapshot)
            if interaction is None:
                continue
            if performed >= plan.max_steps or time.monotonic() - started >= plan.time_limit_s:
                navigation_outputs.append("stopped: browser action/time limit reached")
                break
            names, arguments = interaction
            action_tool = None
            if self.tools is not None:
                for name in names:
                    candidate = self.tools.get(name)
                    if candidate is not None and not candidate.is_broken and candidate.is_live():
                        action_tool = candidate
                        break
            if action_tool is None:
                raise ToolError(f"browser backend lacks the tool needed for {action!r}")
            payload = "TOOL_DIRECTIVE: " + json.dumps({"arguments": arguments})
            try:
                result = action_tool.execute(payload)
            except Exception as exc:  # noqa: BLE001
                self._log({"event": "error", "action": action, "error": str(exc)[:500]})
                raise ToolError(f"browser failed on {action!r}: {exc}") from exc
            performed += 1
            self._log({"event": "action", "action": action, "result": str(result)[:500]})
            navigation_outputs.append(f"{performed}. {action} -> {str(result)[:500]}")
            if (snapshot is not None and performed < plan.max_steps
                    and time.monotonic() - started < plan.time_limit_s):
                snapshot_result = snapshot.execute("")
                performed += 1
                last_snapshot = str(snapshot_result)
                navigation_outputs.append(
                    f"after {action}:\n{_snapshot_highlights(last_snapshot)}")

        self._log({"event": "finish", "actions": performed,
                   "seconds": round(time.monotonic() - started, 2)})
        body = "\n".join(navigation_outputs) or "(no actions performed)"
        return (f"[computer_use] {plan.backend}: {performed} action(s) on "
                f"{', '.join(plan.targets)}\n{body}")
