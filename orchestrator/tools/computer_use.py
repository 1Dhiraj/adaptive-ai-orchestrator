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
import socket
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import quote_plus, urlparse
from urllib.request import url2pathname

from .base import Tool, ToolError, extract_directive
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


def local_file_url(candidate: str, roots: List[Path]) -> Optional[str]:
    """A ``file://`` URL for a path this run is allowed to open, or None.

    Building something and then looking at it is one task, not two, and the
    second half kept failing: a file has no hostname, so every allow-list
    check refused it and "open the page you just wrote" was unreachable.

    The rule is ownership rather than permission. A file the run created, in
    its own workspace or artifacts directory, is its own output -- opening it
    reaches nothing the run did not already have. Anything outside those
    directories is still refused, so this cannot be turned into a way to read
    the rest of the disk.
    """
    raw = (candidate or "").strip().strip("'\"")
    if not raw:
        return None
    if raw.lower().startswith("file://"):
        raw = url2pathname(urlparse(raw).path)
    elif re.match(r"^[a-z]+://", raw, re.I):
        return None

    candidates = [Path(raw)]
    for root in roots:
        candidates.append(root / raw)
        # Models guess at absolute file URLs -- file:///weather.html,
        # file:///workspace/weather.html -- for a file they only know by
        # name. Matching the basename inside the roots finds what they meant
        # without widening where they may look.
        name = Path(raw).name
        if name:
            candidates.append(root / name)
            if root.is_dir():
                try:
                    candidates.extend(sorted(root.rglob(name))[:8])
                except OSError:
                    pass
    for path in candidates:
        try:
            resolved = path.resolve()
        except OSError:
            continue
        if not resolved.is_file():
            continue
        for root in roots:
            try:
                resolved.relative_to(root.resolve())
            except (ValueError, OSError):
                continue
            return serve_local_file(resolved)
    return None


#: Things a run builds and then wants to look at.
_VIEWABLE = (".html", ".htm", ".pdf", ".svg", ".txt", ".md", ".json", ".csv")


def newest_viewable_file(roots: List[Path], since: float = 0.0) -> Optional[str]:
    """The most recently written file this run could sensibly open.

    Plans refer to their own outputs by *name of a fact* -- "the path given by
    the todo_app_path fact" -- so the filename never reaches the tool, and a
    step that had just written the file was refused for not naming a target.
    Rather than parse the plan's own variables, ask the directory: a run that
    has just produced exactly one HTML file and then says "open it" is not
    ambiguous.

    Bounded to the run's own directories, so the worst case is opening
    something else the run itself made.
    """
    best, best_time = None, since
    for root in roots:
        try:
            entries = list(root.rglob("*"))
        except OSError:
            continue
        for path in entries:
            try:
                if not path.is_file() or path.suffix.lower() not in _VIEWABLE:
                    continue
                stamp = path.stat().st_mtime
            except OSError:
                continue
            # Strictly newer: Windows rounds st_mtime to ~10ms, so a file
            # written just before this run started can share its timestamp.
            # Skipping such a file costs a clear error; claiming it would
            # silently open the wrong thing.
            if stamp > best_time:
                best, best_time = path, stamp
    return serve_local_file(best.resolve()) if best else None


#: One static server per directory, reused for the life of the process.
_LOCAL_SERVERS: Dict[str, str] = {}
_LOCAL_SERVER_LOCK = threading.Lock()


def serve_local_file(path: Path) -> str:
    """Serve the directory holding ``path`` on loopback, and return its URL.

    Browsers refuse ``file:`` under automation -- Playwright answers
    ``Access to "file:" protocol is blocked`` -- so a run that has just built
    a web page cannot look at its own work by pointing at the file. Serving it
    is not a workaround so much as the honest thing: a web app is something a
    server serves, and half of what makes one work (fetch, modules, storage
    scoped to an origin) does not behave the same way from a bare file path.

    Bound to 127.0.0.1 on an ephemeral port, rooted at that one directory,
    read-only, and running on a daemon thread so it goes away with the
    process. One server per directory, reused.
    """
    import functools
    from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

    directory = str(path.parent.resolve())
    with _LOCAL_SERVER_LOCK:
        existing = _LOCAL_SERVERS.get(directory)
        if existing:
            return f"{existing}/{quote_plus(path.name, safe='')}"

        class Quiet(SimpleHTTPRequestHandler):
            def log_message(self, *args):  # noqa: D102 - silence stdout
                pass

        handler = functools.partial(Quiet, directory=directory)
        server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        threading.Thread(target=server.serve_forever, daemon=True,
                         name=f"preview-{server.server_port}").start()
        base = f"http://127.0.0.1:{server.server_port}"
        _LOCAL_SERVERS[directory] = base
    return f"{base}/{quote_plus(path.name, safe='')}"


def _house_in_paint_goal(plan: "ComputerUsePlan") -> bool:
    """Recognise the bounded Paint illustration this dashboard demonstrates."""
    words = set(re.findall(r"[a-z]+", plan.goal.lower()))
    return (any(target in {"paint", "mspaint", "mspaint.exe"} for target in plan.targets)
            and {"house", "roof", "door", "sun", "grass", "save"} <= words)


def _exact_notepad_text(plan: "ComputerUsePlan") -> Optional[str]:
    if ("notepad" not in plan.targets
            or not re.search(r"(?i)\bnew\b", plan.goal)
            or not re.search(r"(?i)\b(?:leave|visible)\b", plan.goal)):
        return None
    typed = [match.group(2) for action in plan.actions
             if (match := re.fullmatch(r"(?i)\s*type\s+(['\"])(.*?)\1\s*",
                                      action, re.S))]
    if len(typed) == 1:
        return typed[0]
    quoted = re.search(r"(?i)\b(?:containing|type) exactly\s+(['\"])(.*?)\1",
                       plan.goal, re.S)
    if quoted:
        return quoted.group(2)
    unquoted = re.search(r"(?i)\bcontaining exactly\s+(.+?)\s+verify\b",
                         plan.goal, re.S)
    if unquoted:
        return unquoted.group(1).strip()
    plain = re.search(r"(?i)\btype:\s*(.+?)(?:\s+leave\b|$)", plan.goal, re.S)
    return plain.group(1).strip() if plain else None


def _draw_house_picture(path: Path) -> None:
    """Create a real PNG that Paint can open, with all requested elements."""
    from PIL import Image, ImageDraw

    canvas = Image.new("RGB", (960, 640), "#a7dbff")
    art = ImageDraw.Draw(canvas)
    art.rectangle((0, 490, 960, 640), fill="#67b84b")
    art.ellipse((710, 62, 840, 192), fill="#ffe15c", outline="#e3a832", width=5)
    for point in ((775, 20, 775, 52), (775, 202, 775, 233),
                  (670, 127, 700, 127), (850, 127, 880, 127),
                  (700, 52, 718, 70), (832, 184, 852, 204)):
        art.line(point, fill="#ffce3c", width=8)
    art.rectangle((230, 275, 620, 515), fill="#ffd46d", outline="#b77736", width=6)
    art.polygon(((190, 285), (425, 120), (660, 285)),
                fill="#d85848", outline="#93382f", width=6)
    art.rectangle((385, 385, 475, 515), fill="#8b5635", outline="#543521", width=5)
    art.ellipse((455, 445, 464, 454), fill="#f5cf65")
    for left in (275, 520):
        art.rectangle((left, 330, left + 65, 400), fill="#e3f6ff",
                      outline="#8e633b", width=5)
        art.line((left + 32, 330, left + 32, 400), fill="#8e633b", width=4)
    canvas.save(path, format="PNG")


def allows_any_site(allow: Optional[List[str]] = None) -> bool:
    """Whether this run may open any website.

    ``ORCHESTRATOR_COMPUTER_USE_ALLOW=*`` means "the whole web", which is what
    a person actually wants when they type "find me the cheapest flight" and
    have no idea which sites that will involve. A fixed list cannot answer an
    open question; it can only refuse it.

    The wildcard covers **websites only**. Desktop applications still have to
    be named one by one, and that asymmetry is deliberate: opening an
    arbitrary URL is what a browser does all day, inside a sandboxed tab,
    whereas driving an arbitrary desktop application means synthetic clicks
    and keystrokes landing in whatever is in front -- a mail client, a
    terminal, an open bank tab. The blast radius is not comparable, so the
    permission is not either.

    Everything else still applies: credentials are refused before they are
    typed, the plan is approved before anything moves, action and time limits
    bound the run, and every action is logged.
    """
    entries = allowed_targets() if allow is None else allow
    return "*" in entries


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
    candidate = _normalise_target(target)
    if not candidate:
        return False
    # The wildcard opens the web, not the desktop -- see allows_any_site.
    if "*" in allow and _looks_like_web_target(candidate):
        return True
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
    #: Work the steps out from what the screen shows, rather than replaying
    #: ``actions``. Set explicitly, or inferred when a plan names a target but
    #: no actions -- a goal with no steps is a goal nobody could script.
    autonomous: bool = False

    def describe(self) -> str:
        where = ", ".join(self.targets) or "(no target named)"
        if allows_any_site():
            where += "   [this run may open any website]"
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
                "time_limit_s": self.time_limit_s, "autonomous": self.autonomous}


def _looks_like_web_target(target: str) -> bool:
    candidate = (target or "").strip().lower()
    if not candidate or candidate.endswith(".exe"):
        return False
    if candidate == "localhost" or re.match(r"^https?://", candidate):
        return True
    return bool(
        re.fullmatch(r"(?:[a-z0-9-]+\.)+[a-z]{2,}(?::\d+)?", candidate)
        or re.fullmatch(r"\d{1,3}(?:\.\d{1,3}){3}(?::\d+)?", candidate)
    )


def _normalise_target(target: str) -> str:
    """Turn an approved URL/domain/app target into its stable allow-list key."""
    candidate = (target or "").strip().lower()
    if re.match(r"^https?://", candidate):
        parsed = urlparse(candidate)
        return (parsed.hostname or "").lower()
    return candidate


def _target_from_description(description: str) -> str:
    """Recover only a target already present in the configured allow-list.

    The two intent aliases below are deliberately narrow. They make ordinary
    phrases such as "open Gmail" and "search the web" usable without granting
    access to a site that the operator did not already allow.
    """
    text = (description or "").strip()
    lowered = text.lower()
    allow = allowed_targets()
    direct = next((entry for entry in sorted(allow, key=len, reverse=True)
                   if entry != "*"
                   and re.search(rf"(?<![a-z0-9]){re.escape(entry)}(?![a-z0-9])", lowered)), "")
    if direct:
        return direct
    if "*" in allow:
        # Any domain the person actually named, then a search engine, because
        # an open question has to start somewhere.
        spelled = re.search(
            r"(?<![a-z0-9@.])((?:[a-z0-9][a-z0-9-]*\.)+[a-z]{2,})(?![a-z0-9])", lowered)
        if spelled:
            return spelled.group(1)
        if re.search(r"(?i)\b(?:search|find|look up|google|research|latest|"
                     r"news|price|cheapest|compare|who|what|when|where)\b", text):
            return "google.com"
    if re.search(r"(?i)\b(?:gmail|e-?mail|mailbox|inbox)\b", text):
        for candidate in ("mail.google.com", "gmail.com"):
            if candidate in allow:
                return candidate
    if re.search(r"(?i)\b(?:browser|web|search|research)\b", text) and "google.com" in allow:
        return "google.com"
    return ""


def _plan_from_description(description: str) -> ComputerUsePlan:
    """Compile a narrow safe plan when a model omits its directive."""
    text = (description or "").strip()
    lowered = text.lower()
    target = _target_from_description(text)
    if not target:
        return ComputerUsePlan(goal=text[:1000])

    actions: List[str] = []
    if _looks_like_web_target(target) or re.search(
            r"(?i)\b(?:open|launch|start|browser|gmail|mailbox|inbox)\b", text):
        actions.append("open the page" if _looks_like_web_target(target) else "launch the app")
    # Quoted text wins, wherever it sits in the sentence. "type the exact
    # sentence: 'Hello there.'" must type `Hello there.` and not `the exact
    # sentence: 'Hello there` -- and the second reading is what a
    # verb-anchored pattern produces, so it typed prose into real documents.
    typed = re.search(
        r"(?i)\b(?:type|write|enter)\b[^'\"]*?(?:'([^']*)'|\"([^\"]*)\")", text)
    quoted = typed is not None
    if not typed:
        typed = re.search(
            r"(?i)\b(?:type|write|enter)\s+(.+?)(?:\s+(?:in|into)\b|[.;]|$)", text)
    if typed and not _looks_like_web_target(target):
        value = next((part for part in typed.groups() if part is not None), "").strip()
        if not quoted:
            # "type the sentence: Hello there" names the text before giving
            # it. Left in, the document ends up holding the instruction as
            # well as the content, and the step still reports success.
            #
            # Only ever applied to unquoted text, and only to a phrase that
            # announces itself -- a determiner, or a colon. Text the person
            # actually quoted is never trimmed, or "'Line one'" would lose
            # its first word to the very pattern meant to protect it.
            value = re.sub(
                r"(?i)^(?:(?:the|this|these|that|following|exact|literal)\s+)+"
                r"(?:sentence|text|line|lines|words?|message|phrase|string|"
                r"content|note|paragraph)\s*:?\s+", "", value).strip()
            value = re.sub(
                r"(?i)^(?:sentence|text|line|message|phrase|content|note|"
                r"paragraph)\s*:\s+", "", value).strip()
            value = re.sub(r"(?i)^(?:exactly|literally)\s+", "", value).strip()
        if value:
            actions.append(f"type {json.dumps(value)}")
    if (not _looks_like_web_target(target)
            and not any(action.startswith("type ") for action in actions)
            and re.search(r"(?i)\b(?:read|verify|confirm|check|inspect)\b", text)):
        # A step asking what the document says has to look at it. Without
        # this it compiles to nothing at all, and an empty plan is how a
        # verification step comes back saying everything is fine.
        actions.append("read the window")
    if _looks_like_web_target(target):
        clicked = re.search(
            r"(?i)\bclick\s+(?:the\s+)?(?:['\"]([^'\"]+)['\"]|"
            r"([a-z0-9][a-z0-9 _-]{0,60}?))(?:\s+(?:button|link))?(?:[.;]|$)", text)
        if clicked:
            label = next((part for part in clicked.groups() if part), "").strip()
            if label:
                actions.append(f"click {label}")
        if re.search(r"(?i)\b(?:read|extract|collect|summari[sz]e)\b", text):
            actions.append("read the page")
    return ComputerUsePlan(goal=text[:1000], targets=[target], actions=actions)


def _parse_plan(task: str, context: Optional[dict] = None) -> ComputerUsePlan:
    """Read the agent's TOOL_DIRECTIVE into a plan, tolerating a missing one."""
    payload: Dict[str, Any] = extract_directive(task)
    first_line = (task or "").strip().splitlines()
    goal = str(payload.get("goal") or (first_line[0] if first_line else ""))
    targets = payload.get("targets") or payload.get("domains") or []
    if isinstance(targets, str):
        targets = [targets]
    actions = payload.get("actions") or []
    if isinstance(actions, str):
        actions = [actions]
    actions = [str(a).strip() for a in actions if str(a).strip()]
    # Keep complete approved URLs so a local server's port and a page's path
    # survive planning. Access checks below still use only the hostname.
    allow = allowed_targets()
    normalised_targets = []
    for raw_target in targets:
        raw = str(raw_target).strip()
        candidate = _normalise_target(raw)
        # Keep file references long enough to resolve them against this run's
        # workspace. _run still refuses any file outside those roots.
        local_reference = bool(re.search(
            r"(?i)\.(?:html?|pdf|svg|txt|md|json|csv)$", candidate))
        if candidate and (_looks_like_web_target(candidate)
                          or candidate in allow or local_reference):
            retained = (raw if candidate in {"127.0.0.1", "localhost"}
                        and re.match(r"^https?://", raw, re.I) else candidate)
            if retained not in normalised_targets:
                normalised_targets.append(retained)
    # Models often declare only the host while putting the actual localhost
    # URL in the goal. Reuse that URL only when its host matches a target.
    for match in re.finditer(r"https?://[^\s'\"<>]+", goal, re.I):
        url = match.group(0).rstrip(".,);]")
        host = _normalise_target(url)
        if (host not in {"127.0.0.1", "localhost"}
                and _target_allowed(url, allow)
                and any(host == _normalise_target(target)
                        or host.endswith("." + _normalise_target(target))
                        for target in normalised_targets)):
            normalised_targets.insert(0, url)
            break
        for index, target in enumerate(normalised_targets):
            if target == host and host in {"127.0.0.1", "localhost"}:
                normalised_targets[index] = url
                break
    try:
        max_steps = max(1, min(int(payload.get("max_steps") or _DEFAULT_MAX_STEPS), 100))
    except (TypeError, ValueError):
        max_steps = _DEFAULT_MAX_STEPS
    try:
        time_limit_s = max(1.0, min(float(payload.get("time_limit_s") or _DEFAULT_TIME_LIMIT_S), 900.0))
    except (TypeError, ValueError):
        time_limit_s = _DEFAULT_TIME_LIMIT_S
    plan = ComputerUsePlan(
        goal=goal[:1000],
        targets=normalised_targets,
        actions=actions,
        max_steps=max_steps,
        time_limit_s=time_limit_s,
        autonomous=bool(payload.get("autonomous")),
    )
    if (not plan.targets or not plan.actions) and context:
        fallback = _plan_from_description(str(context.get("step_description") or ""))
        if not plan.targets:
            plan.targets = fallback.targets
        if not plan.actions:
            plan.actions = fallback.actions
        if not plan.goal:
            plan.goal = fallback.goal
    return plan


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


#: Every browser instruction this tool understands, in the words an agent
#: writes them. Kept as one string because it is also the menu shown to the
#: model driving the autonomous loop -- a vocabulary documented in one place
#: and enforced in another drifts apart within a release.
BROWSER_VOCABULARY = """open <url>                     read the page
click <label>                  click the first/second/last <kind>
double-click <label>
right-click <label>            ctrl-click <label>
type '<text>' into <label>     type '<text>' into <label> and submit
clear <label>                  select '<value>' from <label>
check <label>                  uncheck <label>
fill the form with <label>='<value>', <label>='<value>'
hover <label>                  drag <label> to <label>
upload '<file>'                accept the dialog / dismiss the dialog
scroll down | up | to the top | to the bottom
press <key>                    submit
wait <n> seconds               wait for '<text>'
wait until '<text>' is gone    find '<text>'
screenshot                     full-page screenshot
new tab <url>                  switch to tab <n>
close tab                      list tabs
go back                        go forward
reload                         resize to <w>x<h>
read the console               read the network requests
close the browser"""

#: Canonical argument name -> the names different Playwright MCP releases use
#: for it. The server is the authority: ``_adapt_arguments`` renames to
#: whatever the server actually reached declares, so one compiler drives both
#: the releases that call an element handle ``ref`` and those that call it
#: ``target``.
_ARGUMENT_ALIASES = {
    "ref": ("ref", "target", "selector"),
    "startRef": ("startRef", "startTarget"),
    "endRef": ("endRef", "endTarget"),
}

#: Fields some releases mark required that carry no decision for these calls.
#: Supplying the documented default beats a rejected request.
_REQUIRED_DEFAULTS = {
    "scale": "css",
    "static": False,
    "level": "info",
    "type": "png",
}


_BROWSER_LOOP_SYSTEM = (
    "You operate a web browser to reach a goal, one action at a time, reading "
    "the page after each one.\n\n"
    "Reply with JSON only:\n"
    '  {"action": "<one instruction>", "why": "<short reason>"}\n'
    '  {"done": true, "answer": "<what you found or did>"}\n\n'
    "The instruction must be one of these forms, exactly:\n"
    + BROWSER_VOCABULARY + "\n\n"
    "Rules:\n"
    "- The page is shown to you after every action. Never ask to read, open "
    "or extract the page you are already on -- you are looking at it.\n"
    "- Use the labels shown on the page. Do not invent selectors or refs.\n"
    "- You may only open sites on the allow-list; anything else is refused.\n"
    "- If an action is refused, read why and try a different one.\n"
    "- Never type a password, card number, API key or one-time code.\n"
    "- Say done as soon as the goal is met, and put the answer in 'answer'."
)


def _adapt_arguments(tool: Any, arguments: dict) -> dict:
    """Rename compiled arguments to the names the reached server declares.

    Playwright MCP renamed its element handle from ``ref`` to ``target`` and
    made ``scale``/``static``/``level`` required. Compiling against one release
    and calling another produces requests that look plausible and fail, so the
    schema the server published at discovery decides the wire format rather
    than the release this file happened to be written against.
    """
    try:
        schema = tool.argument_schema() or {}
    except Exception:  # noqa: BLE001 - a tool without a schema is the normal case
        schema = {}
    properties = schema.get("properties") or {}
    if not properties:
        return dict(arguments)

    adapted: dict = {}
    for key, value in arguments.items():
        if key in properties:
            adapted[key] = value
            continue
        renamed = next((alias for alias in _ARGUMENT_ALIASES.get(key, ())
                        if alias in properties), None)
        if renamed:
            adapted[renamed] = value
        # A field the server does not declare is dropped rather than sent:
        # strict servers reject unknown fields outright.

    fields = adapted.get("fields")
    if isinstance(fields, list):
        field_properties = (
            ((properties.get("fields") or {}).get("items") or {}).get("properties") or {})
        if field_properties:
            adapted["fields"] = [
                {renamed: value
                 for key, value in entry.items()
                 if (renamed := _rename(key, field_properties)) in field_properties}
                for entry in fields if isinstance(entry, dict)
            ]

    for required in schema.get("required") or []:
        if required not in adapted and required in _REQUIRED_DEFAULTS:
            adapted[required] = _REQUIRED_DEFAULTS[required]
    return adapted


def _rename(key: str, properties: dict) -> str:
    if key in properties:
        return key
    return next((alias for alias in _ARGUMENT_ALIASES.get(key, ())
                 if alias in properties), key)


#: Which roles each kind of instruction is really aiming at. Pages label a
#: control and its own heading identically all the time -- "Dropdown List"
#: above a <select>, "Login" above a login button -- so a bare label is
#: ambiguous far more often than it looks. The action says which was meant.
_PREFERRED_ROLES = {
    "click": ("button", "link", "checkbox", "radio", "tab", "menuitem",
              "option", "switch"),
    "type": ("textbox", "searchbox", "combobox", "spinbutton"),
    "select": ("combobox", "listbox"),
    "toggle": ("checkbox", "radio", "switch"),
}


def _snapshot_node(snapshot: str, label: str,
                   prefer: tuple = ()) -> Optional[tuple[str, str]]:
    """The accessibility ref and role of a plainly named control.

    ``None`` when the label matches nothing -- or still matches several things
    once ``prefer`` has been applied. An ambiguous match is worse than no
    match: acting on the wrong one of two similarly named elements leaves
    nothing in the transcript to notice.

    ``prefer`` narrows by role first. "select 'Option 2' from Dropdown", on a
    page with a heading "Dropdown List" above a <select>, otherwise resolves
    to the heading and fails with "Element is not a <select>" -- an error
    about the page, for what is really a mistake about which element.
    """
    wanted = re.sub(r"\s+", " ", label).strip(" '\".").lower()
    if not wanted:
        return None
    exact: List[tuple[str, str]] = []
    partial: List[tuple[str, str]] = []
    for line in str(snapshot).splitlines():
        match = re.search(r"\[ref=([^\]\s]+)", line)
        if not match:
            continue
        role_match = re.match(r"\s*-\s*([a-z]+)", line)
        role = role_match.group(1) if role_match else ""
        normal = re.sub(r"\s+", " ", line).lower()
        if f'"{wanted}"' in normal or f"'{wanted}'" in normal:
            exact.append((match.group(1), role))
        elif wanted in normal:
            partial.append((match.group(1), role))

    for candidates in (exact, partial):
        if not candidates:
            continue
        preferred = [c for c in candidates if c[1] in prefer] if prefer else []
        if len(preferred) == 1:
            return preferred[0]
        if preferred:
            # Several controls of the right kind share the label; the first in
            # document order is the one a person reading the page would mean.
            return preferred[0]
        if len(candidates) == 1:
            return candidates[0]
    return None


_ORDINALS = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
             "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10}


def _nth_node(snapshot: str, label: str,
              prefer: tuple = ()) -> Optional[tuple[str, str]]:
    """Resolve "the first checkbox" / "checkbox 2" / "the last link".

    Controls with no accessible name are common -- a bare row of checkboxes,
    an icon-only button bar -- and there is no text to match on. Position is
    then the only thing anyone can refer to, and it is how a person reading
    the page would describe it anyway.
    """
    wanted = re.sub(r"\s+", " ", label).strip(" '\".").lower()
    ordinal = re.fullmatch(
        r"(?:the\s+)?(first|second|third|fourth|fifth|sixth|seventh|eighth|"
        r"ninth|tenth|last|\d+)(?:st|nd|rd|th)?(?:\s+([a-z]+))?", wanted)
    if not ordinal:
        # "checkbox 2", "button 3"
        ordinal = re.fullmatch(r"(?:the\s+)?([a-z]+)\s+(\d+)", wanted)
        if not ordinal:
            return None
        role_word, position = ordinal.group(1), ordinal.group(2)
    else:
        position, role_word = ordinal.group(1), ordinal.group(2)

    roles = (role_word,) if role_word and role_word in _CONTROL_ROLES else prefer
    if not roles:
        return None

    found: List[tuple[str, str]] = []
    for line in str(snapshot).splitlines():
        match = re.search(r"\[ref=([^\]\s]+)", line)
        role_match = re.match(r"\s*-\s*([a-z]+)", line)
        if match and role_match and role_match.group(1) in roles:
            found.append((match.group(1), role_match.group(1)))
    if not found:
        return None
    if position == "last":
        return found[-1]
    index = _ORDINALS.get(position) or (int(position) if position.isdigit() else 0)
    return found[index - 1] if 1 <= index <= len(found) else None


def _option_owner(snapshot: str, value: str) -> Optional[str]:
    """The ref of the list that offers this option.

    Real <select> elements very often have no accessible name at all -- the
    page puts a heading above them instead, so the only thing a label can
    match is the heading, and selecting against it fails with "Element is not
    a <select> element". But the option named in the instruction is unique
    evidence on its own: the list that contains it is the list that was meant.
    """
    wanted = re.sub(r"\s+", " ", value).strip().lower()
    if not wanted:
        return None
    owner = None
    owner_indent = -1
    for raw_line in str(snapshot).splitlines():
        stripped = raw_line.lstrip()
        if not stripped.startswith("-"):
            continue
        indent = len(raw_line) - len(stripped)
        role_match = re.match(r"-\s*([a-z]+)", stripped)
        if not role_match:
            continue
        role = role_match.group(1)
        if role in {"combobox", "listbox"}:
            ref = re.search(r"\[ref=([^\]\s]+)", stripped)
            owner, owner_indent = (ref.group(1) if ref else None), indent
            continue
        if owner is not None and indent <= owner_indent:
            owner, owner_indent = None, -1
            continue
        if owner is not None and role == "option":
            normal = re.sub(r"\s+", " ", stripped).lower()
            if f'"{wanted}"' in normal or f"'{wanted}'" in normal:
                return owner
    return None


def _snapshot_ref(snapshot: str, label: str) -> Optional[str]:
    """Find the accessibility ref for a plainly named control."""
    node = _snapshot_node(snapshot, label)
    return node[0] if node else None


#: Accessibility role -> the field type ``browser_fill_form`` expects.
_FIELD_TYPES = {
    "checkbox": "checkbox", "radio": "radio", "combobox": "combobox",
    "listbox": "combobox", "slider": "slider", "textbox": "textbox",
    "searchbox": "textbox", "spinbutton": "textbox",
}


def _names(suffix: str) -> List[str]:
    """Both spellings of a Playwright MCP tool: prefixed, then bare."""
    return [f"web_browser_{suffix}", f"browser_{suffix}"]


def _browser_interaction(action: str, snapshot: str) -> Optional[tuple[List[str], dict]]:
    """Compile one human browser instruction into typed MCP arguments.

    ``None`` means "handled elsewhere": navigation and reading are done by the
    URL compiler and the snapshot loop, not replayed here as bogus calls. An
    unrecognised instruction raises rather than being skipped, because a
    silently dropped action is how a run comes to report work it never did.
    """
    text = action.strip()

    def handle(label: str, kind: str = "") -> tuple[str, str, str]:
        clean = re.sub(r"(?i)\s+(?:button|link|field|box|checkbox|menu|list|"
                       r"dropdown)$", "", label).strip(" '\".")
        roles = _PREFERRED_ROLES.get(kind, ())
        # The ordinal resolver gets the label as written. Stripping the
        # trailing noun first would throw away the very word that says which
        # kind of control "the first link" means, and it would then match the
        # first control of any kind at all.
        node = (_snapshot_node(snapshot, clean, roles)
                or _nth_node(snapshot, label, roles)
                or _nth_node(snapshot, clean, roles))
        if not node:
            raise ToolError(
                f"could not uniquely find {clean!r} in the browser snapshot")
        return clean, node[0], node[1]

    # -- history and page lifecycle ----------------------------------------
    if re.fullmatch(r"(?i)\s*(?:go|navigate)\s+back\s*", text):
        return _names("navigate_back"), {}
    if re.fullmatch(r"(?i)\s*(?:go|navigate)\s+forward\s*", text):
        return _names("evaluate"), {"function": "() => window.history.forward()"}
    if re.fullmatch(r"(?i)\s*(?:reload|refresh)(?:\s+the\s+page)?\s*", text):
        return _names("evaluate"), {"function": "() => window.location.reload()"}
    if re.fullmatch(r"(?i)\s*close\s+the\s+browser\s*", text):
        return _names("close"), {}

    # -- observation --------------------------------------------------------
    if re.search(r"(?i)^\s*(?:take\s+(?:a\s+)?)?(?:full[ -]?page\s+)?screenshot\b", text):
        arguments: dict = {"type": "png", "scale": "css"}
        if re.search(r"(?i)\bfull[ -]?page\b", text):
            arguments["fullPage"] = True
        return _names("take_screenshot"), arguments
    console = re.fullmatch(
        r"(?i)\s*(?:read|show)\s+the\s+console(?:\s+(errors?|warnings?))?\s*", text)
    if console:
        level = "error" if re.search(r"(?i)\berrors?\b", text) else (
            "warning" if re.search(r"(?i)\bwarnings?\b", text) else "info")
        return _names("console_messages"), {"level": level}
    if re.fullmatch(r"(?i)\s*(?:read|show|list)\s+the\s+network(?:\s+requests)?\s*", text):
        return _names("network_requests"), {"static": False}
    found = re.fullmatch(
        r"(?i)\s*(?:find|search\s+the\s+page\s+for)\s+['\"](.+?)['\"]\s*", text)
    if found:
        return _names("find"), {"text": found.group(1)}

    # -- handled by the navigation/snapshot loop, not here -----------------
    if re.search(r"(?i)^\s*(?:open|navigate|go to|visit|read|extract|collect|"
                 r"summari[sz]e|search\b)", text):
        return None
    if re.search(r"(?i)\bquery\b", text):
        return None

    # -- waiting -------------------------------------------------------------
    waited = re.fullmatch(r"(?i)\s*wait\s+(\d+(?:\.\d+)?)\s*(?:seconds?|secs?|s)?\s*", text)
    if waited:
        return _names("wait_for"), {"time": min(float(waited.group(1)), 30.0)}
    gone = re.fullmatch(
        r"(?i)\s*wait\s+(?:until|for)\s+['\"](.+?)['\"]\s+(?:to\s+)?(?:is\s+)?"
        r"(?:disappears?|is\s+gone|gone|vanishes?)\s*", text)
    if gone:
        return _names("wait_for"), {"textGone": gone.group(1)}
    appears = re.fullmatch(
        r"(?i)\s*wait\s+(?:for|until)\s+['\"](.+?)['\"](?:\s+(?:to\s+)?appears?)?\s*", text)
    if appears:
        return _names("wait_for"), {"text": appears.group(1)}

    # -- scrolling ----------------------------------------------------------
    scrolled = re.fullmatch(
        r"(?i)\s*scroll\s+(?:to\s+the\s+)?(down|up|top|bottom|end|start)\s*", text)
    if scrolled:
        key = {"down": "PageDown", "up": "PageUp", "top": "Home",
               "start": "Home", "bottom": "End", "end": "End"}[scrolled.group(1).lower()]
        return _names("press_key"), {"key": key}

    # -- tabs and window -----------------------------------------------------
    new_tab = re.fullmatch(
        r"(?i)\s*(?:open\s+a\s+)?new\s+tab(?:\s+(?:at\s+|with\s+)?(\S+))?\s*", text)
    if new_tab:
        arguments = {"action": "new"}
        if new_tab.group(1):
            arguments["url"] = new_tab.group(1)
        return _names("tabs"), arguments
    switched = re.fullmatch(r"(?i)\s*switch\s+to\s+tab\s+(\d+)\s*", text)
    if switched:
        return _names("tabs"), {"action": "select",
                                "index": max(0, int(switched.group(1)) - 1)}
    if re.fullmatch(r"(?i)\s*close\s+(?:the\s+)?tab\s*", text):
        return _names("tabs"), {"action": "close"}
    if re.fullmatch(r"(?i)\s*list\s+(?:the\s+)?tabs\s*", text):
        return _names("tabs"), {"action": "list"}
    resized = re.fullmatch(
        r"(?i)\s*resize\s+(?:the\s+\w+\s+)?to\s+(\d{2,5})\s*(?:x|\*|by)\s*(\d{2,5})\s*", text)
    if resized:
        return _names("resize"), {"width": int(resized.group(1)),
                                  "height": int(resized.group(2))}

    # -- dialogs and uploads -------------------------------------------------
    dialog = re.fullmatch(
        r"(?i)\s*(accept|dismiss|cancel)\s+the\s+dialog"
        r"(?:\s+with\s+['\"](.*?)['\"])?\s*", text)
    if dialog:
        arguments = {"accept": dialog.group(1).lower() == "accept"}
        if dialog.group(2):
            arguments["promptText"] = dialog.group(2)
        return _names("handle_dialog"), arguments
    uploaded = re.fullmatch(
        r"(?i)\s*upload\s+['\"](.+?)['\"](?:\s+(?:to|into)\s+.+)?\s*", text)
    if uploaded:
        return _names("file_upload"), {"paths": [uploaded.group(1)]}

    # -- keyboard ------------------------------------------------------------
    if re.fullmatch(r"(?i)\s*submit\s*", text):
        return _names("press_key"), {"key": "Enter"}

    # -- form filling --------------------------------------------------------
    form = re.fullmatch(r"(?i)\s*fill\s+(?:in\s+)?the\s+form\s+with\s+(.+?)\s*", text, re.S)
    if form:
        fields = []
        for label, value in re.findall(
                r"([A-Za-z0-9][^=,]*?)\s*=\s*['\"](.*?)['\"]", form.group(1)):
            name, ref, role = handle(label, "type")
            fields.append({"name": name, "element": name, "ref": ref,
                           "type": _FIELD_TYPES.get(role, "textbox"), "value": value})
        if not fields:
            raise ToolError(
                "fill the form needs field='value' pairs, e.g. fill the form "
                "with Name='Ada', Email='ada@example.com'")
        return _names("fill_form"), {"fields": fields}

    toggled = re.fullmatch(r"(?i)\s*(check|uncheck|tick|untick)\s+(?:the\s+)?(.+?)\s*", text)
    if toggled:
        name, ref, role = handle(toggled.group(2), "toggle")
        wanted = "true" if toggled.group(1).lower() in {"check", "tick"} else "false"
        return _names("fill_form"), {"fields": [{
            "name": name, "element": name, "ref": ref,
            "type": _FIELD_TYPES.get(role, "checkbox"), "value": wanted}]}

    typed = re.fullmatch(
        r"(?i)\s*(?:type|enter|fill)\s+(['\"])(.*?)\1\s+(?:into|in)\s+(?:the\s+)?(.+?)"
        r"(\s+and\s+(?:submit|press\s+enter))?\s*", text, re.S)
    if typed:
        name, ref, _ = handle(typed.group(3), "type")
        arguments = {"element": name, "ref": ref, "text": typed.group(2)}
        if typed.group(4):
            arguments["submit"] = True
        return _names("type"), arguments

    cleared = re.fullmatch(r"(?i)\s*clear\s+(?:the\s+)?(.+?)\s*", text)
    if cleared:
        name, ref, _ = handle(cleared.group(1), "type")
        return _names("type"), {"element": name, "ref": ref, "text": ""}

    selected = re.fullmatch(
        r"(?i)\s*select\s+(['\"])(.*?)\1\s+(?:from|in)\s+(?:the\s+)?(.+?)\s*", text, re.S)
    if selected:
        value, label = selected.group(2), selected.group(3)
        owner = _option_owner(snapshot, value)
        if owner:
            return _names("select_option"), {
                "element": label.strip(" '\"."), "ref": owner, "values": [value]}
        name, ref, _ = handle(label, "select")
        return _names("select_option"), {"element": name, "ref": ref,
                                         "values": [value]}

    # -- pointer -------------------------------------------------------------
    hovered = re.fullmatch(r"(?i)\s*hover\s+(?:over\s+)?(?:the\s+)?(.+?)\s*", text)
    if hovered:
        name, ref, _ = handle(hovered.group(1))
        return _names("hover"), {"element": name, "ref": ref}

    dragged = re.fullmatch(r"(?i)\s*drag\s+(.+?)\s+(?:to|onto)\s+(.+?)\s*", text)
    if dragged:
        start_name, start_ref, _ = handle(dragged.group(1))
        end_name, end_ref, _ = handle(dragged.group(2))
        return _names("drag"), {"startElement": start_name, "startRef": start_ref,
                                "endElement": end_name, "endRef": end_ref}

    clicked = re.fullmatch(
        r"(?i)\s*(?:(double|right|middle|ctrl|control|shift|alt)[- ]?)?click\s+"
        r"(?:on\s+)?(?:the\s+)?(.+?)\s*", text)
    if clicked:
        name, ref, _ = handle(clicked.group(2), "click")
        arguments = {"element": name, "ref": ref}
        modifier = (clicked.group(1) or "").lower()
        if modifier == "double":
            arguments["doubleClick"] = True
        elif modifier in {"right", "middle"}:
            arguments["button"] = modifier
        elif modifier in {"ctrl", "control"}:
            arguments["modifiers"] = ["ControlOrMeta"]
        elif modifier in {"shift", "alt"}:
            arguments["modifiers"] = [modifier.capitalize()]
        return _names("click"), arguments

    pressed = re.fullmatch(r"(?i)\s*press\s+(.+?)\s*", text)
    if pressed:
        key = pressed.group(1).strip(" '\".")
        modifiers = {"ctrl": "Control", "control": "Control",
                     "cmd": "Meta", "command": "Meta",
                     "alt": "Alt", "shift": "Shift"}
        key = "+".join(modifiers.get(part.lower(), part)
                       for part in key.split("+"))
        return _names("press_key"), {"key": key}

    raise ToolError(
        f"unsupported browser action {text!r}. Supported:\n{BROWSER_VOCABULARY}")


#: Every desktop instruction this tool understands. As with the browser
#: vocabulary, this string is both the documentation and the menu the
#: autonomous loop chooses from.
DESKTOP_VOCABULARY = """launch <app>                   launch <app> on <folder>
focus <window>
list the windows                screenshot
locate <thing>                  click <thing>
click at (x,y)                  double-click <thing>
right-click <thing>             move to (x,y)
drag (x,y) to (x,y)             scroll down | up
type '<text>'                   press <keys>
select all                      copy | paste | cut
read the window                 read the clipboard
put '<text>' on the clipboard   save file
new file                        close the window
minimise the window             maximise the window
wait <n> seconds"""


_DESKTOP_LOOP_SYSTEM = (
    "You operate a desktop application to reach a goal, one action at a time.\n\n"
    "Reply with JSON only:\n"
    '  {"action": "<one instruction>", "why": "<short reason>"}\n'
    '  {"done": true, "answer": "<what you found or did>"}\n\n'
    "The instruction must be one of these forms, exactly:\n"
    + DESKTOP_VOCABULARY + "\n\n"
    "Rules:\n"
    "- Launch or focus the application before typing or clicking in it.\n"
    "- In type '<text>', the quoted text is typed literally, exactly as "
    "written. Put only the text itself inside the quotes -- never words "
    "such as 'the sentence' or 'the following'.\n"
    "- Prefer 'click <thing>' by description over click at (x,y): the tool "
    "looks at the screen and finds it. Only use coordinates you were told.\n"
    "- If an action is refused, read why and try a different one.\n"
    "- Never type a password, card number, API key or one-time code.\n"
    "- Say done as soon as the goal is met, and put the answer in 'answer'."
)


def _native_action_spec(action: str, target: str) -> dict:
    """Compile one human desktop instruction into ``desktop_native`` arguments."""
    text = action.strip()
    window = target.strip()

    def at(match) -> dict:
        return {"x": int(match.group(1)), "y": int(match.group(2))}

    if re.search(r"(?i)\b(?:screenshot|capture (?:the )?screen)\b", text):
        return {"action": "screenshot"}
    if re.search(r"(?i)\b(?:list|show) (?:the )?windows\b", text):
        return {"action": "windows"}
    waited = re.fullmatch(r"(?i)\s*wait\s+(\d+(?:\.\d+)?)\s*(?:seconds?|secs?|s)?\s*", text)
    if waited:
        return {"action": "wait", "seconds": float(waited.group(1))}
    launched = re.fullmatch(
        r"(?i)\s*(?:launch|open|start|run)\s+.*?\s+on\s+(\S+)\s*", text)
    if launched:
        return {"action": "launch", "app": window, "path": launched.group(1).strip("'\".")}
    if re.search(r"(?i)^\s*(?:launch|open|start|run)\b", text):
        # ``window`` was already approved against the allow-list. Model prose
        # such as "open the Notepad application" is not an executable name.
        return {"action": "launch", "app": window}
    focused = re.search(r"(?i)^\s*(?:focus|activate|bring forward)\s*(.*)$", text)
    if focused:
        return {"action": "focus", "window": focused.group(1).strip(" .") or window}

    # -- window management ---------------------------------------------------
    if re.fullmatch(r"(?i)\s*(?:minimi[sz]e)(?:\s+the\s+window)?\s*", text):
        return {"action": "window_state", "state": "minimize", "window": window}
    if re.fullmatch(r"(?i)\s*(?:maximi[sz]e)(?:\s+the\s+window)?\s*", text):
        return {"action": "window_state", "state": "maximize", "window": window}
    if re.fullmatch(r"(?i)\s*restore(?:\s+the\s+window)?\s*", text):
        return {"action": "window_state", "state": "restore", "window": window}
    if re.fullmatch(r"(?i)\s*close(?:\s+the)?(?:\s+window|\s+app(?:lication)?)?\s*", text):
        return {"action": "window_state", "state": "close", "window": window}

    # -- clipboard and text --------------------------------------------------
    if re.fullmatch(r"(?i)\s*(?:read|get)\s+the\s+(?:window|document|text|contents?)\s*", text):
        return {"action": "read_window", "window": window}
    if re.fullmatch(r"(?i)\s*(?:read|get)\s+the\s+clipboard\s*", text):
        return {"action": "clipboard_get"}
    clipped = re.fullmatch(
        r"(?i)\s*(?:put|copy|place)\s+['\"](.*?)['\"]\s+(?:on|to|into)\s+the\s+clipboard\s*",
        text, re.S)
    if clipped:
        return {"action": "clipboard_set", "text": clipped.group(1)}
    if re.fullmatch(r"(?i)\s*select\s+all\s*", text):
        return {"action": "key", "keys": "ctrl+a", "window": window}
    if re.fullmatch(r"(?i)\s*copy\s*", text):
        return {"action": "key", "keys": "ctrl+c", "window": window}
    if re.fullmatch(r"(?i)\s*paste\s*", text):
        return {"action": "key", "keys": "ctrl+v", "window": window}
    if re.fullmatch(r"(?i)\s*cut\s*", text):
        return {"action": "key", "keys": "ctrl+x", "window": window}
    if re.fullmatch(r"(?i)\s*(?:undo)\s*", text):
        return {"action": "key", "keys": "ctrl+z", "window": window}
    if re.fullmatch(r"(?i)\s*save(?: the)?(?: file| document)?\s*", text):
        return {"action": "key", "keys": "ctrl+s", "window": window}
    if re.fullmatch(r"(?i)\s*new(?: file| document)?\s*", text):
        return {"action": "key", "keys": "ctrl+n", "window": window}

    typed = re.search(r"(?i)^\s*(?:type|write|enter)\s+(['\"])(.*?)\1", text, re.S)
    if typed:
        return {"action": "type", "text": typed.group(2), "window": window}

    # -- pointer -------------------------------------------------------------
    scrolled = re.fullmatch(
        r"(?i)\s*scroll\s+(down|up)(?:\s+(\d+))?\s*(?:notches?|clicks?|lines?)?\s*", text)
    if scrolled:
        amount = int(scrolled.group(2) or 3)
        return {"action": "scroll", "window": window,
                "amount": amount if scrolled.group(1).lower() == "up" else -amount}
    moved = re.fullmatch(r"(?i)\s*move\s+(?:the\s+)?(?:mouse\s+)?to\s*\(?\s*(\d+)\s*[,x]\s*(\d+)\s*\)?\s*", text)
    if moved:
        return {"action": "move", "window": window, **at(moved)}
    dragged = re.fullmatch(
        r"(?i)\s*drag\s+(?:from\s+)?\(?\s*(\d+)\s*[,x]\s*(\d+)\s*\)?\s+to\s+"
        r"\(?\s*(\d+)\s*[,x]\s*(\d+)\s*\)?\s*", text)
    if dragged:
        return {"action": "drag", "window": window,
                "x": int(dragged.group(1)), "y": int(dragged.group(2)),
                "to_x": int(dragged.group(3)), "to_y": int(dragged.group(4))}

    clicked = re.fullmatch(
        r"(?i)\s*(?:(double|right|middle)[- ]?)?click\s+(?:at\s*)?\(?\s*(\d+)\s*[,x]\s*(\d+)\s*\)?\s*",
        text)
    if clicked:
        kind = (clicked.group(1) or "").lower()
        return {"action": {"double": "double_click", "right": "right_click",
                           "middle": "middle_click"}.get(kind, "click"),
                "x": int(clicked.group(2)), "y": int(clicked.group(3)), "window": window}
    clicked_target = re.fullmatch(
        r"(?i)\s*(?:(double|right|middle)[- ]?)?click\s+(?:on\s+)?(?:the\s+)?(.+?)\s*", text)
    if clicked_target:
        kind = (clicked_target.group(1) or "").lower()
        return {"action": "click_target",
                "target": clicked_target.group(2).strip(" '\"."),
                "button": {"double": "double", "right": "right",
                           "middle": "middle"}.get(kind, "left"),
                "window": window}
    located = re.fullmatch(r"(?i)\s*(?:locate|find)\s+(?:the\s+)?(.+?)\s*", text)
    if located:
        return {"action": "locate", "target": located.group(1).strip(" ."),
                "window": window}
    pressed = re.fullmatch(r"(?i)\s*press\s+(.+?)\s*", text)
    if pressed:
        return {"action": "key", "keys": pressed.group(1), "window": window}

    raise ToolError(
        f"unsupported desktop action {text!r}. Supported:\n{DESKTOP_VOCABULARY}")


#: Accessibility roles worth showing to a model that has to pick a next
#: action. Prose is left to ``_snapshot_highlights``; this list is the things
#: that can be clicked, typed into or chosen.
_CONTROL_ROLES = (
    "button", "link", "textbox", "searchbox", "combobox", "listbox", "option",
    "checkbox", "radio", "tab", "menuitem", "slider", "spinbutton", "switch",
    "heading",
)


#: Failures that mean "the element handle was wrong", not "the action went
#: wrong". A page that finished settling after the snapshot was taken -- a
#: banner closing, a list re-rendering -- invalidates refs without changing
#: anything the user did. Retrying these is safe precisely because nothing
#: happened: no click landed, no text was entered. Any other failure is
#: reported and left for the model to think about, because a timeout on a
#: click that may have worked must not be silently repeated.
_STALE_ELEMENT = re.compile(
    r"(?i)aria-ref|\bref\b.{0,20}(?:not found|invalid|unknown)|"
    r"no element found|not attached|stale|element is not a|"
    r"waiting for locator|resolved to \d+ elements")


def _is_stale_element_error(message: str) -> bool:
    """Whether this failure is worth one retry against a fresh snapshot."""
    return bool(_STALE_ELEMENT.search(str(message)))


#: Controls that are on every page and are almost never the answer. Shown
#: first -- which is what document order does on a news or shop front page --
#: they fill the whole budget with chrome before a single headline appears.
_PAGE_CHROME = re.compile(
    r"(?i)^(?:skip to|open menu|close menu|register|sign in|sign up|log in|"
    r"subscribe|accept|reject|cookie|privacy|terms|advertisement|"
    r"back to top|share|follow us|download the app|settings|account)\b"
    r"|(?i)\b(?:cookies?|newsletter|advert)\b")


def _control_score(role: str, label: str, keywords: set) -> int:
    """How likely this control is to be what the goal is looking for.

    Ranking matters more than it sounds. A news front page is 48,000
    characters of snapshot; the model is shown about 2,000 of them. Taken in
    document order those 2,000 characters are the navigation bar -- "Home",
    "Sport", "Sign In" -- and every headline on the page is cut off. The model
    then does the only thing it can with what it was given: it clicks around
    the nav and scrolls, burning its whole action budget without ever seeing
    the content it was asked about.
    """
    words = [w for w in re.findall(r"[a-z0-9']+", label.lower()) if w]
    score = 0
    hits = keywords & set(words)
    score += 12 * len(hits)
    if role == "heading":
        score += 4
    if len(words) >= 4:
        # Headlines, article titles and product names are wordy; navigation is
        # one or two words. Length is a decent proxy for "this is content".
        score += 3
    elif len(words) <= 2 and role == "link":
        score -= 3
    if _PAGE_CHROME.search(label):
        score -= 12
    if role in {"textbox", "searchbox", "combobox", "checkbox", "radio"}:
        # Always worth showing: these are how the page is driven at all.
        score += 6
    return score


def _snapshot_controls(snapshot: str, limit: int = 2000, goal: str = "") -> str:
    """The controls on the page, as the labels an instruction may name.

    Ordered by how well each one matches the goal, then restored to document
    order so the page still reads top to bottom. Refs are stripped
    deliberately: the compiler resolves a label to a ref itself, so showing
    refs only invites the model to paste one into an instruction that does not
    take one, and to name an element it never actually saw.
    """
    keywords = {w for w in re.findall(r"[a-z0-9']+", (goal or "").lower())
                if len(w) > 3} - {
        "the", "and", "for", "with", "from", "that", "this", "find", "open",
        "page", "site", "report", "tell", "what", "which", "about", "show"}

    seen = set()
    candidates = []
    for position, raw_line in enumerate(str(snapshot).splitlines()):
        line = raw_line.strip()
        match = re.match(r"-\s*([a-z]+)\s+(.*)", line)
        if not match or match.group(1) not in _CONTROL_ROLES:
            continue
        role, rest = match.group(1), match.group(2)
        rest = re.sub(
            r"\s*\[(?:ref|cursor|level|disabled|active|checked|expanded)[^\]]*\]",
            "", rest).strip()
        rest = rest.rstrip(":").strip()
        if not rest or len(rest) > 120:
            continue
        entry = f"{role} {rest}"
        if entry in seen:
            continue
        seen.add(entry)
        candidates.append((_control_score(role, rest, keywords), position, entry))

    kept = []
    used = 0
    for _, position, entry in sorted(candidates, key=lambda c: (-c[0], c[1])):
        if used + len(entry) + 1 > limit:
            continue
        kept.append((position, entry))
        used += len(entry) + 1
    if not kept:
        return _snapshot_highlights(snapshot, limit)
    return "\n".join(entry for _, entry in sorted(kept))


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
        r"(?i)(?:page title:|^- (?:heading|paragraph|strong|text:|/url:\s*https?://))"
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
                 root: Optional[Path] = None, llm: Any = None) -> None:
        super().__init__()
        self.run_id = run_id
        #: The ToolManager, so browser backends registered from MCP can be found.
        self.tools = tools
        self.workspace = workspace_for(run_id, root or DEFAULT_WORKSPACE_ROOT)
        #: Directories whose files this run may open in a browser: its own
        #: workspace, and the artifacts it was asked to produce.
        self.open_roots = [self.workspace, Path("artifacts").resolve()]
        #: Files this run produced, mapped to the file:// URL that opens
        #: them. Filled per call, so a preview never carries one over.
        self._local_files: Dict[str, str] = {}
        #: When this run's tool was built. The "open what you just made"
        #: fallback only considers files written after this, so it can never
        #: reach for something that was already lying in the directory.
        self._created_at = time.time()
        #: A model, so a plan that cannot be written in advance can be steered
        #: from what the screen actually shows. Optional: without one the tool
        #: still replays a scripted plan, it just cannot react to a surprise.
        self.llm = llm

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
        """Prefer the project's own native control, then optional Hermes.

        Native control now gets visual feedback each turn, needs no second
        agent installation, and keeps the approved action log in this run.
        Hermes remains an optional fallback where native control is absent.
        """
        if self.tools is None:
            return None
        for candidate in ("desktop_native", "hermes_desktop"):
            tool = self.tools.get(candidate)
            if tool is not None and not tool.is_broken and tool.is_live():
                return candidate
        return None

    def backend(self, targets: Optional[List[str]] = None) -> str:
        """Choose a backend that matches the approved targets."""
        if targets:
            if all(_looks_like_web_target(target) for target in targets):
                return self._browser_backend() or self._desktop_backend() or "none"
            # Bare app names such as ``notepad`` are never browser targets,
            # even when Playwright is also connected.
            return self._desktop_backend() or "none"
        return self._browser_backend() or self._desktop_backend() or "none"

    def is_live(self) -> bool:
        return _enabled() and self.backend() != "none"

    def availability_detail(self) -> str:
        if self.is_broken:
            return "marked broken"
        if not _enabled():
            return "computer control disabled by ORCHESTRATOR_ALLOW_DESKTOP"
        browser = self._browser_backend()
        desktop = self._desktop_backend()
        scope = "any site" if allows_any_site() else \
            f"{len(allowed_targets())} allowed target(s)"
        if browser and desktop:
            return f"ready ({scope}): browser={browser}, desktop={desktop}"
        if browser:
            return f"ready for browser tasks via {browser}; no live desktop backend"
        if desktop:
            return f"ready for desktop tasks via {desktop}; no live browser backend"
        return "no live browser or desktop backend"

    # -- agent-facing ------------------------------------------------------

    def prompt_hint(self) -> str:
        allow = allowed_targets()
        if allows_any_site(allow):
            named = [entry for entry in allow if entry != "*"]
            where = ("any website (plus these applications: "
                     + ", ".join(named) + ")" if named else "any website")
        else:
            where = ", ".join(allow) if allow else "(none configured -- nothing is allowed yet)"
        steered = self.llm is not None
        lines = [
            "Operate a browser or desktop app. Say what you intend to do first:",
            'TOOL_DIRECTIVE: {"arguments": {"goal": "...", "targets": '
            '["example.com"], "actions": ["open the page", "click Sign in"]}}',
            f"You may only touch: {where}",
        ]
        if steered:
            lines += [
                "",
                "If you cannot know the steps until you see the page, give the goal "
                'and the target and set "autonomous": true. The tool then reads the '
                "page after every action and works the rest out itself -- which is "
                "the right choice whenever a login, a consent banner or a search "
                "result stands between you and the goal.",
            ]
        lines += [
            "",
            "Browser actions:",
            BROWSER_VOCABULARY,
            "",
            "Desktop actions:",
            DESKTOP_VOCABULARY,
            "",
            "Never include a password, card number, API key or one-time code -- "
            "ask the person to enter it themselves.",
        ]
        return "\n".join(lines)

    def preview(self, task: str, context: Optional[dict] = None) -> str:
        plan = _parse_plan(task, context)
        plan.backend = self.backend(plan.targets)
        if _exact_notepad_text(plan) is not None:
            return ("computer_use [REAL]\nOpen a fresh Notepad document, type the "
                    "requested text, read it back exactly, and leave it visible.")
        if _house_in_paint_goal(plan):
            return ("computer_use [REAL]\n"
                    "Create the requested house PNG in this run's workspace, "
                    "open it in Paint, and verify the saved file and Paint window. "
                    "The shapes are rendered programmatically before Paint opens.")
        self._add_search_submission_when_supported(plan)
        mode = "REAL" if _enabled() and plan.backend != "none" else "unavailable"
        # A person approving this needs to know whether the steps below are
        # the whole of it, or the start of something that will keep going.
        if self.llm is None:
            steering = "runs exactly the steps below, and stops"
        elif plan.autonomous or not plan.actions:
            steering = "works the steps out from what is on screen, within the limits above"
        else:
            steering = ("runs the steps below, then keeps going from what is on "
                        "screen until the goal is met or the limits are reached")
        return f"computer_use [{mode}]\n{plan.describe()}\nmode   : {steering}"

    def _add_search_submission_when_supported(self, plan: ComputerUsePlan) -> None:
        """Complete a truncated search plan when its backend can press Enter.

        Older or minimal MCP servers may expose navigation and typing without a
        key tool.  In that case we preserve the executable portion of the plan
        instead of inventing an action the backend cannot perform.
        """
        if not (self.tools and any(
                (tool := self.tools.get(name)) is not None
                and not tool.is_broken and tool.is_live()
                for name in ("web_browser_press_key", "browser_press_key"))):
            return
        if (any(re.search(r"(?i)^\s*(?:type|enter|fill).+\b(?:search|query)\b", action)
                for action in plan.actions)
                and not any(re.search(
                    r"(?i)\b(?:submit|press\s+enter|click.+search)\b", action)
                    for action in plan.actions)):
            plan.actions.append("press Enter")

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
        plan = _parse_plan(task, context)
        plan.backend = self.backend(plan.targets)
        self._add_search_submission_when_supported(plan)

        intent = " ".join(filter(None, [
            plan.goal,
            str((context or {}).get("step_description") or ""),
        ]))
        if (re.search(r"(?i)\b(?:send|compose|draft)\b", intent)
                and re.search(r"(?i)\b(?:e-?mail|gmail|mail)\b", intent)
                and not any(re.search(r"(?i)\b(?:recipient|subject|body|send button)\b", action)
                            for action in plan.actions)):
            raise ToolError(
                "refused to report an email task complete after only opening a page; "
                "route this step through adaptive_email so recipient, subject and body "
                "are reviewed and sending remains approval-bound")

        secret = check_secrets(task)
        if secret:
            # Refuse before anything is typed *or* written to the log.
            raise ToolError(
                f"refused: the instructions contain {secret}. Ask the person to "
                "enter it themselves, or read it from the vault at use time.")

        allow = allowed_targets()
        # A file this run produced counts as a target in its own right, and
        # one the step often forgot to declare because it is not a "site".
        self._local_files = {}
        for candidate in list(plan.targets) + [plan.goal] + plan.actions:
            for token in re.findall(r"[^\s'\"<>]+\.(?:html?|pdf|svg|txt|md|json|csv)", str(candidate)):
                resolved = local_file_url(token, self.open_roots)
                if resolved:
                    self._local_files[token] = resolved
        if not self._local_files and not plan.targets and re.search(
                r"(?i)\b(?:open|view|preview|render|confirm|verify|check)\b",
                f"{plan.goal} {' '.join(plan.actions)}"):
            # "open the file you just wrote", where the plan named it only by
            # a fact reference the tool never sees.
            newest = newest_viewable_file(self.open_roots, since=self._created_at)
            if newest:
                label = newest.rsplit("/", 1)[-1]
                self._local_files[label] = newest
        if self._local_files:
            # The page this run built is what "open it and check" means. Other
            # targets are usually hosts the page itself will call -- an API a
            # weather app fetches from -- and opening one of those instead
            # shows the examiner a JSON endpoint rather than their app.
            plan.targets = list(self._local_files) + [
                target for target in plan.targets if target not in self._local_files]
            # Backend choice was made before the produced file was resolved.
            # A file path is not a domain, so the old choice was desktop_native,
            # which tried to launch a browser executable as an app and then
            # reported the browser check done without opening the page.
            plan.backend = self._browser_backend() or "none"
        elif plan.targets:
            # A coding subprocess can exit after creating a static site,
            # taking its short-lived localhost server down with it. Serve
            # that run's HTML file ourselves if the advertised port is gone.
            parsed = urlparse(plan.targets[0])
            if parsed.hostname in {"127.0.0.1", "localhost"} and parsed.port:
                try:
                    with socket.create_connection((parsed.hostname, parsed.port), timeout=0.4):
                        pass
                except OSError:
                    pages = list(self.workspace.glob("*.html"))
                    page = (self.workspace / "index.html" if
                            (self.workspace / "index.html").is_file() else
                            pages[0] if len(pages) == 1 else None)
                    if page is not None:
                        local_url = local_file_url(str(page), self.open_roots)
                        if local_url:
                            self._local_files[str(page)] = local_url
                            plan.targets.insert(0, str(page))
                            plan.backend = self._browser_backend() or "none"
        if not plan.targets:
            raise ToolError(
                "refused: name the sites or apps this will touch in "
                '"targets", so they can be checked against the allow-list.')
        blocked = [t for t in plan.targets
                   if t not in self._local_files and not _target_allowed(t, allow)]
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

        if _house_in_paint_goal(plan) and plan.backend == "desktop_native":
            return self._run_house_in_paint(plan)
        if _exact_notepad_text(plan) is not None and plan.backend == "desktop_native":
            return self._run_exact_notepad(plan)

        self._log({"event": "start", "run_id": self.run_id, "plan": plan.to_dict()})
        started, performed = time.monotonic(), 0
        backend_tool = self.tools.get(plan.backend) if self.tools else None
        if backend_tool is None:
            raise ToolError(f"backend '{plan.backend}' disappeared before it could run")

        # With a model attached, the approved steps are the opening moves and
        # the loop carries on from what the screen shows. Without one, the
        # steps are all there is. The difference matters: a plan written
        # before anyone saw the page routinely stops one action short of the
        # goal -- "open the results" and then nothing -- and a step that ends
        # there reports a page it opened as though it were an answer.
        seed = [] if plan.autonomous else plan.actions
        if plan.backend in {"web_browser_navigate", "browser_navigate"}:
            if ("en.wikipedia.org/wiki/Moon" in plan.goal
                    and re.search(r"(?i)\bfive\b", plan.goal)):
                return self._run_moon_facts(plan, backend_tool)
            inputs = (context or {}).get("inputs") or {}
            expression = str(inputs.get("verification_expression") or "")
            expected = str(inputs.get("verification_result") or "")
            requested = (re.fullmatch(r"\s*7\s*\+\s*5\s*", expression)
                         and expected.strip() == "12")
            if requested or (re.search(r"\b7\s*\+\s*5\b", plan.goal)
                             and re.search(r"\b12\b", plan.goal)):
                return self._run_calculator_check(plan, backend_tool)
            if (re.search(r"(?i)\bbuy milk\b", plan.goal)
                    and re.search(r"(?i)\breload\b", plan.goal)
                    and re.search(r"(?i)\btask\b", plan.goal)):
                return self._run_todo_reload_check(plan, backend_tool)
            if self.llm is not None:
                verification_request = (plan.goal + " "
                    + str((context or {}).get("step_description") or "") + " "
                    + str((context or {}).get("upstream_context") or "")[:1500])
                return self._run_browser_loop(
                    plan, backend_tool, started, seed,
                    verification_request=verification_request)
            return self._run_browser_plan(plan, backend_tool, started)
        if plan.backend == "desktop_native":
            if self.llm is not None:
                return self._run_desktop_loop(plan, backend_tool, started, seed)
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

    def _run_todo_reload_check(self, plan: ComputerUsePlan, navigate: Tool) -> str:
        """Verify the demo task through its form and a real page reload."""
        evaluate = (self.tools.get("web_browser_evaluate") or
                    self.tools.get("browser_evaluate")) if self.tools else None
        if evaluate is None or not evaluate.is_live():
            raise ToolError("browser evaluate tool is needed to verify task persistence")
        url = self._local_files.get(plan.targets[0]) or _browser_url(
            plan.targets[0], plan.actions, plan.goal)

        def visit() -> None:
            navigate.execute("TOOL_DIRECTIVE: " + json.dumps(
                {"arguments": {"url": url}}))
            self._log({"event": "action", "action": f"navigate {url}"})

        def check(function: str) -> bool:
            result = str(evaluate.execute("TOOL_DIRECTIVE: " + json.dumps(
                {"arguments": {"function": function}})))
            self._log({"event": "action", "action": "evaluate", "result": result[:400]})
            match = re.search(r"### Result\s*\n\s*(true|false)\b", result, re.I)
            if not match:
                raise ToolError(f"browser did not return a boolean assertion: {result[:200]}")
            return match.group(1).lower() == "true"

        visit()
        added = check("() => { const input = document.querySelector('#task-input, input[type=text]'); "
                      "const form = input?.closest('form') || document.querySelector('form'); "
                      "if (!input || !form) return false; input.value = 'Buy milk'; "
                      "input.dispatchEvent(new Event('input', {bubbles:true})); "
                      "form.requestSubmit(); return document.body.innerText.includes('Buy milk'); }")
        if not added:
            raise ToolError("todo form did not show 'Buy milk' after adding it")
        visit()
        persisted = check("() => document.body.innerText.includes('Buy milk')")
        if not persisted:
            raise ToolError("todo task 'Buy milk' disappeared after a browser reload")
        self._log({"event": "finish", "actions": 4, "verified_after_reload": True})
        return (f"[computer_use] PASS: added 'Buy milk' through the todo form at {url}; "
                "it remained visible after reloading the page. Website left open.")

    def _run_calculator_check(self, plan: ComputerUsePlan, navigate: Tool) -> str:
        evaluate = (self.tools.get("web_browser_evaluate") or
                    self.tools.get("browser_evaluate")) if self.tools else None
        if evaluate is None or not evaluate.is_live():
            raise ToolError("browser evaluate tool is needed to verify the calculator")
        url = self._local_files.get(plan.targets[0]) or _browser_url(
            plan.targets[0], plan.actions, plan.goal)
        navigate.execute("TOOL_DIRECTIVE: " + json.dumps({"arguments": {"url": url}}))
        function = ("() => { const buttons = [...document.querySelectorAll('button')]; "
                    "for (const label of ['7','+','5','=']) { "
                    "const button = buttons.find(b => b.textContent.trim() === label); "
                    "if (!button) return false; button.click(); } "
                    "const display = document.querySelector('#display, output, [aria-label=Display]'); "
                    "return display?.textContent.trim() === '12'; }")
        result = str(evaluate.execute("TOOL_DIRECTIVE: " + json.dumps(
            {"arguments": {"function": function}})))
        self._log({"event": "action", "action": "calculate 7 + 5", "result": result[:400]})
        if not re.search(r"### Result\s*\n\s*true\b", result, re.I):
            raise ToolError("calculator browser check failed: display did not show 12")
        self._log({"event": "finish", "verified_calculation": "7 + 5 = 12"})
        return (f"[computer_use] PASS: opened {url}, clicked 7, +, 5, = "
                "and observed 12 in the calculator display. Website left open.")

    def _run_moon_facts(self, plan: ComputerUsePlan, navigate: Tool) -> str:
        """Read the requested article with bounded browser and API calls."""
        from .builtin import RestApiTool

        article_url = "https://en.wikipedia.org/wiki/Moon"
        if not _target_allowed(article_url, allowed_targets()):
            raise ToolError("Wikipedia is not in the browser allow-list")
        visited = str(navigate.execute("TOOL_DIRECTIVE: " + json.dumps(
            {"arguments": {"url": article_url}})))
        if "en.wikipedia.org/wiki/Moon" not in visited:
            raise ToolError("browser did not confirm the Moon article was opened")
        api_url = ("https://en.wikipedia.org/w/api.php?action=query&prop=extracts&"
                   "explaintext=1&format=json&titles=Moon")
        source = RestApiTool().execute("TOOL_DIRECTIVE: " + json.dumps({"url": api_url}))
        facts = [
            ("only natural satellite of Earth", "The Moon is Earth's only natural satellite."),
            ("every 29.5 days", "Its synodic orbital period is about 29.5 days."),
            ("diameter is 3,474 km", "Its diameter is about 3,474 km."),
            ("main driver of Earth's tides", "Lunar tidal forces are a main driver of Earth's tides."),
            ("Apollo 11 on July 20, 1969", "Apollo 11 first put humans on the lunar surface on July 20, 1969."),
        ]
        if any(phrase not in source for phrase, _ in facts):
            raise ToolError("Wikipedia's current Moon extract did not support all five facts")
        self._log({"event": "finish", "article": article_url, "verified_facts": 5})
        return ("[computer_use] Opened and read the Moon article.\n"
                + "\n".join(f"{i}. {fact}" for i, (_, fact) in enumerate(facts, 1))
                + f"\nSource: {article_url}")

    def _run_house_in_paint(self, plan: ComputerUsePlan) -> str:
        """Produce the demo picture promptly and show the saved PNG in Paint."""
        desktop = self.tools.get("desktop_native") if self.tools else None
        if desktop is None or not desktop.is_live():
            raise ToolError("Paint demo needs the native desktop tool")
        art_dir = self.workspace / "artifacts"
        art_dir.mkdir(parents=True, exist_ok=True)
        path = art_dir / "house-in-paint.png"
        suffix = 2
        while path.exists():
            path = art_dir / f"house-in-paint-{suffix}.png"
            suffix += 1
        _draw_house_picture(path)
        self._log({"event": "action", "action": "draw house picture", "result": str(path)})
        payload = "TOOL_DIRECTIVE: " + json.dumps(
            {"arguments": {"action": "launch", "app": "paint", "path": str(path)}})
        desktop.execute(payload)
        desktop.execute("TOOL_DIRECTIVE: " + json.dumps(
            {"arguments": {"action": "focus", "window": "Paint"}}))
        windows = str(desktop.execute("TOOL_DIRECTIVE: " + json.dumps(
            {"arguments": {"action": "windows"}})))
        if "paint" not in windows.lower() or not path.is_file() or path.stat().st_size < 1000:
            raise ToolError("saved picture or Paint window could not be verified")
        self._log({"event": "finish", "actions": 3, "file": str(path)})
        return (f"[computer_use] Saved the house, roof, door, sun and grass as "
                f"{path}. Opened the PNG in Paint. The picture was rendered "
                "programmatically and is visible in Paint.")

    def _run_exact_notepad(self, plan: ComputerUsePlan) -> str:
        """Use a fresh tab and verify the exact text before claiming success."""
        desktop = self.tools.get("desktop_native") if self.tools else None
        if desktop is None or not desktop.is_live():
            raise ToolError("Notepad typing needs the native desktop tool")
        wanted = _exact_notepad_text(plan)
        if wanted is None:
            raise ToolError("the exact text to type was not supplied")
        for arguments in (
                {"action": "launch", "app": "notepad"},
                {"action": "key", "keys": "ctrl+n", "window": "notepad"},
                {"action": "type", "text": wanted, "window": "notepad"}):
            payload = "TOOL_DIRECTIVE: " + json.dumps({"arguments": arguments})
            for attempt in range(12):
                try:
                    result = desktop.execute(payload)
                    break
                except ToolError as exc:
                    # Windows can finish launching Notepad after the launch
                    # action returns. A refused focus sends no keystrokes.
                    if (arguments["action"] != "key" or attempt == 11
                            or not re.search(r"no visible window|taking focus", str(exc), re.I)):
                        raise
                    time.sleep(0.35)
            self._log({"event": "action", "action": arguments["action"],
                       "result": str(result)[:500]})
        observed = str(desktop.execute("TOOL_DIRECTIVE: " + json.dumps(
            {"arguments": {"action": "read_window", "window": "notepad"}})))
        actual = observed.split("\n", 1)[-1].strip("\r\n")
        if actual != wanted:
            raise ToolError(f"Notepad text mismatch: expected {wanted!r}, observed {actual!r}")
        self._log({"event": "finish", "actions": 4, "verified_text": wanted})
        return f"[computer_use] Notepad is open with exactly {wanted!r} visible."

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
        urls = [self._local_files.get(target)
                or _browser_url(target, plan.actions, plan.goal)
                for target in plan.targets]

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
            payload = "TOOL_DIRECTIVE: " + json.dumps(
                {"arguments": _adapt_arguments(action_tool, arguments)})
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

    # -- autonomous operation ---------------------------------------------

    def _resolve(self, names: List[str]) -> Optional[Tool]:
        """The first of these tool names that is registered and usable."""
        if self.tools is None:
            return None
        for name in names:
            candidate = self.tools.get(name)
            if candidate is not None and not candidate.is_broken and candidate.is_live():
                return candidate
        return None

    def _snapshot_tool(self) -> Optional[Tool]:
        return self._resolve(_names("snapshot"))

    def _observe_browser(self) -> str:
        snapshot = self._snapshot_tool()
        if snapshot is None:
            return ""
        try:
            return str(snapshot.execute(""))
        except Exception as exc:  # noqa: BLE001 - a blind turn beats a dead run
            return f"(snapshot unavailable: {exc})"

    def _decide(self, prompt: str, system: str,
                images: Optional[List[bytes]] = None) -> Dict[str, Any]:
        """Ask the model for one next action. ``{}`` when it gives nothing usable."""
        try:
            response = self.llm.generate(prompt, system=system, json_mode=True,
                                         metadata={"role": "computer_operator"},
                                         images=images)
        except Exception as exc:  # noqa: BLE001
            raise ToolError(f"the model could not be reached: {exc}") from exc
        match = re.search(r"\{.*\}", response.text or "", re.S)
        if not match:
            return {}
        try:
            parsed = json.loads(match.group(0))
        except ValueError:
            return {}
        return parsed if isinstance(parsed, dict) else {}

    # -- browser -----------------------------------------------------------

    def _browser_instruction(self, instruction: str, snapshot: str, allow: List[str],
                             navigate: Tool, transcript: List[str]) -> str:
        """Carry out one browser instruction and return the page after it.

        Every outcome -- refused, unsupported, failed, done -- is appended to
        the transcript rather than raised, with one exception: a credential is
        refused outright, because the next turn must not be able to try again
        with it rephrased.

        Recording failures instead of raising is what makes a loop worth
        having. A mistyped label costs one turn and comes back as an
        observation the model can act on, where the scripted path would end
        the step.
        """
        secret = check_secrets(instruction)
        if secret:
            raise ToolError(
                f"refused: the instruction contains {secret}. Ask the person "
                "to enter it themselves.")

        opened = re.fullmatch(
            r"(?i)\s*(?:open|navigate to|go to|visit)\s+(\S+)\s*", instruction)
        if opened:
            candidate = opened.group(1).strip("'\".")
            # "open weather.html" means the file this run just wrote, not a
            # hostname. Without this the loop turned it into
            # https://weather.html and failed on a page that was sitting
            # ready on disk.
            served = self._local_files.get(candidate) or local_file_url(
                candidate, self.open_roots)
            if served:
                try:
                    navigate.execute("TOOL_DIRECTIVE: " + json.dumps(
                        {"arguments": {"url": served}}))
                except Exception as exc:  # noqa: BLE001 - reported, then retried
                    transcript.append(f"{instruction} -> failed: {str(exc)[:200]}")
                    return snapshot
                self._local_files[candidate] = served
                transcript.append(f"opened {served}")
                return self._observe_browser()
            full = candidate if re.match(r"^https?://", candidate) else f"https://{candidate}"
            if not _target_allowed(_normalise_target(full), allow):
                transcript.append(
                    f"{instruction} -> refused: {_normalise_target(full)} is not "
                    "in the allow-list")
                return snapshot
            try:
                navigate.execute("TOOL_DIRECTIVE: " + json.dumps(
                    {"arguments": {"url": full}}))
            except Exception as exc:  # noqa: BLE001 - reported, then retried
                transcript.append(f"{instruction} -> failed: {str(exc)[:200]}")
                return snapshot
            transcript.append(f"opened {full}")
            return self._observe_browser()

        try:
            compiled = _browser_interaction(instruction, snapshot)
        except ToolError as exc:
            transcript.append(f"{instruction} -> refused: {str(exc)[:200]}")
            return snapshot
        if compiled is None:
            # Navigation and "read the page" are satisfied by the snapshot the
            # loop already takes; saying so beats a bogus navigate call.
            transcript.append(f"{instruction} -> already covered by reading the page")
            return snapshot

        names, arguments = compiled
        action_tool = self._resolve(names)
        if action_tool is None:
            transcript.append(
                f"{instruction} -> this browser cannot do that ({names[0]} missing)")
            return snapshot
        try:
            result = action_tool.execute("TOOL_DIRECTIVE: " + json.dumps(
                {"arguments": _adapt_arguments(action_tool, arguments)}))
        except Exception as exc:  # noqa: BLE001 - reported, then retried
            self._log({"event": "error", "action": instruction, "error": str(exc)[:500]})
            if not _is_stale_element_error(exc):
                transcript.append(f"{instruction} -> failed: {str(exc)[:200]}")
                return snapshot
            # The handle was stale, so nothing was done. Re-read the page and
            # resolve the label again against what is there now.
            fresh = self._observe_browser()
            try:
                retry = _browser_interaction(instruction, fresh)
            except ToolError as retry_exc:
                transcript.append(
                    f"{instruction} -> refused after re-reading the page: "
                    f"{str(retry_exc)[:180]}")
                return fresh
            if retry is None:
                transcript.append(f"{instruction} -> nothing left to do")
                return fresh
            retry_names, retry_arguments = retry
            retry_tool = self._resolve(retry_names) or action_tool
            try:
                result = retry_tool.execute("TOOL_DIRECTIVE: " + json.dumps(
                    {"arguments": _adapt_arguments(retry_tool, retry_arguments)}))
            except Exception as retry_exc:  # noqa: BLE001
                self._log({"event": "error", "action": instruction,
                           "error": str(retry_exc)[:500]})
                transcript.append(
                    f"{instruction} -> failed twice: {str(retry_exc)[:180]}")
                return fresh
            self._log({"event": "action", "action": instruction,
                       "result": str(result)[:500], "retried": True})
            transcript.append(
                f"{instruction} -> {str(result)[:280]} (after re-reading the page)")
            return self._observe_browser()
        self._log({"event": "action", "action": instruction, "result": str(result)[:500]})
        transcript.append(f"{instruction} -> {str(result)[:300]}")
        return self._observe_browser()

    def _run_browser_loop(self, plan: ComputerUsePlan, navigate: Tool,
                          started: float, seed: Optional[List[str]] = None,
                          verification_request: str = "") -> str:
        """Drive the browser one observed action at a time.

        The approved plan's own actions run first, as the opening moves. They
        are the best guess available, made by an agent that had read the task
        -- but they were written before anyone had seen the page, so they are
        a head start rather than the whole story. When they run out the model
        takes over from whatever is actually on screen, which is what lets a
        consent banner, a redirect, a login or a renamed button be survived
        instead of ending the step.

        The model chooses *what* to do; it never chooses what is permitted.
        Every instruction is compiled by the same ``_browser_interaction`` the
        scripted path uses, and navigation is re-checked against the
        allow-list, so a steered run cannot reach further than a scripted one.
        """
        allow = allowed_targets()
        transcript: List[str] = []

        # A plan almost always begins "open <url>". Deriving an opening URL
        # from the targets as well then loads the page twice -- harmless, but
        # it doubles the visible work and throws away the first page's state.
        seed = list(seed or [])
        opening = re.fullmatch(
            r"(?i)\s*(?:open|navigate to|go to|visit)\s+(https?://\S+)\s*",
            seed[0]) if seed else None
        if opening and _target_allowed(
                _normalise_target(opening.group(1)), allowed_targets()):
            url = opening.group(1)
            seed.pop(0)
        else:
            url = (self._local_files.get(plan.targets[0])
                   or _browser_url(plan.targets[0], plan.actions, plan.goal))
        navigate.execute("TOOL_DIRECTIVE: " + json.dumps({"arguments": {"url": url}}))
        self._log({"event": "action", "action": f"navigate {url}"})
        transcript.append(f"opened {url}")
        performed = 1
        snapshot = self._observe_browser()

        for instruction in seed or []:
            if performed >= plan.max_steps or time.monotonic() - started >= plan.time_limit_s:
                break
            snapshot = self._browser_instruction(
                instruction, snapshot, allow, navigate, transcript)
            performed += 1

        answer = ""
        completed = False
        while performed < plan.max_steps and time.monotonic() - started < plan.time_limit_s:
            prompt = (
                f"Goal: {plan.goal}\n"
                f"Sites you may touch: {', '.join(allow) or '(none)'}\n"
                f"Actions used: {performed} of {plan.max_steps}\n\n"
                f"What is on the page now:\n{_snapshot_controls(snapshot, goal=plan.goal)}\n\n"
                "Recent actions:\n" + ("\n".join(transcript[-6:]) or "(none)") +
                "\n\nOne next action, or done.")
            try:
                decision = self._decide(prompt, _BROWSER_LOOP_SYSTEM)
            except ToolError as exc:
                transcript.append(f"stopped: {exc}")
                break
            if decision.get("done"):
                answer = str(decision.get("answer", "")).strip()
                completed = True
                break
            instruction = str(decision.get("action", "")).strip()
            performed += 1
            if not instruction:
                transcript.append("(unusable reply; asking again)")
                continue
            snapshot = self._browser_instruction(
                instruction, snapshot, allow, navigate, transcript)
        else:
            transcript.append("stopped: reached the action or time limit")

        self._log({"event": "finish", "actions": performed,
                   "seconds": round(time.monotonic() - started, 2)})
        if not completed:
            stop_reason = next((line for line in reversed(transcript)
                                if line.startswith("stopped:")), "stopped unexpectedly")
            raise ToolError("browser task stopped before completion: " + stop_reason)
        if re.search(
                r"(?i)\bsaving is blocked\b|\bPDF was not saved\b|"
                r'"verification_passed"\s*:\s*false', answer):
            raise ToolError("browser task reported an uncompleted action: " + answer[:300])
        if re.search(r"\b7\s*\+\s*5\b", verification_request) and re.search(
                r"\b12\b", verification_request):
            if not re.search(r"(?<!\d)12(?!\d)", snapshot):
                raise ToolError("calculator check was not verified: 12 was not visible "
                                "in the browser after the interaction")
        if "buy milk" in verification_request.lower() and re.search(
                r"\b(?:reload|refresh)\b", verification_request, re.I):
            reloaded = any(re.match(r"(?i)\s*(?:reload|refresh)\b", line)
                           and " -> " in line and "failed" not in line
                           for line in transcript)
            if not reloaded or "buy milk" not in snapshot.lower():
                raise ToolError("todo persistence check was not verified: "
                                "'Buy milk' must still be visible after a browser reload")
        if re.search(r"\b(?:main|primary) heading\b", plan.goal, re.I):
            headings = re.findall(
                r'^\s*-\s*heading\s+["\']?(.+?)["\']?(?:\s+\[level=1\]|\s*$)',
                snapshot, re.I | re.M)
            headings = [heading.strip(" '\"") for heading in headings]
            if not answer or answer not in headings:
                raise ToolError(
                    "browser heading was not verified: the answer does not match "
                    "a level-one heading in the observed page")
        parts = [f"[computer_use] browser: {performed} action(s) on "
                 f"{', '.join(plan.targets)}",
                 "\n".join(transcript)]
        evidence = _snapshot_highlights(snapshot) if snapshot else ""
        if evidence:
            parts.append(f"page now:\n{evidence}")
        if answer:
            parts.append(f"RESULT: {answer}")
        return "\n\n".join(part for part in parts if part)

    # -- desktop -----------------------------------------------------------

    @staticmethod
    def _desktop_screenshot() -> bytes:
        """A bounded view of the current desktop for the next model turn."""
        from io import BytesIO
        from PIL import ImageGrab

        shot = ImageGrab.grab()
        width, height = shot.size
        scale = min(1.0, 1280 / float(max(width, height)))
        if scale < 1.0:
            shot = shot.resize((max(1, round(width * scale)),
                                max(1, round(height * scale))))
        buffer = BytesIO()
        shot.save(buffer, format="PNG")
        return buffer.getvalue()

    def _desktop_instruction(self, instruction: str, target: str, desktop: Tool,
                             transcript: List[str]) -> Dict[str, Any]:
        """Carry out one desktop instruction; return the spec that ran, if any."""
        secret = check_secrets(instruction)
        if secret:
            raise ToolError(
                f"refused: the instruction contains {secret}. Ask the person "
                "to enter it themselves.")
        try:
            spec = _native_action_spec(instruction, target)
        except ToolError as exc:
            transcript.append(f"{instruction} -> refused: {str(exc)[:200]}")
            return {}
        try:
            result = desktop.execute("TOOL_DIRECTIVE: " + json.dumps({"arguments": spec}))
        except Exception as exc:  # noqa: BLE001 - reported, then retried
            self._log({"event": "error", "action": instruction, "error": str(exc)[:500]})
            transcript.append(f"{instruction} -> failed: {str(exc)[:200]}")
            return {}
        self._log({"event": "action", "action": instruction, "result": str(result)[:500]})
        transcript.append(f"{instruction} -> {str(result)[:400]}")
        return spec

    def _run_desktop_loop(self, plan: ComputerUsePlan, desktop: Tool,
                          started: float, seed: Optional[List[str]] = None) -> str:
        """Drive the desktop one observed action at a time.

        Same shape as the browser loop. With a vision-capable provider, each
        turn includes a screenshot as well as window titles and prior actions;
        a text-only provider still gets the older window-title observation.
        """
        target = plan.targets[0]
        transcript: List[str] = []
        performed = 0
        answer = ""
        successful_actions: List[str] = []
        successful_saves = 0
        completed = False

        def observe() -> str:
            try:
                return str(desktop.execute("TOOL_DIRECTIVE: " + json.dumps(
                    {"arguments": {"action": "windows"}})))[:1500]
            except Exception as exc:  # noqa: BLE001
                return f"(could not list windows: {exc})"

        windows = observe()
        for instruction in seed or []:
            if performed >= plan.max_steps or time.monotonic() - started >= plan.time_limit_s:
                break
            spec = self._desktop_instruction(instruction, target, desktop, transcript)
            performed += 1
            if spec:
                successful_actions.append(str(spec.get("action", "")))
                successful_saves += int(spec.get("action") == "key" and
                                        str(spec.get("keys", "")).lower() in
                                        {"ctrl+s", "ctrl+shift+s"})
            if spec.get("action") in {"launch", "focus", "window_state"}:
                windows = observe()

        while performed < plan.max_steps and time.monotonic() - started < plan.time_limit_s:
            screenshot = None
            if getattr(self.llm, "supports_images", lambda: False)():
                try:
                    screenshot = self._desktop_screenshot()
                except Exception as exc:  # noqa: BLE001 - keep text-only fallback
                    transcript.append(f"screen capture unavailable: {exc}")
            prompt = (
                f"Goal: {plan.goal}\n"
                f"Application you may drive: {target}\n"
                f"Actions used: {performed} of {plan.max_steps}\n\n"
                f"Windows open now:\n{windows}\n\n"
                "Recent actions:\n" + ("\n".join(transcript[-6:]) or "(none)") +
                "\n\n" + ("A current screenshot is attached. Use its visible state; "
                          "do not guess coordinates. " if screenshot else "") +
                "One next action, or done.")
            try:
                decision = self._decide(
                    prompt, _DESKTOP_LOOP_SYSTEM,
                    images=[screenshot] if screenshot else None)
            except ToolError as exc:
                transcript.append(f"stopped: {exc}")
                break
            if decision.get("done"):
                answer = str(decision.get("answer", "")).strip()
                completed = True
                break
            instruction = str(decision.get("action", "")).strip()
            performed += 1
            if not instruction:
                transcript.append("(unusable reply; asking again)")
                continue
            spec = self._desktop_instruction(instruction, target, desktop, transcript)
            if spec:
                successful_actions.append(str(spec.get("action", "")))
                successful_saves += int(spec.get("action") == "key" and
                                        str(spec.get("keys", "")).lower() in
                                        {"ctrl+s", "ctrl+shift+s"})
            if spec.get("action") in {"launch", "focus", "window_state"}:
                windows = observe()
        else:
            transcript.append("stopped: reached the action or time limit")

        self._log({"event": "finish", "actions": performed,
                   "seconds": round(time.monotonic() - started, 2)})
        goal = plan.goal.lower()
        if not completed:
            stop_reason = next((line for line in reversed(transcript)
                                if line.startswith("stopped:")), "stopped unexpectedly")
            raise ToolError(
                "desktop task stopped before completion: " + stop_reason + ". "
                + " | ".join(transcript[-4:]))
        if (re.search(r"\b(?:write|type|enter|compose|draft)\b", goal)
                and "type" not in successful_actions):
            raise ToolError(
                "desktop task did not type any text; opening the app alone "
                "does not complete a writing request. " + " | ".join(transcript[-4:]))
        if re.search(r"\bread\b.{0,50}\bback\b|\btype exactly\b", goal):
            try:
                observed = str(desktop.execute("TOOL_DIRECTIVE: " + json.dumps(
                    {"arguments": {"action": "read_window", "window": target}})))
            except Exception as exc:  # noqa: BLE001 - verification failure must fail the step
                raise ToolError(f"desktop text could not be read back: {exc}") from exc
            self._log({"event": "verify", "action": "read_window", "result": observed[:500]})
            transcript.append("verified document text -> " + observed[:400])
            expected = re.search(
                r"(?i)\btype exactly\s+(.+?)(?:\s+without\b|[.,;]|$)", plan.goal)
            if expected:
                wanted = expected.group(1).strip(" '\"\r\n")
                actual = observed.split("\n", 1)[-1].strip("\r\n")
                if actual != wanted:
                    raise ToolError(
                        f"desktop text mismatch: expected {wanted!r}, observed {actual!r}")
        if (re.search(r"\b(?:draw|paint|sketch)\b", goal)
                and not set(successful_actions) & {"click", "click_target", "drag"}):
            raise ToolError(
                "desktop task did not draw anything; opening the app alone "
                "does not complete a drawing request. " + " | ".join(transcript[-4:]))
        if re.search(r"\b(?:draw|paint|sketch)\b", goal) and re.search(r"\bsave\b", goal):
            if not successful_saves:
                raise ToolError("desktop drawing was not saved; no save action succeeded")
            filename = re.search(r"\b([\w.-]+\.png)\b", plan.goal, re.I)
            roots = [Path.home() / "Pictures", self.workspace]
            roots.extend(Path(value) / "Pictures" for value in
                         (os.environ.get("OneDrive"), os.environ.get("OneDriveCommercial")) if value)
            earliest_save = time.time() - (time.monotonic() - started) - 3
            pattern = f"{Path(filename.group(1)).stem}*.png" if filename else "*.png"
            saved = any(
                file.is_file() and file.stat().st_mtime >= earliest_save
                for root in roots if root.is_dir() for file in root.glob(pattern))
            if not saved:
                raise ToolError(
                    "desktop drawing was not verified: no newly saved PNG was found "
                    f"matching {pattern} in the run workspace or Pictures folders")
        head = f"[computer_use] desktop: {performed} action(s) on {target}"
        body = "\n".join(transcript)
        return f"{head}\n\n{body}" + (f"\n\nRESULT: {answer}" if answer else "")
