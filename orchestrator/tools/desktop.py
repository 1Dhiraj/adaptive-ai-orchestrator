"""Native Windows desktop control: screenshots, windows, mouse, keyboard.

This is the backend behind ``computer_use`` when the work is not in a browser
and Hermes is not installed. It needs nothing beyond pywin32 and Pillow,
which the project already depends on, so "control the computer" does not
require a second agent with its own account and API key.

It is also the most dangerous thing in this codebase, and is written that
way on purpose. Raw clicks and keystrokes go wherever the focus happens to
be, so the guard that matters is not "which app did you ask for" but **which
window is actually in front right now**: every click and keystroke re-checks
the foreground window against the allow-list immediately before acting. A
step allowed to drive Notepad cannot type into a banking tab that stole
focus a moment earlier.

The rest of the rules are the same ones ``computer_use`` enforces -- opt-in,
approval, limits, logging -- repeated here rather than assumed, because a
tool this sharp should not be safe only by virtue of its caller.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import Tool, ToolError
from .workspace import DEFAULT_WORKSPACE_ROOT, workspace_for

__all__ = ["DesktopTool", "foreground_window", "list_windows"]

#: Key names an agent can use, mapped to Windows virtual-key codes.
_KEYS = {
    "enter": 0x0D, "return": 0x0D, "tab": 0x09, "esc": 0x1B, "escape": 0x1B,
    "space": 0x20, "backspace": 0x08, "delete": 0x2E, "del": 0x2E,
    "home": 0x24, "end": 0x23, "pageup": 0x21, "pagedown": 0x22,
    "up": 0x26, "down": 0x28, "left": 0x25, "right": 0x27,
    "ctrl": 0x11, "control": 0x11, "alt": 0x12, "shift": 0x10, "win": 0x5B,
    **{f"f{i}": 0x70 + i - 1 for i in range(1, 13)},
}
_MODIFIERS = {"ctrl", "control", "alt", "shift", "win"}


def _enabled() -> bool:
    return os.environ.get("ORCHESTRATOR_ALLOW_DESKTOP", "").strip().lower() in {
        "1", "true", "yes", "on"}


def _win32():
    """Import pywin32 lazily so the module imports on non-Windows machines."""
    try:
        import win32api
        import win32con
        import win32gui
        import win32process
        return win32api, win32con, win32gui, win32process
    except ImportError as exc:  # pragma: no cover - platform dependent
        raise ToolError(
            "native desktop control needs pywin32 (pip install pywin32)") from exc


def list_windows() -> List[Dict[str, Any]]:
    """Visible, titled top-level windows."""
    _, _, win32gui, _ = _win32()
    found: List[Dict[str, Any]] = []

    def collect(handle, _):
        if not win32gui.IsWindowVisible(handle):
            return
        title = win32gui.GetWindowText(handle)
        if title.strip():
            found.append({"handle": handle, "title": title})

    win32gui.EnumWindows(collect, None)
    return found


def foreground_window() -> str:
    """Title of the window currently in front. Empty when there is none."""
    _, _, win32gui, _ = _win32()
    try:
        return win32gui.GetWindowText(win32gui.GetForegroundWindow()) or ""
    except Exception:  # noqa: BLE001
        return ""


def foreground_process() -> str:
    """Executable owning the front window, e.g. ``notepad.exe``.

    Titles alone are too brittle to gate on: Notepad's own file dialog is
    titled "Open", which matches no sensible allow-list entry, so a run
    permitted to drive Notepad would be blocked the moment it opened a file.
    The owning process does not change when a dialog appears.
    """
    win32api, win32con, win32gui, win32process = _win32()
    try:
        handle = win32gui.GetForegroundWindow()
        _, pid = win32process.GetWindowThreadProcessId(handle)
        process = win32api.OpenProcess(
            win32con.PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        return os.path.basename(win32process.GetModuleFileNameEx(process, 0)).lower()
    except Exception:  # noqa: BLE001 - absence is handled by the caller
        return ""


def _allow_list() -> List[str]:
    raw = os.environ.get("ORCHESTRATOR_COMPUTER_USE_ALLOW", "")
    return [item.strip().lower() for item in re.split(r"[,\s]+", raw) if item.strip()]


def _window_allowed(title: str, allow: List[str]) -> bool:
    """Whether a window title belongs to something the run may touch.

    Matched loosely on purpose: real titles are "Untitled - Notepad" and
    "report.docx - Word", so an allow-list entry is a substring of the title
    rather than the whole of it.
    """
    haystack = (title or "").lower()
    return bool(haystack) and any(entry in haystack for entry in allow)


class DesktopTool(Tool):
    """Screenshot, focus, click and type on the real Windows desktop."""

    name = "desktop_native"
    capability = "desktop_native"
    description = ("Control the Windows desktop directly: screenshot, list and "
                   "focus windows, click, type and press keys")
    fallbacks: List[str] = []
    side_effect = True
    #: Clicks and keystrokes land on a real machine and cannot be undone.
    irreversible = True

    def __init__(self, run_id: str = "default", root: Optional[Path] = None,
                 llm: Any = None) -> None:
        super().__init__()
        self.run_id = run_id
        self.workspace = workspace_for(run_id, root or DEFAULT_WORKSPACE_ROOT)
        #: A vision model, for finding things on screen by description instead
        #: of by hard-coded coordinates. Optional: without it the tool still
        #: works, it just cannot answer "where is the Save button?".
        self.llm = llm

    # -- vision ------------------------------------------------------------

    def _locate(self, target: str, allow: List[str], window: str = "") -> str:
        """Find something on screen by describing it, not by coordinates.

        This is how Claude- and Codex-style computer use works, and why it is
        not tied to particular applications: it reads pixels rather than an
        accessibility tree, so a game, an Electron app and a native dialog are
        all equally legible.

        The screenshot is downscaled before sending and the coordinates it
        returns are scaled back up. A 1920x1080 PNG is a lot of tokens for a
        question that does not need that much detail, and every step of a
        vision loop pays it.
        """
        if self.llm is None:
            raise ToolError("no model is attached, so nothing can be located on screen")
        if not getattr(self.llm, "supports_images", lambda: False)():
            raise ToolError(
                f"the configured model cannot see images. Point the run at a vision "
                f"model, e.g. OPENAI_BASE_URL=https://openrouter.ai/api/v1 with "
                f"OPENAI_MODEL=minimax/minimax-m3.")
        if not target.strip():
            raise ToolError('refused: say what to find, e.g. {"target": "the Save button"}')

        if window:
            self._ensure_target(window, allow)
        else:
            self._require_allowed_foreground(allow)

        from io import BytesIO

        from PIL import ImageGrab

        shot = ImageGrab.grab()
        full_w, full_h = shot.size
        scale = min(1.0, 1280 / float(full_w))
        small = shot.resize((int(full_w * scale), int(full_h * scale))) if scale < 1 else shot
        buffer = BytesIO()
        small.save(buffer, format="PNG")

        prompt = (
            f"Screenshot of a {small.size[0]}x{small.size[1]} screen.\n"
            f"Find: {target}\n\n"
            "Reply with JSON only:\n"
            '{"found": true, "x": <int>, "y": <int>, "what": "<what you see there>"}\n'
            'If it is not visible: {"found": false, "why": "<reason>"}\n'
            "Coordinates must be the centre of the element, in the image's own "
            "pixel space."
        )
        response = self.llm.generate(
            prompt,
            system="You locate interface elements in screenshots. JSON only.",
            json_mode=True, images=[buffer.getvalue()],
            metadata={"role": "screen_locator"})

        match = re.search(r"\{.*\}", response.text or "", re.S)
        if not match:
            raise ToolError(f"the model did not return coordinates: {response.text[:200]!r}")
        try:
            found = json.loads(match.group(0))
        except ValueError as exc:
            raise ToolError(f"unreadable coordinates from the model: {exc}") from exc

        if not found.get("found"):
            return (f"not on screen: {target!r} -- "
                    f"{found.get('why', 'the model did not say why')}")

        # Back to real screen pixels, which is what click expects.
        x = int(round(float(found.get("x", 0)) / scale))
        y = int(round(float(found.get("y", 0)) / scale))
        x = max(0, min(x, full_w - 1))
        y = max(0, min(y, full_h - 1))
        return (f"{target!r} is at ({x},{y}) -- {found.get('what', '')} "
                f"[{response.usage.total_tokens} tok]")

    def is_live(self) -> bool:
        if not _enabled() or os.name != "nt":
            return False
        try:
            _win32()
            return True
        except ToolError:
            return False

    def prompt_hint(self) -> str:
        allow = _allow_list()
        where = ", ".join(allow) if allow else "(nothing configured yet)"
        return (
            "Control the desktop. One action per call:\n"
            'TOOL_DIRECTIVE: {"arguments": {"action": "screenshot"}}\n'
            '  actions: screenshot | windows | focus | launch | click | type | key\n'
            '  focus   {"action":"focus","window":"Notepad"}\n'
            '  launch  {"action":"launch","app":"notepad"}\n'
            '  click   {"action":"click","x":400,"y":300}\n'
            '  type    {"action":"type","text":"hello"}\n'
            '  key     {"action":"key","keys":"ctrl+s"}\n'
            f"You may only act on: {where}\n"
            "Always set \"window\" so the right app is raised first, whatever the "
            "user is doing. Take a screenshot before clicking anywhere. "
            "Never type a password, card number or one-time code.")

    def preview(self, task: str, context: Optional[dict] = None) -> str:
        spec = self._directive(task)
        mode = "REAL" if self.is_live() else "simulated"
        detail = {k: v for k, v in spec.items() if k != "text"}
        if "text" in spec:
            detail["text"] = spec["text"][:60]
        return (f"desktop_native [{mode}]: {json.dumps(detail)}\n"
                f"  front window now: {foreground_window() or '(none)'}")

    # -- internals ---------------------------------------------------------

    @staticmethod
    def _directive(task: str) -> Dict[str, Any]:
        match = re.search(r"TOOL_DIRECTIVE:\s*(\{.*\})\s*$", task or "", re.S | re.M)
        if not match:
            return {}
        try:
            parsed = json.loads(match.group(1))
        except ValueError:
            return {}
        if isinstance(parsed, dict):
            return parsed.get("arguments", parsed) or {}
        return {}

    def _log(self, entry: Dict[str, Any]) -> None:
        try:
            self.workspace.mkdir(parents=True, exist_ok=True)
            with (self.workspace / "desktop.log.jsonl").open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({"at": round(time.time(), 3), **entry}) + "\n")
        except OSError:
            pass

    def _ensure_target(self, window: str, allow: List[str]) -> str:
        """Put the intended window in front, then confirm it got there.

        A step that says "type this into Notepad" should not fail because the
        user clicked Chrome a moment earlier. When it names its target we
        focus that window first -- which checks the allow-list on the way --
        and only then act.

        This is also the safer order. Checking what happens to be in front and
        then typing leaves a gap in which focus can move; raising the intended
        window ourselves and re-checking closes it, because we know what we
        put there.
        """
        if not window:
            return self._require_allowed_foreground(allow)

        self._focus(window, allow)           # refuses if not in the allow-list
        settled = foreground_window()
        process = foreground_process()
        if not (_window_allowed(settled, allow) or _window_allowed(process, allow)):
            raise ToolError(
                f"tried to bring {window!r} forward but {settled or '(unknown)'!r} "
                f"({process or 'unknown'}) is in front instead -- something else "
                "is taking focus, so nothing was typed or clicked.")
        return f"{settled} [{process}]" if process else settled

    def _require_allowed_foreground(self, allow: List[str]) -> str:
        """Refuse to act unless the window in front is one we may touch.

        Checked immediately before every click or keystroke rather than once
        at the start: focus can change between actions, and input follows
        focus wherever it went.

        Either the title or the owning executable may satisfy the list, so
        "notepad" covers both the editor and its "Open" dialog, without
        widening the list enough to cover an unrelated window that happens to
        be called Open.
        """
        title = foreground_window()
        process = foreground_process()
        if not (_window_allowed(title, allow) or _window_allowed(process, allow)):
            raise ToolError(
                f"refused: the window in front is {title or '(unknown)'!r} "
                f"({process or 'unknown process'}), which is not in the allow-list "
                f"({', '.join(allow) or 'empty'}). Focus an allowed window first.")
        return f"{title} [{process}]" if process else title

    # -- actions -----------------------------------------------------------

    def _screenshot(self) -> str:
        from PIL import ImageGrab

        self.workspace.mkdir(parents=True, exist_ok=True)
        path = self.workspace / f"screen-{int(time.time())}.png"
        ImageGrab.grab().save(path)
        return f"screenshot saved to {path.name} (front window: {foreground_window()!r})"

    def _focus(self, wanted: str, allow: List[str]) -> str:
        win32api, win32con, win32gui, _ = _win32()
        if not _window_allowed(wanted, allow):
            raise ToolError(f"refused: {wanted!r} is not in the allow-list")
        for window in list_windows():
            if wanted.lower() in window["title"].lower():
                handle = window["handle"]
                if win32gui.IsIconic(handle):
                    win32gui.ShowWindow(handle, win32con.SW_RESTORE)
                # Windows refuses SetForegroundWindow from a background
                # process unless input is attached; alt-tap is the usual
                # workaround and is harmless when it is already allowed.
                try:
                    win32api.keybd_event(_KEYS["alt"], 0, 0, 0)
                    win32gui.SetForegroundWindow(handle)
                finally:
                    win32api.keybd_event(_KEYS["alt"], 0, 2, 0)
                time.sleep(0.3)
                return f"focused {window['title']!r}"
        raise ToolError(f"no visible window matching {wanted!r}")

    def _launch(self, app: str, allow: List[str]) -> str:
        if not _window_allowed(app, allow):
            raise ToolError(f"refused: {app!r} is not in the allow-list")
        if not re.fullmatch(r"[A-Za-z0-9_.-]{1,60}", app or ""):
            raise ToolError("refused: app name must be a bare executable name")
        try:
            subprocess.Popen([app], shell=False)
        except OSError as exc:
            raise ToolError(f"could not start {app!r}: {exc}") from exc
        time.sleep(1.2)
        return f"launched {app!r} (front window: {foreground_window()!r})"

    def _click(self, x: int, y: int, allow: List[str], window: str = "") -> str:
        win32api, win32con, _, _ = _win32()
        title = self._ensure_target(window, allow)
        width = win32api.GetSystemMetrics(0)
        height = win32api.GetSystemMetrics(1)
        if not (0 <= x < width and 0 <= y < height):
            raise ToolError(f"refused: ({x},{y}) is outside the screen {width}x{height}")
        win32api.SetCursorPos((x, y))
        time.sleep(0.05)
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
        return f"clicked ({x},{y}) in {title!r}"

    def _type(self, text: str, allow: List[str], window: str = "") -> str:
        win32api, _, _, _ = _win32()
        title = self._ensure_target(window, allow)
        for char in text[:2000]:
            code = win32api.VkKeyScan(char)
            if code == -1:
                continue
            vk, shift = code & 0xFF, (code >> 8) & 1
            if shift:
                win32api.keybd_event(_KEYS["shift"], 0, 0, 0)
            win32api.keybd_event(vk, 0, 0, 0)
            win32api.keybd_event(vk, 0, 2, 0)
            if shift:
                win32api.keybd_event(_KEYS["shift"], 0, 2, 0)
            time.sleep(0.01)
        return f"typed {len(text)} character(s) into {title!r}"

    def _key(self, combo: str, allow: List[str], window: str = "") -> str:
        win32api, _, _, _ = _win32()
        title = self._ensure_target(window, allow)
        parts = [p.strip().lower() for p in re.split(r"[+\-\s]+", combo) if p.strip()]
        unknown = [p for p in parts if p not in _KEYS and len(p) != 1]
        if unknown:
            raise ToolError(f"unknown key(s): {', '.join(unknown)}")

        codes = []
        for part in parts:
            if part in _KEYS:
                codes.append(_KEYS[part])
            else:
                scan = win32api.VkKeyScan(part)
                codes.append(scan & 0xFF if scan != -1 else 0)
        for code in codes:
            win32api.keybd_event(code, 0, 0, 0)
        for code in reversed(codes):
            win32api.keybd_event(code, 0, 2, 0)
        return f"pressed {'+'.join(parts)} in {title!r}"

    # -- entry point -------------------------------------------------------

    def _run(self, task: str, context: Optional[dict] = None) -> str:
        spec = self._directive(task)
        action = str(spec.get("action", "")).strip().lower()
        if not action:
            return ('[desktop_native] no "action" in TOOL_DIRECTIVE; nothing done')

        allow = _allow_list()
        if not allow:
            raise ToolError(
                "refused: ORCHESTRATOR_COMPUTER_USE_ALLOW is empty, so no window "
                "may be touched. Name the apps this run may drive.")

        if not self.is_live():
            return (f"[simulated:desktop_native] would {action} "
                    f"{json.dumps({k: v for k, v in spec.items() if k != 'action'})}\n"
                    "  (needs Windows and ORCHESTRATOR_ALLOW_DESKTOP=1)")

        try:
            if action == "screenshot":
                result = self._screenshot()
            elif action == "windows":
                titles = [w["title"] for w in list_windows()]
                result = f"{len(titles)} window(s): " + "; ".join(t[:60] for t in titles)
            elif action == "focus":
                result = self._focus(str(spec.get("window", "")), allow)
            elif action == "launch":
                result = self._launch(str(spec.get("app", "")), allow)
            elif action == "click":
                result = self._click(int(spec.get("x", -1)), int(spec.get("y", -1)), allow,
                                     str(spec.get("window", "")))
            elif action == "type":
                result = self._type(str(spec.get("text", "")), allow,
                                    str(spec.get("window", "")))
            elif action == "locate":
                result = self._locate(str(spec.get("target", "")), allow,
                                      str(spec.get("window", "")))
            elif action == "key":
                result = self._key(str(spec.get("keys", "")), allow,
                                   str(spec.get("window", "")))
            else:
                raise ToolError(f"unknown action {action!r}")
        except ToolError as exc:
            self._log({"action": action, "refused": str(exc)[:300]})
            raise

        self._log({"action": action, "result": result[:300]})
        return f"[desktop_native] {result}"
