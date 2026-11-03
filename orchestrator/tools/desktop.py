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
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import Tool, ToolError, extract_directive
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


_DPI_READY = False


def _ensure_dpi_aware() -> None:
    """Make this process see the screen in the same pixels the screenshot does.

    Windows lies about the screen to programs that have not declared
    themselves DPI-aware. On a display at 125% scaling ``GetSystemMetrics``
    reports 1536x864 while ``ImageGrab`` captures the real 1920x1080. The
    vision model is shown the capture, so it answers in physical pixels, and
    the click is then issued in logical ones -- every coordinate lands 1.25x
    away from what the model was looking at. Nothing errors; the pointer
    simply arrives somewhere else, which in Paint drew a house a quarter too
    large and a quarter too far down.

    Declaring awareness once, before any screen work, makes measurement,
    capture and pointer all agree. Failures are ignored: the call is refused
    if awareness was already set (which is the outcome we wanted anyway), and
    absent on very old Windows.
    """
    global _DPI_READY
    if _DPI_READY or os.name != "nt":
        return
    _DPI_READY = True
    import ctypes

    try:
        # 2 = per-monitor aware; correct when monitors differ in scaling.
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
        return
    except Exception:  # noqa: BLE001 - older Windows, or already set
        pass
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:  # noqa: BLE001 - nothing more to try
        pass


def _win32():
    """Import pywin32 lazily so the module imports on non-Windows machines."""
    try:
        import win32api
        import win32con
        import win32gui
        import win32process
        # Every screen action reaches win32 through here, so this is the one
        # place guaranteed to run before anything is measured or clicked.
        _ensure_dpi_aware()
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


def window_process(handle: int) -> str:
    """Executable owning a listed top-level window, when accessible."""
    win32api, win32con, _, win32process = _win32()
    try:
        _, pid = win32process.GetWindowThreadProcessId(handle)
        process = win32api.OpenProcess(
            win32con.PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        try:
            return os.path.basename(win32process.GetModuleFileNameEx(process, 0)).lower()
        finally:
            win32api.CloseHandle(process)
    except Exception:  # noqa: BLE001 - inaccessible windows are not candidates
        return ""


def _clipboard_get() -> str:
    """The clipboard as text. Empty when it holds something else."""
    import win32clipboard

    win32clipboard.OpenClipboard()
    try:
        if not win32clipboard.IsClipboardFormatAvailable(win32clipboard.CF_UNICODETEXT):
            return ""
        return str(win32clipboard.GetClipboardData(win32clipboard.CF_UNICODETEXT) or "")
    finally:
        win32clipboard.CloseClipboard()


def _clipboard_set(text: str) -> None:
    import win32clipboard

    win32clipboard.OpenClipboard()
    try:
        win32clipboard.EmptyClipboard()
        win32clipboard.SetClipboardText(text, win32clipboard.CF_UNICODETEXT)
    finally:
        win32clipboard.CloseClipboard()


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

    def _locate_point(self, target: str, allow: List[str],
                      window: str = "") -> tuple[int, int, str, int]:
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

        _ensure_dpi_aware()
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
            raise ToolError(
                f"not on screen: {target!r} -- "
                f"{found.get('why', 'the model did not say why')}")

        # Back to real screen pixels, which is what click expects.
        x = int(round(float(found.get("x", 0)) / scale))
        y = int(round(float(found.get("y", 0)) / scale))
        x = max(0, min(x, full_w - 1))
        y = max(0, min(y, full_h - 1))
        return x, y, str(found.get("what", "")), response.usage.total_tokens

    def _locate(self, target: str, allow: List[str], window: str = "") -> str:
        x, y, what, tokens = self._locate_point(target, allow, window)
        return f"{target!r} is at ({x},{y}) -- {what} [{tokens} tok]"

    def _click_target(self, target: str, allow: List[str], window: str = "",
                      button: str = "left") -> str:
        x, y, what, tokens = self._locate_point(target, allow, window)
        if button in {"right", "middle", "double"}:
            clicked = self._click_variant(
                x, y, allow, window,
                "left" if button == "double" else button, double=button == "double")
        else:
            clicked = self._click(x, y, allow, window)
        return f"located {target!r} as {what!r} [{tokens} tok]; {clicked}"

    def is_live(self) -> bool:
        if not _enabled() or os.name != "nt":
            return False
        try:
            _win32()
            return True
        except ToolError:
            return False

    def availability_detail(self) -> str:
        if self.is_broken:
            return "marked broken"
        if not _enabled():
            return "native desktop disabled by ORCHESTRATOR_ALLOW_DESKTOP"
        if os.name != "nt":
            return "native desktop requires Windows"
        try:
            _win32()
        except ToolError as exc:
            return str(exc)
        return "Windows native control ready"

    def prompt_hint(self) -> str:
        allow = _allow_list()
        where = ", ".join(allow) if allow else "(nothing configured yet)"
        return (
            "Control the desktop. One action per call:\n"
            'TOOL_DIRECTIVE: {"arguments": {"action": "screenshot"}}\n'
            "  actions: screenshot | windows | focus | launch | locate | click |\n"
            "           click_target | double_click | right_click | middle_click |\n"
            "           move | drag | scroll | type | key | window_state |\n"
            "           read_window | clipboard_get | clipboard_set | wait\n"
            '  focus   {"action":"focus","window":"Notepad"}\n'
            '  launch  {"action":"launch","app":"notepad"}\n'
            '  click   {"action":"click","x":400,"y":300}\n'
            '  find    {"action":"click_target","target":"the Save button"}\n'
            '  type    {"action":"type","text":"hello"}\n'
            '  key     {"action":"key","keys":"ctrl+s"}\n'
            '  read    {"action":"read_window","window":"Notepad"}\n'
            '  window  {"action":"window_state","state":"maximize"}\n'
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
        return extract_directive(task)

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
        if not (_window_allowed(settled, [window.lower()])
                or _window_allowed(process, [window.lower()])):
            raise ToolError(
                f"tried to bring {window!r} forward but {settled or '(unknown)'!r} "
                f"({process or 'unknown'}) is in front instead -- target mismatch.")
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

        _ensure_dpi_aware()

        self.workspace.mkdir(parents=True, exist_ok=True)
        path = self.workspace / f"screen-{int(time.time())}.png"
        ImageGrab.grab().save(path)
        return f"screenshot saved to {path.name} (front window: {foreground_window()!r})"

    def _focus(self, wanted: str, allow: List[str]) -> str:
        if not wanted.strip():
            raise ToolError("no window named for focus")
        # Launch commonly leaves the new application in front. Keep that
        # exact instance instead of enumerating same-named windows and
        # accidentally selecting an older document (for example, another
        # Notepad tab/window). Both title and process are checked because a
        # fresh editor may still have the generic title "Untitled".
        current_title = foreground_window()
        current_process = foreground_process()
        if ((_window_allowed(current_title, [wanted.lower()])
                or _window_allowed(current_process, [wanted.lower()]))
                and (_window_allowed(current_title, allow)
                     or _window_allowed(current_process, allow))):
            return f"focused {current_title!r}"

        win32api, win32con, win32gui, _ = _win32()
        for window in list_windows():
            handle = window["handle"]
            process = window_process(handle)
            if (not (_window_allowed(window["title"], [wanted.lower()])
                     or _window_allowed(process, [wanted.lower()]))
                    or not (_window_allowed(window["title"], allow)
                            or _window_allowed(process, allow))):
                continue
            try:
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
                settled, settled_process = foreground_window(), foreground_process()
                if ((_window_allowed(settled, [wanted.lower()])
                     or _window_allowed(settled_process, [wanted.lower()]))
                        and (_window_allowed(settled, allow)
                             or _window_allowed(settled_process, allow))):
                    return f"focused {settled!r}"
            except Exception:  # noqa: BLE001 - another window may still work
                continue
        raise ToolError(f"no visible window matching {wanted!r}")

    def _launch(self, app: str, allow: List[str], path: str = "") -> str:
        """Start an application, optionally on one of this run's own folders.

        Editors are only useful pointed at something, so a single path may be
        passed -- but only one this run owns. It is resolved and required to
        sit inside the workspace or artifacts directory, so "launch the editor
        on the project you just wrote" works while "launch it on anything
        else on this disk" does not. Arguments are passed as a list with no
        shell, so nothing in the path can be read as a command.
        """
        if not _window_allowed(app, allow):
            raise ToolError(f"refused: {app!r} is not in the allow-list")
        if not re.fullmatch(r"[A-Za-z0-9_.-]{1,60}", app or ""):
            raise ToolError("refused: app name must be a bare executable name")

        # The user-facing allow-list uses friendly application names. Windows
        # does not ship a `paint.exe`; its executable is `mspaint.exe`.
        executable_name = {"paint": "mspaint.exe"}.get(app.lower(), app)
        command = [executable_name]
        if path:
            roots = [self.workspace.resolve(), Path("artifacts").resolve()]
            candidates = [Path(path)] + [root / path for root in roots]
            opened = None
            for candidate in candidates:
                try:
                    resolved = candidate.resolve()
                except OSError:
                    continue
                if not resolved.exists():
                    continue
                if any(resolved == root or root in resolved.parents for root in roots):
                    opened = resolved
                    break
            if opened is None:
                raise ToolError(
                    f"refused: {path!r} is not inside this run's workspace or "
                    "artifacts directory")
            command.append(str(opened))

        # Resolve through PATH rather than handing a bare name to
        # CreateProcess, which cannot start the .cmd shims that several
        # editors install ("code" is code.cmd on Windows). The resolved path
        # comes from the system, not from the model, and arguments stay a
        # list, so no shell parsing is introduced either way.
        executable = shutil.which(command[0])
        if executable is None:
            raise ToolError(
                f"could not start {app!r}: it is not on PATH")
        command[0] = executable
        if executable.lower().endswith((".cmd", ".bat")):
            command = ["cmd.exe", "/c", *command]
        try:
            subprocess.Popen(command, shell=False)
        except OSError as exc:
            raise ToolError(f"could not start {app!r}: {exc}") from exc
        time.sleep(2.0 if path else 1.2)
        where = f" on {command[-1]}" if path else ""
        return f"launched {app!r}{where} (front window: {foreground_window()!r})"

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

    def _click_variant(self, x: int, y: int, allow: List[str], window: str,
                       button: str, double: bool = False) -> str:
        """Right, middle and double clicks, on the same guarded path as left.

        Routed through ``_ensure_target`` like every other click: a
        right-click opens a context menu wherever the focus happens to be,
        which is exactly the kind of stray action the foreground check exists
        to prevent.
        """
        win32api, win32con, _, _ = _win32()
        title = self._ensure_target(window, allow)
        width, height = win32api.GetSystemMetrics(0), win32api.GetSystemMetrics(1)
        if not (0 <= x < width and 0 <= y < height):
            raise ToolError(f"refused: ({x},{y}) is outside the screen {width}x{height}")
        down, up = {
            "left": (win32con.MOUSEEVENTF_LEFTDOWN, win32con.MOUSEEVENTF_LEFTUP),
            "right": (win32con.MOUSEEVENTF_RIGHTDOWN, win32con.MOUSEEVENTF_RIGHTUP),
            "middle": (win32con.MOUSEEVENTF_MIDDLEDOWN, win32con.MOUSEEVENTF_MIDDLEUP),
        }[button]
        win32api.SetCursorPos((x, y))
        time.sleep(0.05)
        for _ in range(2 if double else 1):
            win32api.mouse_event(down, 0, 0, 0, 0)
            win32api.mouse_event(up, 0, 0, 0, 0)
            if double:
                time.sleep(0.05)
        kind = "double-clicked" if double else f"{button}-clicked"
        return f"{kind} ({x},{y}) in {title!r}"

    def _move(self, x: int, y: int, allow: List[str], window: str = "") -> str:
        win32api, _, _, _ = _win32()
        title = self._ensure_target(window, allow)
        width, height = win32api.GetSystemMetrics(0), win32api.GetSystemMetrics(1)
        if not (0 <= x < width and 0 <= y < height):
            raise ToolError(f"refused: ({x},{y}) is outside the screen {width}x{height}")
        win32api.SetCursorPos((x, y))
        return f"moved the pointer to ({x},{y}) in {title!r}"

    def _drag(self, x: int, y: int, to_x: int, to_y: int, allow: List[str],
              window: str = "") -> str:
        win32api, win32con, _, _ = _win32()
        title = self._ensure_target(window, allow)
        width, height = win32api.GetSystemMetrics(0), win32api.GetSystemMetrics(1)
        for point in ((x, y), (to_x, to_y)):
            if not (0 <= point[0] < width and 0 <= point[1] < height):
                raise ToolError(
                    f"refused: {point} is outside the screen {width}x{height}")
        win32api.SetCursorPos((x, y))
        time.sleep(0.05)
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
        # Moved in steps rather than jumped: applications that implement drag
        # with mousemove handlers ignore a single teleporting cursor.
        for index in range(1, 11):
            win32api.SetCursorPos((x + (to_x - x) * index // 10,
                                   y + (to_y - y) * index // 10))
            time.sleep(0.02)
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
        return f"dragged ({x},{y}) to ({to_x},{to_y}) in {title!r}"

    def _scroll(self, amount: int, allow: List[str], window: str = "") -> str:
        win32api, win32con, _, _ = _win32()
        title = self._ensure_target(window, allow)
        notches = max(-20, min(int(amount), 20))
        win32api.mouse_event(win32con.MOUSEEVENTF_WHEEL, 0, 0, notches * 120, 0)
        return (f"scrolled {abs(notches)} notch(es) "
                f"{'up' if notches > 0 else 'down'} in {title!r}")

    def _window_state(self, state: str, allow: List[str], window: str = "") -> str:
        """Minimise, maximise, restore or close a window.

        Closing goes through ``WM_CLOSE`` rather than terminating the process,
        so the application still gets to ask about unsaved work instead of
        losing it.
        """
        _, win32con, win32gui, _ = _win32()
        wanted = window or foreground_window()
        handle = next((w["handle"] for w in list_windows()
                       if (_window_allowed(w["title"], [wanted.lower()])
                           or _window_allowed(window_process(w["handle"]), [wanted.lower()]))
                       and (_window_allowed(w["title"], allow)
                            or _window_allowed(window_process(w["handle"]), allow))), None)
        if handle is None:
            raise ToolError(f"no visible window matching {wanted!r}")
        if state == "close":
            win32gui.PostMessage(handle, win32con.WM_CLOSE, 0, 0)
            return f"asked {wanted!r} to close"
        win32gui.ShowWindow(handle, {
            "minimize": win32con.SW_MINIMIZE,
            "maximize": win32con.SW_MAXIMIZE,
            "restore": win32con.SW_RESTORE,
        }[state])
        return f"{state}d {wanted!r}"

    def _read_window(self, allow: List[str], window: str = "") -> str:
        """Read a window's text through select-all and copy.

        Win32 can read a title but not a document, so the only backend-neutral
        way to see what an editor contains is the clipboard. The person's own
        clipboard is saved and put back afterwards: reading something should
        not quietly destroy what they had copied.
        """
        title = self._ensure_target(window, allow)
        try:
            previous = _clipboard_get()
        except Exception:  # noqa: BLE001 - an empty or foreign clipboard
            previous = ""
        self._key("ctrl+a", allow, window)
        time.sleep(0.1)
        self._key("ctrl+c", allow, window)
        time.sleep(0.25)
        try:
            body = _clipboard_get()
        finally:
            if previous:
                try:
                    _clipboard_set(previous)
                except Exception:  # noqa: BLE001
                    pass
        return f"{title!r} contains {len(body)} character(s):\n{body[:4000]}"

    def _type(self, text: str, allow: List[str], window: str = "",
              method: str = "auto") -> str:
        """Put exact text into the focused control.

        Pasting is the default, and the reason is worth stating plainly:
        synthesised keystrokes are not reliable for text.
        ``KEYEVENTF_UNICODE`` events carry their character in ``wScan``, and
        the window has to turn each keydown into a ``WM_CHAR`` before the next
        one arrives. When injection outruns the application's message pump --
        which it does, at any pacing that is not painfully slow -- the pending
        events all resolve to the most recent character. The failure is not a
        dropped keystroke, which would be obvious. It is the right *number* of
        characters with the wrong *content*: typing "the quick brown fox"
        lands as "the nnnnnnnnnnn ggggggggggg", and a step that then saves the
        file has written rubbish and reported success.

        A paste is one message and cannot half-arrive. The person's clipboard
        is saved and put back, because borrowing it is not the same as taking
        it. ``method="keystrokes"`` forces the old path for the rare control
        that refuses paste.
        """
        _win32()  # fail with the normal dependency diagnostic before acting
        title = self._ensure_target(window, allow)
        exact = text[:2000]
        if not exact:
            return f"typed nothing into {title!r}"

        if method not in {"auto", "paste", "keystrokes"}:
            raise ToolError(
                f"unknown typing method {method!r}; use auto, paste or keystrokes")

        if method in {"auto", "paste"}:
            try:
                return self._type_by_paste(exact, allow, window, title)
            except ToolError:
                if method == "paste":
                    raise
                # Fall through: a control that refuses paste is better served
                # by slow keystrokes than by a failed step.

        self._send_unicode(exact)
        time.sleep(min(5.0, max(0.1, len(exact) * 0.02)))
        return (f"typed {len(exact)} character(s) into {title!r} as keystrokes "
                "(unverified: synthesised typing can corrupt long text)")

    def _type_by_paste(self, exact: str, allow: List[str], window: str,
                       title: str) -> str:
        """Paste text through the clipboard, then give the clipboard back."""
        try:
            previous = _clipboard_get()
        except Exception:  # noqa: BLE001 - empty, or holding something foreign
            previous = ""
        try:
            _clipboard_set(exact)
        except Exception as exc:  # noqa: BLE001
            raise ToolError(f"could not reach the clipboard: {exc}") from exc
        try:
            self._key("ctrl+v", allow, window)
            # Long enough for the target to act on the paste before the
            # clipboard is handed back to its owner.
            time.sleep(max(0.25, min(2.0, len(exact) / 2000.0)))
        finally:
            if previous:
                try:
                    _clipboard_set(previous)
                except Exception:  # noqa: BLE001
                    pass
        return f"typed {len(exact)} character(s) into {title!r}"

    #: Keystrokes per ``SendInput`` call, and the pause between calls.
    #: Injected input is delivered faster than an application consumes it, and
    #: a burst that overruns the target does not simply arrive late -- it
    #: arrives *wrong*. Sending "the quick brown fox" in one call reproducibly
    #: lands as "the nnnnnnnnnnn ggggggggggg" in Windows 11 Notepad: the right
    #: number of characters, the wrong ones. Verified against real Notepad:
    #: one batch fails, one character with no pause still fails, and 10
    #: characters every 10ms is correct every time.
    _TYPE_CHUNK_CHARS = 10
    _TYPE_CHUNK_PAUSE_S = 0.01

    @classmethod
    def _send_unicode(cls, text: str) -> None:
        """Inject exact Unicode text without depending on keyboard state.

        ``VkKeyScan`` synthesizes Shift and is affected by Caps Lock/layout;
        in real Notepad audits that changed case and even dropped characters.
        ``SendInput`` with ``KEYEVENTF_UNICODE`` sends UTF-16 code units
        directly and leaves the person's clipboard untouched.

        Sent in paced batches -- see ``_TYPE_CHUNK_CHARS`` for why that is not
        optional.
        """
        import ctypes
        from ctypes import wintypes

        ulong_ptr = wintypes.WPARAM

        class MouseInput(ctypes.Structure):
            _fields_ = [
                ("dx", wintypes.LONG), ("dy", wintypes.LONG),
                ("mouseData", wintypes.DWORD), ("dwFlags", wintypes.DWORD),
                ("time", wintypes.DWORD), ("dwExtraInfo", ulong_ptr),
            ]

        class KeyboardInput(ctypes.Structure):
            _fields_ = [
                ("wVk", wintypes.WORD), ("wScan", wintypes.WORD),
                ("dwFlags", wintypes.DWORD), ("time", wintypes.DWORD),
                ("dwExtraInfo", ulong_ptr),
            ]

        class HardwareInput(ctypes.Structure):
            _fields_ = [
                ("uMsg", wintypes.DWORD), ("wParamL", wintypes.WORD),
                ("wParamH", wintypes.WORD),
            ]

        class InputValue(ctypes.Union):
            _fields_ = [("mi", MouseInput), ("ki", KeyboardInput),
                        ("hi", HardwareInput)]

        class Input(ctypes.Structure):
            _anonymous_ = ("value",)
            _fields_ = [("type", wintypes.DWORD), ("value", InputValue)]

        # Declared rather than left to ctypes' defaults, so a refused call
        # reports the Windows error instead of a bare count mismatch.
        user32 = ctypes.WinDLL("user32", use_last_error=True)
        user32.SendInput.argtypes = (wintypes.UINT, ctypes.POINTER(Input),
                                     ctypes.c_int)
        user32.SendInput.restype = wintypes.UINT

        encoded = text.encode("utf-16-le")
        units = [int.from_bytes(encoded[i:i + 2], "little")
                 for i in range(0, len(encoded), 2)]
        if not units:
            return

        key_up, unicode_key = 0x0002, 0x0004
        for start in range(0, len(units), cls._TYPE_CHUNK_CHARS):
            batch = units[start:start + cls._TYPE_CHUNK_CHARS]
            events = []
            for unit in batch:
                events.append(Input(type=1, ki=KeyboardInput(
                    0, unit, unicode_key, 0, 0)))
                events.append(Input(type=1, ki=KeyboardInput(
                    0, unit, unicode_key | key_up, 0, 0)))
            array = (Input * len(events))(*events)
            sent = user32.SendInput(len(array), array, ctypes.sizeof(Input))
            if sent != len(array):
                raise ToolError(
                    f"Windows accepted only {sent} of {len(array)} keyboard "
                    f"events (error {ctypes.get_last_error()}); "
                    f"{start} character(s) were already typed")
            time.sleep(cls._TYPE_CHUNK_PAUSE_S)

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
            raise ToolError(
                'desktop_native received no valid "action" in TOOL_DIRECTIVE; '
                "nothing was done")

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
                result = self._launch(str(spec.get("app", "")), allow,
                                      str(spec.get("path", "")))
            elif action == "click":
                result = self._click(int(spec.get("x", -1)), int(spec.get("y", -1)), allow,
                                     str(spec.get("window", "")))
            elif action == "click_target":
                result = self._click_target(str(spec.get("target", "")), allow,
                                            str(spec.get("window", "")),
                                            str(spec.get("button", "left")))
            elif action in {"double_click", "right_click", "middle_click"}:
                result = self._click_variant(
                    int(spec.get("x", -1)), int(spec.get("y", -1)), allow,
                    str(spec.get("window", "")),
                    "left" if action == "double_click" else action.split("_")[0],
                    double=action == "double_click")
            elif action == "move":
                result = self._move(int(spec.get("x", -1)), int(spec.get("y", -1)),
                                    allow, str(spec.get("window", "")))
            elif action == "drag":
                result = self._drag(
                    int(spec.get("x", -1)), int(spec.get("y", -1)),
                    int(spec.get("to_x", -1)), int(spec.get("to_y", -1)),
                    allow, str(spec.get("window", "")))
            elif action == "scroll":
                result = self._scroll(int(spec.get("amount", -3)), allow,
                                      str(spec.get("window", "")))
            elif action == "window_state":
                result = self._window_state(
                    str(spec.get("state", "restore")).lower(), allow,
                    str(spec.get("window", "")))
            elif action == "read_window":
                result = self._read_window(allow, str(spec.get("window", "")))
            elif action == "clipboard_get":
                result = f"the clipboard holds: {_clipboard_get()[:4000]}"
            elif action == "clipboard_set":
                _clipboard_set(str(spec.get("text", "")))
                result = f"put {len(str(spec.get('text', '')))} character(s) on the clipboard"
            elif action == "wait":
                seconds = max(0.0, min(float(spec.get("seconds", 1)), 30.0))
                time.sleep(seconds)
                result = f"waited {seconds:.1f}s"
            elif action == "type":
                result = self._type(str(spec.get("text", "")), allow,
                                    str(spec.get("window", "")),
                                    str(spec.get("method", "auto")).lower())
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
