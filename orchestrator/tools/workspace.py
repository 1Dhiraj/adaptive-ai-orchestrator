"""Let agents actually build software: a confined workspace and a terminal.

A "frontend agent" that only writes prose about React is not building
anything. To work as a team on real code, agents need to write files that the
next agent can read, and run commands to install, build and test.

Both are confined to one directory per run, and the terminal refuses a list of
commands that could damage the host or escalate privilege.

**Read this before enabling it.** A command blocklist is a guardrail, not a
security boundary. It stops accidents and obvious mistakes; it will not stop a
determined adversarial prompt, because shells offer endless ways to spell the
same thing. If you are running untrusted task descriptions, put the whole
process in a container or VM -- that is the only real isolation. This is why
the terminal is opt-in rather than registered by default.
"""

from __future__ import annotations

import os
import re
import shlex
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..config import PROJECT_ROOT
from .base import Tool, ToolError
from .builtin import _directive

DEFAULT_WORKSPACE_ROOT = PROJECT_ROOT / "workspace"

#: Commands refused outright. Matched on the first word of any statement in
#: the command line, so `foo && sudo bar` is caught as well as plain `sudo`.
BLOCKED_COMMANDS = {
    # privilege escalation
    "sudo", "su", "runas", "doas", "pkexec",
    # user / permission management
    "useradd", "usermod", "userdel", "net", "icacls", "takeown", "chown",
    "chmod", "setfacl", "passwd",
    # disk / system destruction
    "mkfs", "fdisk", "diskpart", "format", "dd", "shred", "wipefs",
    # power / process control of the host
    "shutdown", "reboot", "halt", "poweroff", "systemctl", "sc",
    # registry and firewall
    "reg", "regedit", "netsh", "bcdedit",
    # remote shells
    "ssh", "telnet", "nc", "ncat",
}

#: Patterns refused wherever they appear, for the shapes a blocklist of
#: command names alone would miss.
BLOCKED_PATTERNS = [
    (re.compile(r"\brm\s+(-[a-zA-Z]*\s+)*[-/]{0,2}\s*/(\s|$)"), "deleting the filesystem root"),
    (re.compile(r"\brm\s+-[a-zA-Z]*r[a-zA-Z]*f|\brm\s+-[a-zA-Z]*f[a-zA-Z]*r"), "rm -rf"),
    (re.compile(r":\(\)\s*\{.*\|.*&.*\}\s*;?\s*:"), "a fork bomb"),
    (re.compile(r"\bcurl\b[^|]*\|\s*(ba)?sh"), "piping a download straight into a shell"),
    (re.compile(r"\bwget\b[^|]*\|\s*(ba)?sh"), "piping a download straight into a shell"),
    (re.compile(r">\s*/dev/(sd|hd|nvme)"), "writing directly to a disk device"),
]

#: Environment variables never passed to a subprocess. An agent's shell has no
#: business reading the operator's API keys, and a leaked key in a build log is
#: a real incident.
SECRET_ENV_PATTERN = re.compile(
    r"(KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL|WEBHOOK|DSN|DATABASE_URL)", re.IGNORECASE)


class WorkspaceError(ToolError):
    """A path escaped the workspace, or a command was refused."""


def workspace_for(run_id: str, root: Optional[Path] = None) -> Path:
    """The directory one run is allowed to touch. Created on demand."""
    safe_run = re.sub(r"[^\w.\-]", "_", run_id or "default")
    path = (root or DEFAULT_WORKSPACE_ROOT) / safe_run
    path.mkdir(parents=True, exist_ok=True)
    return path


def _resolve_inside(workspace: Path, relative: str) -> Path:
    """Resolve a path and refuse anything that escapes the workspace.

    Uses fully-resolved paths so ``../``, symlinks and absolute paths are all
    caught by the same check rather than by string matching.
    """
    if not relative or not str(relative).strip():
        raise WorkspaceError("no path given")
    root = workspace.resolve()
    requested = Path(str(relative).strip())
    if requested.is_absolute():
        raise WorkspaceError(
            f"path '{relative}' is outside the workspace and was refused")

    # Resolve the lexical path without asking Windows to resolve a
    # simultaneously-created parent directory. Path.resolve() can briefly
    # return an unrelated path when parallel workers create the same new
    # directory, which caused valid writes to fail and then pass on retry.
    candidate = Path(os.path.abspath(root / requested))
    try:
        inside_lexically = os.path.commonpath(
            [os.path.normcase(str(root)), os.path.normcase(str(candidate))]
        ) == os.path.normcase(str(root))
    except ValueError:  # different Windows drives
        inside_lexically = False
    if not inside_lexically:
        raise WorkspaceError(
            f"path '{relative}' is outside the workspace and was refused")

    # Preserve the symlink defence: the nearest existing ancestor must also
    # resolve inside the workspace. This works for both new files and reads.
    ancestor = candidate if candidate.exists() else candidate.parent
    while not ancestor.exists() and ancestor != root:
        ancestor = ancestor.parent
    resolved_ancestor = ancestor.resolve()
    if resolved_ancestor != root and root not in resolved_ancestor.parents:
        raise WorkspaceError(
            f"path '{relative}' is outside the workspace and was refused")
    return candidate


def check_command(command: str) -> Optional[str]:
    """Return why a command is refused, or None if it is allowed."""
    if not command or not command.strip():
        return "empty command"

    for pattern, reason in BLOCKED_PATTERNS:
        if pattern.search(command):
            return f"refused: this looks like {reason}"

    # Split on shell statement separators, then check the leading word of each.
    for statement in re.split(r"(?:\|\||&&|[;|&\n])", command):
        statement = statement.strip()
        if not statement:
            continue
        try:
            parts = shlex.split(statement, posix=False)
        except ValueError:
            parts = statement.split()
        if not parts:
            continue
        head = Path(parts[0].strip("'\"")).name.lower()
        head = head[:-4] if head.endswith(".exe") else head
        if head in BLOCKED_COMMANDS:
            return (f"refused: '{head}' is not allowed -- it can change system state "
                    "or escalate privilege")
    return None


def safe_environment() -> Dict[str, str]:
    """A copy of the environment with secrets stripped out."""
    return {k: v for k, v in os.environ.items() if not SECRET_ENV_PATTERN.search(k)}


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


class WorkspaceFileTool(Tool):
    """Read and write files inside one run's workspace.

    This is what lets a team of agents hand real code to each other: the
    backend agent writes ``api/routes.py``, the test agent reads it back.
    """

    name = "workspace"
    capability = "filesystem"
    description = "Create, read and list files in this run's workspace directory"
    fallbacks: List[str] = []
    side_effect = True
    irreversible = False  # confined to a scratch directory we own

    def __init__(self, run_id: str = "default", root: Optional[Path] = None):
        super().__init__()
        self.workspace = workspace_for(run_id, root)

    def is_live(self) -> bool:
        return True

    def prompt_hint(self) -> str:
        return (
            "Write files other agents can read back. End your response with:\n"
            'TOOL_DIRECTIVE: {"action": "write", "path": "src/app.py", "content": "..."}\n'
            'Other actions: {"action": "read", "path": "..."}, {"action": "list"}.\n'
            "Paths are relative to the workspace; you cannot write outside it.")

    def _run(self, task: str, context: Optional[dict] = None) -> str:
        directive = _directive(task)
        action = str(directive.get("action", "write")).lower()

        if action == "list":
            files = sorted(
                str(p.relative_to(self.workspace))
                for p in self.workspace.rglob("*") if p.is_file())
            return (f"[workspace] {len(files)} file(s): " + ", ".join(files[:40])
                    if files else "[workspace] empty")

        # Validate the action before the path, or an unknown action reports the
        # confusing "no path given" instead of naming the actual mistake.
        if action not in {"read", "write"}:
            raise WorkspaceError(
                f"unknown workspace action '{action}'; use write, read or list")

        path = _resolve_inside(self.workspace, directive.get("path", ""))

        if action == "read":
            if not path.is_file():
                raise WorkspaceError(f"no such file: {directive.get('path')}")
            content = path.read_text(encoding="utf-8", errors="replace")
            return f"[workspace] {directive['path']} ({len(content)} chars):\n{content[:4000]}"

        if action == "write":
            content = directive.get("content")
            if content is None:
                # No explicit content: fall back to the agent's own output,
                # minus the directive line, so a plain "write this file" works.
                content = re.sub(r"TOOL_DIRECTIVE:.*$", "", task,
                                 flags=re.MULTILINE | re.DOTALL).strip()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(str(content), encoding="utf-8")
            return (f"[workspace] wrote {path.relative_to(self.workspace)} "
                    f"({len(str(content))} chars)")



class TerminalTool(Tool):
    """Run shell commands inside the workspace, with guardrails.

    Opt-in: not registered by default, and disabled unless
    ``ORCHESTRATOR_ALLOW_TERMINAL=1``. Every command is checked against a
    blocklist, run with the workspace as its working directory, denied the
    operator's secrets, and killed at a timeout.
    """

    name = "terminal"
    capability = "shell"
    description = "Run a shell command in this run's workspace (no admin, no host changes)"
    fallbacks: List[str] = []
    side_effect = True
    #: Running a command can install packages or mutate files -- always worth
    #: a human's eye before the first one goes through.
    irreversible = True

    def __init__(self, run_id: str = "default", root: Optional[Path] = None,
                 timeout_s: float = 120.0):
        super().__init__()
        self.workspace = workspace_for(run_id, root)
        self.timeout_s = timeout_s

    def is_live(self) -> bool:
        return os.environ.get("ORCHESTRATOR_ALLOW_TERMINAL", "").strip().lower() in {
            "1", "true", "yes", "on"}

    def prompt_hint(self) -> str:
        return (
            "Run a command in the workspace. End your response with:\n"
            'TOOL_DIRECTIVE: {"command": "npm install && npm test"}\n'
            "It runs in the workspace directory only. Commands that need admin "
            "rights, change system settings, or touch anything outside the "
            "workspace are refused.")

    def preview(self, task: str, context: Optional[dict] = None) -> str:
        command = _directive(task).get("command", "(no command)")
        mode = "REAL" if self.is_live() else "simulated"
        return f"terminal [{mode}] in {self.workspace.name}: {command}"

    def _run(self, task: str, context: Optional[dict] = None) -> str:
        command = str(_directive(task).get("command", "")).strip()
        if not command:
            return "[terminal] no 'command' in TOOL_DIRECTIVE; nothing run"

        refusal = check_command(command)
        if refusal:
            raise WorkspaceError(f"{refusal}\n  command was: {command}")

        if not self.is_live():
            return (f"[simulated:terminal] would run: {command}\n"
                    "  (set ORCHESTRATOR_ALLOW_TERMINAL=1 to actually execute commands)")

        try:
            proc = subprocess.run(
                command, shell=True, cwd=str(self.workspace),
                capture_output=True, text=True, timeout=self.timeout_s,
                env=safe_environment(), check=False,
            )
        except subprocess.TimeoutExpired:
            raise WorkspaceError(
                f"command exceeded {self.timeout_s}s and was killed: {command}") from None
        except Exception as exc:  # noqa: BLE001
            raise WorkspaceError(f"could not run command: {exc}") from exc

        output = (proc.stdout or "") + (("\n" + proc.stderr) if proc.stderr else "")
        verdict = "ok" if proc.returncode == 0 else f"exit {proc.returncode}"
        return f"[terminal] {verdict}: {command}\n{output.strip()[:3000]}"


def coding_team_tools(run_id: str = "default", root: Optional[Path] = None) -> List[Tool]:
    """The tools a team of agents needs to actually build and test software."""
    return [WorkspaceFileTool(run_id, root), TerminalTool(run_id, root)]
