"""Hermes Agent adapter for bounded, high-level desktop work.

Planning, approval, persistence and retries stay in our orchestrator. Hermes
is invoked as one worker through its official one-shot CLI interface.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import List, Optional

from .base import Tool, ToolError
from .workspace import DEFAULT_WORKSPACE_ROOT, safe_environment, workspace_for


def _enabled() -> bool:
    return os.environ.get("ORCHESTRATOR_ALLOW_DESKTOP", "").strip().lower() in {
        "1", "true", "yes", "on"
    }


class HermesDesktopTool(Tool):
    """Delegate one desktop outcome to a locally installed Hermes Agent."""

    name = "hermes_desktop"
    capability = "desktop_automation"
    description = (
        "Complete a high-level task in Windows desktop applications using a "
        "locally installed Hermes Agent"
    )
    fallbacks: List[str] = []
    side_effect = True
    irreversible = True

    def __init__(self, run_id: str = "default", root: Optional[Path] = None,
                 timeout_s: float = 600.0, executable: Optional[str] = None):
        super().__init__()
        self.workspace = workspace_for(run_id, root or DEFAULT_WORKSPACE_ROOT)
        self.timeout_s = timeout_s
        self.executable = executable or os.environ.get("HERMES_EXECUTABLE", "hermes")

    def _binary(self) -> Optional[str]:
        return shutil.which(self.executable)

    def is_live(self) -> bool:
        return _enabled() and self._binary() is not None

    def prompt_hint(self) -> str:
        return (
            "Delegate a complete desktop outcome to Hermes. Describe the desired "
            "result, applications, supplied inputs and acceptance checks. Do not "
            "include passwords or API keys. Human approval is required before it starts."
        )

    def preview(self, task: str, context: Optional[dict] = None) -> str:
        first = (task.strip().splitlines() or ["(empty task)"])[0][:140]
        mode = "REAL" if self.is_live() else "simulated"
        return f"hermes_desktop [{mode}]: {first}"

    def _run(self, task: str, context: Optional[dict] = None) -> str:
        objective = task.strip()
        if not objective:
            raise ToolError("Hermes desktop task is empty")

        binary = self._binary()
        if not _enabled():
            return (
                "[simulated:hermes_desktop] desktop task was not executed; set "
                "ORCHESTRATOR_ALLOW_DESKTOP=1 after reviewing the requested action"
            )
        if binary is None:
            return (
                "[simulated:hermes_desktop] Hermes is not installed or is not on PATH; "
                "install Hermes Agent and run 'hermes computer-use install'"
            )

        prompt = (
            "You are a desktop worker controlled by another orchestrator. Complete only "
            "the objective below. No administrator privileges are allowed; do not request "
            "or use elevation. Do not "
            "reveal credentials. Do not send, publish, purchase, delete, submit, or modify "
            "external records unless the objective explicitly asks for that exact action. "
            "Work only in the supplied workspace when creating files. Return a concise "
            "result with actions performed, evidence, files created, and anything that "
            f"remains incomplete.\n\nOBJECTIVE:\n{objective}"
        )
        env = safe_environment()
        env["HERMES_ENVIRONMENT_HINT"] = (
            f"Orchestrator workspace: {self.workspace}. No administrator privileges."
        )

        try:
            process = subprocess.run(
                [binary, "-z", prompt], cwd=str(self.workspace),
                capture_output=True, text=True, timeout=self.timeout_s,
                env=env, check=False,
            )
        except subprocess.TimeoutExpired:
            raise ToolError(
                f"Hermes desktop task exceeded {self.timeout_s:.0f} seconds and was stopped"
            ) from None
        except OSError as exc:
            raise ToolError(f"could not start Hermes Agent: {exc}") from exc

        output = (process.stdout or "").strip()
        error = (process.stderr or "").strip()
        if process.returncode != 0:
            detail = error or output or "no diagnostic output"
            raise ToolError(
                f"Hermes Agent exited with code {process.returncode}: {detail[:1200]}"
            )
        if not output:
            raise ToolError("Hermes Agent completed without returning a result")
        return f"[hermes_desktop] completed\n{output[:12000]}"
