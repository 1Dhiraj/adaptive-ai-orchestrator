"""Run the signed-in Codex agent harness inside a single run workspace.

The ordinary Codex LLM provider is deliberately read-only and returns text.
This tool gives a software step Codex's own file and shell loop instead.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Optional

from ..llm import CodexSubscriptionProvider, LLMError
from .base import Tool, ToolError, extract_directive
from .workspace import DEFAULT_WORKSPACE_ROOT, workspace_for


class CodexAgentTool(Tool):
    name = "codex_agent"
    capability = "codex_agent"
    description = ("Use the signed-in Codex agent to implement, edit, debug, "
                   "build, and test software in this run's workspace. "
                   'Give a TOOL_DIRECTIVE with {"arguments":{"goal":"the complete task"}}.')
    side_effect = True
    irreversible = True  # Codex may run commands; the dashboard reviews the task first.

    def __init__(self, run_id: str = "default", root: Optional[Path] = None) -> None:
        super().__init__()
        self.workspace = workspace_for(run_id, root or DEFAULT_WORKSPACE_ROOT)
        self.progress_callback = None

    def is_live(self) -> bool:
        try:
            CodexSubscriptionProvider()
            return True
        except LLMError:
            return False

    def availability_detail(self) -> str:
        return ("signed-in Codex CLI available" if self.is_live()
                else "Codex CLI unavailable; sign in with codex login")

    def preview(self, task: str, context: Optional[dict] = None) -> str:
        goal = self._goal(task, context)
        return ("codex_agent [REAL] in " + str(self.workspace) + "\n"
                "Codex may edit files and run commands inside this workspace.\n"
                "Task: " + goal[:500])

    @staticmethod
    def _goal(task: str, context: Optional[dict]) -> str:
        directive = extract_directive(task)
        goal = str(directive.get("goal") or "").strip()
        description = str((context or {}).get("step_description") or "").strip()
        if not goal:
            goal = description or re.sub(r"(?s)TOOL_DIRECTIVE:.*$", "", task).strip()
        upstream = str((context or {}).get("upstream_context") or "").strip()
        if upstream:
            goal += "\n\nResults from prerequisite steps:\n" + upstream[:8000]
        inputs = (context or {}).get("inputs") or {}
        if inputs:
            goal += "\n\nOperator-provided values:\n" + json.dumps(inputs, ensure_ascii=False)
        return goal.strip()

    def _files(self) -> Dict[str, tuple[int, int]]:
        excluded = {"codex_agent.events.jsonl", "codex_agent.last_message.txt",
                    "codex_agent.thread.json", "codex_agent.stderr.log"}
        return {str(path.relative_to(self.workspace)).replace("\\", "/"):
                (path.stat().st_size, path.stat().st_mtime_ns)
                for path in self.workspace.rglob("*")
                if path.is_file() and path.name not in excluded}

    def _run(self, task: str, context: Optional[dict] = None) -> str:
        goal = self._goal(task, context)
        if not goal:
            raise ToolError("Codex agent received no task")
        try:
            provider = CodexSubscriptionProvider()
        except LLMError as exc:
            raise ToolError(str(exc)) from exc
        before = self._files()
        last_message = self.workspace / "codex_agent.last_message.txt"
        events_path = self.workspace / "codex_agent.events.jsonl"
        if last_message.exists():
            last_message.unlink()
        timeout = min(max(int(os.environ.get("CODEX_AGENT_TIMEOUT_S", "600")), 30), 1800)
        prompt = (
            "Complete this software task in the current workspace. You may inspect files, "
            "edit code, and run relevant build or test commands. Keep all new or edited files "
            "inside this workspace. Verify the result before saying it is done. "
            "Report actual file paths and test results; never invent them.\n\n" + goal)
        model = os.environ.get("CODEX_AGENT_MODEL", "").strip() or provider.model
        child_env = os.environ.copy()
        for key in list(child_env):
            if key in {"OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_MODEL"} or re.search(
                    r"(?:KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL)", key, re.I):
                child_env.pop(key, None)
        started = time.monotonic()
        backend = os.environ.get("CODEX_AGENT_BACKEND", "exec").strip().lower()
        if backend == "app-server":
            from .codex_app_server import run_turn

            def progress(message: str) -> None:
                if self.progress_callback:
                    self.progress_callback(str((context or {}).get("step_id") or ""), message)

            message = run_turn(provider.executable, self.workspace, model, prompt,
                               child_env, timeout, progress)
        elif backend == "exec":
            command = [provider.executable, "exec", "--sandbox", "workspace-write",
                       "--skip-git-repo-check", "--json", "-C", str(self.workspace),
                       "--model", model, "--output-last-message", str(last_message), "-"]
            try:
                result = subprocess.run(command, input=prompt, text=True,
                                        capture_output=True, encoding="utf-8", errors="replace",
                                        env=child_env, timeout=timeout, check=False)
            except subprocess.TimeoutExpired as exc:
                raise ToolError(f"Codex agent exceeded the {timeout}s time limit") from exc
            except OSError as exc:
                raise ToolError(f"Could not start Codex agent: {exc}") from exc
            events_path.write_text(result.stdout[:2_000_000], encoding="utf-8")
            events = []
            for line in result.stdout.splitlines():
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
            failure = next((event for event in reversed(events)
                            if event.get("type") in {"turn.failed", "error"}), None)
            if failure:
                error = failure.get("error") or failure.get("message") or "unknown error"
                if isinstance(error, dict):
                    error = error.get("message") or str(error)
                raise ToolError(f"Codex agent failed with {model}: {str(error)[:500]}")
            if result.returncode != 0:
                detail = (result.stderr.strip().splitlines() or ["unknown error"])[-1]
                raise ToolError(f"Codex agent failed (exit {result.returncode}, {model}): {detail[:500]}")
            message = last_message.read_text(encoding="utf-8").strip() if last_message.is_file() else ""
        else:
            raise ToolError(f"Unknown CODEX_AGENT_BACKEND {backend!r}")
        if not message:
            raise ToolError("Codex agent exited without a final response")
        after = self._files()
        changed = sorted(name for name, state in after.items() if before.get(name) != state)
        needs_file = bool(re.search(r"\b(?:create|write|edit|modify|implement|develop|build|fix)\b", goal, re.I))
        if needs_file and not changed:
            raise ToolError("Codex agent reported completion but changed no workspace files")
        return (f"[codex_agent] completed in {time.monotonic() - started:.1f}s\n"
                "Files changed: " + (", ".join(changed[:30]) or "none") + "\n\n" + message[:8000])
