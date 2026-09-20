"""An agentic shell loop: the way Claude Code and Codex actually work.

The orchestrator runs one model call per step, feeds its output to one tool,
and moves on. The agent never sees what the tool returned, so it cannot
react: it has to get the whole thing right in a single guess, with no chance
to read an error and try something else.

That single-shot shape, not the absence of GUI control, is the real
difference from Claude Code. Claude Code barely touches the GUI. It drives
the machine through the shell and the filesystem, in a loop:

    run a command -> read the output -> decide the next one -> repeat

Which happens to answer both of the hard questions at once:

* **Background.** A shell command needs no window and no focus. It runs
  while you carry on using the machine -- no raising windows, no fighting
  over which app is in front.
* **App-independence.** Anything with a command line is reachable. No
  accessibility tree to expose, no pixels to interpret, nothing to break
  when a button moves.

What it cannot do is operate software that has no interface but its GUI.
That is what ``computer_use`` and its browser and desktop backends are for.
Between them: shell for most work, GUI for the rest.
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
from .workspace import (DEFAULT_WORKSPACE_ROOT, check_command, safe_environment,
                        workspace_for)

__all__ = ["ComputerTaskTool"]

_DEFAULT_MAX_STEPS = 12
_DEFAULT_TIME_LIMIT_S = 240.0
#: Enough for a directory listing or a stack trace; past that the model is
#: reading noise and paying for it on every later turn.
_MAX_OUTPUT_CHARS = 4000

_SYSTEM = (
    "You accomplish a goal on a computer by running shell commands, one at a "
    "time, reading each result before choosing the next.\n\n"
    "Reply with JSON only:\n"
    '  {"command": "<one shell command>", "why": "<short reason>"}\n'
    '  {"done": true, "answer": "<what you found or did>"}\n\n'
    "Rules:\n"
    "- One command per reply. Read its output before deciding the next.\n"
    "- Prefer reading over writing until you know what is there.\n"
    "- Commands run in a fixed working directory; use relative paths.\n"
    "- Anything needing admin rights, changing system settings, or reaching "
    "outside the working directory will be refused -- do not try.\n"
    "- Say done as soon as the goal is met. Do not pad the transcript."
)


class ComputerTaskTool(Tool):
    """Pursue a goal through a bounded loop of shell commands."""

    name = "computer_task"
    capability = "computer_task"
    description = ("Accomplish a goal on the computer by running shell commands "
                   "in a loop, reading each result before the next")
    fallbacks: List[str] = []
    side_effect = True
    #: Commands change files and start programs, so a person sees the goal and
    #: the limits before the loop starts.
    irreversible = True

    def __init__(self, run_id: str = "default", root: Optional[Path] = None,
                 llm: Any = None, max_steps: int = _DEFAULT_MAX_STEPS,
                 time_limit_s: float = _DEFAULT_TIME_LIMIT_S) -> None:
        super().__init__()
        self.run_id = run_id
        self.workspace = workspace_for(run_id, root or DEFAULT_WORKSPACE_ROOT)
        self.llm = llm
        self.max_steps = max_steps
        self.time_limit_s = time_limit_s

    def is_live(self) -> bool:
        return bool(self.llm) and os.environ.get(
            "ORCHESTRATOR_ALLOW_TERMINAL", "").strip().lower() in {"1", "true", "yes", "on"}

    def prompt_hint(self) -> str:
        return (
            "Give a goal to accomplish on this computer; the tool works out the "
            "commands itself, one at a time, reading each result.\n"
            'TOOL_DIRECTIVE: {"arguments": {"goal": "count the Python files and '
            'report the largest one"}}\n'
            f"It runs in {self.workspace.name}/, needs no window or focus, and "
            f"stops after {self.max_steps} commands or {self.time_limit_s:.0f}s.")

    def preview(self, task: str, context: Optional[dict] = None) -> str:
        goal = self._goal(task)
        mode = "REAL" if self.is_live() else "simulated"
        return (f"computer_task [{mode}]: {goal[:160]}\n"
                f"  in {self.workspace}  |  up to {self.max_steps} commands, "
                f"{self.time_limit_s:.0f}s  |  no window needed")

    # -- internals ---------------------------------------------------------

    @staticmethod
    def _goal(task: str) -> str:
        match = re.search(r"TOOL_DIRECTIVE:\s*(\{.*\})\s*$", task or "", re.S | re.M)
        if match:
            try:
                parsed = json.loads(match.group(1))
                args = parsed.get("arguments", parsed) if isinstance(parsed, dict) else {}
                if isinstance(args, dict) and args.get("goal"):
                    return str(args["goal"])
            except ValueError:
                pass
        # No directive: the step's own words are the goal.
        return re.split(r"^TOOL_DIRECTIVE:", task or "", flags=re.M)[0].strip()

    def _log(self, entry: Dict[str, Any]) -> None:
        try:
            self.workspace.mkdir(parents=True, exist_ok=True)
            with (self.workspace / "computer_task.log.jsonl").open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({"at": round(time.time(), 3), **entry}) + "\n")
        except OSError:
            pass

    def _run_command(self, command: str) -> str:
        """Run one command under the same guards as the terminal tool."""
        refusal = check_command(command)
        if refusal:
            return f"REFUSED: {refusal}"
        try:
            proc = subprocess.run(
                command, shell=True, cwd=str(self.workspace),
                capture_output=True, text=True, timeout=60,
                env=safe_environment(), check=False)
        except subprocess.TimeoutExpired:
            return "REFUSED: command exceeded 60s and was stopped"
        except Exception as exc:  # noqa: BLE001
            return f"REFUSED: could not run it ({exc})"
        body = (proc.stdout or "") + (("\n" + proc.stderr) if proc.stderr else "")
        body = body.strip()[:_MAX_OUTPUT_CHARS] or "(no output)"
        return f"exit {proc.returncode}\n{body}"

    @staticmethod
    def _decision(text: str) -> Dict[str, Any]:
        match = re.search(r"\{.*\}", text or "", re.S)
        if not match:
            return {}
        try:
            parsed = json.loads(match.group(0))
        except ValueError:
            return {}
        return parsed if isinstance(parsed, dict) else {}

    # -- the loop ----------------------------------------------------------

    def _run(self, task: str, context: Optional[dict] = None) -> str:
        goal = self._goal(task)
        if not goal:
            raise ToolError('refused: no goal given, e.g. {"goal": "list the tests"}')
        if self.llm is None:
            raise ToolError("no model is attached, so there is nothing to drive the loop")
        if not self.is_live():
            return (f"[simulated:computer_task] would pursue {goal[:120]!r} in "
                    f"{self.workspace.name}/\n"
                    "  (set ORCHESTRATOR_ALLOW_TERMINAL=1 to run commands for real)")

        self.workspace.mkdir(parents=True, exist_ok=True)
        self._log({"event": "start", "goal": goal})
        started = time.monotonic()
        transcript: List[str] = []
        history: List[Dict[str, str]] = []
        answer = ""

        for turn in range(self.max_steps):
            elapsed = time.monotonic() - started
            if elapsed >= self.time_limit_s:
                transcript.append(f"stopped: reached the {self.time_limit_s:.0f}s limit")
                break

            context_lines = "\n\n".join(
                f"$ {h['command']}\n{h['output']}" for h in history[-6:]
            ) or "(nothing run yet)"
            prompt = (f"Goal: {goal}\n\nWorking directory: {self.workspace}\n"
                      f"Commands run so far ({len(history)}):\n{context_lines}\n\n"
                      f"You have {self.max_steps - turn} command(s) left. "
                      "Next command, or done.")
            try:
                response = self.llm.generate(prompt, system=_SYSTEM, json_mode=True,
                                             metadata={"role": "computer_operator"})
            except Exception as exc:  # noqa: BLE001
                transcript.append(f"stopped: the model could not be reached ({exc})")
                break

            decision = self._decision(response.text)
            if decision.get("done"):
                answer = str(decision.get("answer", "")).strip()
                break

            command = str(decision.get("command", "")).strip()
            if not command:
                # One bad turn should not end the run: models drop out of JSON
                # occasionally, and the next turn usually recovers. Say what
                # came back either way -- "no command" with no evidence is not
                # something anyone can debug from a transcript.
                snippet = (response.text or "").strip()[:200] or "(empty response)"
                transcript.append(f"(unusable reply, retrying) {snippet}")
                history.append({"command": "(none)",
                                "output": "Your last reply had no command and was "
                                          "not marked done. Reply with JSON only."})
                continue

            output = self._run_command(command)
            history.append({"command": command, "output": output})
            self._log({"event": "command", "command": command, "output": output[:600]})
            transcript.append(f"$ {command}\n{output}")
        else:
            transcript.append(f"stopped: reached the {self.max_steps}-command limit")

        self._log({"event": "finish", "commands": len(history),
                   "seconds": round(time.monotonic() - started, 2), "answer": answer[:300]})

        body = "\n\n".join(transcript) or "(nothing run)"
        head = (f"[computer_task] {len(history)} command(s) in "
                f"{time.monotonic() - started:.1f}s")
        return f"{head}\n\n{body}" + (f"\n\nRESULT: {answer}" if answer else "")
