"""Small stdio JSON-RPC client for persistent, signed-in Codex turns."""

from __future__ import annotations

import json
import queue
import subprocess
import threading
import time
from pathlib import Path
from typing import Callable, Optional

from .base import ToolError


def run_turn(executable: str, workspace: Path, model: str, prompt: str,
             child_env: dict[str, str], timeout: int,
             progress: Optional[Callable[[str], None]] = None) -> str:
    """Run one Codex turn, resuming this workspace's thread when it exists."""
    thread_file = workspace / "codex_agent.thread.json"
    event_file = workspace / "codex_agent.events.jsonl"
    stderr_file = workspace / "codex_agent.stderr.log"
    inbox: queue.Queue[dict] = queue.Queue()
    messages: dict[str, str] = {}
    message_order: list[str] = []
    pending_text = ""
    next_id = 0

    def emit(message: str) -> None:
        if progress:
            try:
                progress(message[:300])
            except Exception:
                pass  # UI observers cannot interrupt the agent.

    with stderr_file.open("w", encoding="utf-8") as stderr, event_file.open(
            "w", encoding="utf-8") as events:
        try:
            process = subprocess.Popen(
                [executable, "app-server", "--stdio"],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=stderr,
                text=True, encoding="utf-8", errors="replace", bufsize=1,
                env=child_env, cwd=workspace)
        except OSError as exc:
            raise ToolError(f"Could not start Codex app-server: {exc}") from exc

        def read_stdout() -> None:
            assert process.stdout is not None
            for line in process.stdout:
                events.write(line)
                events.flush()
                try:
                    inbox.put(json.loads(line))
                except json.JSONDecodeError:
                    inbox.put({"raw": line[:300]})

        reader = threading.Thread(target=read_stdout, daemon=True)
        reader.start()
        deadline = time.monotonic() + timeout

        def send(payload: dict) -> None:
            assert process.stdin is not None
            process.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
            process.stdin.flush()

        def receive() -> dict:
            nonlocal pending_text
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise ToolError(f"Codex app-server exceeded the {timeout}s time limit")
                try:
                    value = inbox.get(timeout=min(remaining, 1.0))
                except queue.Empty:
                    if process.poll() is not None:
                        raise ToolError(f"Codex app-server exited early ({process.returncode})")
                    continue
                method = str(value.get("method") or "")
                if method == "item/agentMessage/delta":
                    params = value.get("params") or {}
                    item_id = str(params.get("itemId") or "")
                    if item_id not in messages:
                        messages[item_id] = ""
                        message_order.append(item_id)
                    delta = str(params.get("delta") or "")
                    messages[item_id] += delta
                    pending_text += delta
                    if len(pending_text) >= 120 or "\n" in delta:
                        emit("Codex: " + pending_text.strip())
                        pending_text = ""
                elif method in {"item/started", "item/completed"}:
                    item = (value.get("params") or {}).get("item") or {}
                    kind = str(item.get("type") or "")
                    if kind == "agentMessage" and pending_text.strip():
                        emit("Codex: " + pending_text.strip())
                        pending_text = ""
                    if kind in {"commandExecution", "fileChange"}:
                        phase = "started" if method == "item/started" else "finished"
                        emit(f"Codex {kind} {phase}: {item.get('status') or ''}")
                elif method == "turn/completed" and pending_text.strip():
                    emit("Codex: " + pending_text.strip())
                    pending_text = ""
                elif method and "id" in value:
                    raise ToolError(f"Codex requested an additional approval ({method}); "
                                    "this bridge cannot grant it silently")
                return value

        def request(method: str, params: dict) -> dict:
            nonlocal next_id
            next_id += 1
            request_id = next_id
            send({"id": request_id, "method": method, "params": params})
            while True:
                value = receive()
                if value.get("id") != request_id:
                    continue
                if value.get("error"):
                    error = value["error"]
                    detail = error.get("message") if isinstance(error, dict) else str(error)
                    raise ToolError(f"Codex {method} failed: {detail}")
                return value.get("result") or {}

        try:
            request("initialize", {"clientInfo": {
                "name": "adaptive_orchestrator", "title": "Adaptive AI Task Orchestrator",
                "version": "1.0.0"}})
            send({"method": "initialized", "params": {}})
            options = {"cwd": str(workspace), "model": model,
                       "sandbox": "workspace-write", "approvalPolicy": "never"}
            if thread_file.is_file():
                saved = json.loads(thread_file.read_text(encoding="utf-8"))
                thread_id = str(saved.get("thread_id") or "")
                if not thread_id:
                    raise ToolError("Saved Codex thread has no ID")
                request("thread/resume", {**options, "threadId": thread_id})
                emit("Resumed Codex thread")
            else:
                started = request("thread/start", options)
                thread_id = str((started.get("thread") or {}).get("id") or "")
                if not thread_id:
                    raise ToolError("Codex did not return a thread ID")
                emit("Started Codex thread")
            request("turn/start", {"threadId": thread_id,
                                   "input": [{"type": "text", "text": prompt}]})
            while True:
                value = receive()
                if value.get("method") != "turn/completed":
                    continue
                turn = (value.get("params") or {}).get("turn") or {}
                if turn.get("status") != "completed":
                    error = turn.get("error") or {}
                    detail = error.get("message") if isinstance(error, dict) else str(error)
                    raise ToolError(f"Codex turn {turn.get('status')}: {detail or 'no detail'}")
                break
            answer = messages[message_order[-1]].strip() if message_order else ""
            if not answer:
                for item in reversed(turn.get("items") or []):
                    if item.get("type") == "agentMessage":
                        answer = "".join(str(part.get("text") or "")
                                         for part in item.get("content") or []).strip()
                        break
            if not answer:
                raise ToolError("Codex completed without an agent response")
            thread_file.write_text(json.dumps({"thread_id": thread_id, "model": model}),
                                   encoding="utf-8")
            return answer
        finally:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
            reader.join(timeout=2)
