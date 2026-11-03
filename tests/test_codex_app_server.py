"""Codex app-server turns keep a thread across separate subprocesses."""

import json
import queue

from orchestrator.tools import codex_app_server


def test_app_server_streams_and_resumes_a_thread(tmp_path, monkeypatch):
    methods = []
    progress = []

    class FakeProcess:
        def __init__(self, *args, **kwargs):
            self.lines = queue.Queue()
            self.stdin = self
            self.stdout = self._stdout()
            self.returncode = None

        def _stdout(self):
            while True:
                line = self.lines.get()
                if line is None:
                    return
                yield line

        def write(self, line):
            data = json.loads(line)
            method = data["method"]
            methods.append(method)
            ident = data.get("id")

            def push(message):
                self.lines.put(json.dumps(message) + "\n")

            if method == "initialize":
                push({"id": ident, "result": {}})
            elif method in {"thread/start", "thread/resume"}:
                push({"id": ident, "result": {"thread": {"id": "saved-thread"}}})
            elif method == "turn/start":
                push({"id": ident, "result": {"turn": {"id": "turn-1"}}})
                push({"method": "item/agentMessage/delta", "params": {
                    "itemId": "message-1", "delta": "Done", "threadId": "saved-thread", "turnId": "turn-1"}})
                push({"method": "turn/completed", "params": {
                    "threadId": "saved-thread", "turn": {"id": "turn-1", "status": "completed", "items": []}}})
            return len(line)

        def flush(self):
            pass

        def poll(self):
            return self.returncode

        def terminate(self):
            self.returncode = 0
            self.lines.put(None)

        def wait(self, timeout=None):
            return 0

    monkeypatch.setattr(codex_app_server.subprocess, "Popen", FakeProcess)
    for _ in range(2):
        assert codex_app_server.run_turn(
            "codex-test", tmp_path, "test-model", "Create something", {}, 10,
            progress.append) == "Done"
    assert methods.count("thread/start") == 1
    assert methods.count("thread/resume") == 1
    assert methods.count("turn/start") == 2
    assert "saved-thread" in (tmp_path / "codex_agent.thread.json").read_text()
    assert any("Codex: Done" in message for message in progress)
