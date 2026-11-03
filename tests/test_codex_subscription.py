"""ChatGPT-plan Codex provider, without real model calls."""

from __future__ import annotations

import subprocess
import os

import pytest

import orchestrator.llm as llm


@pytest.mark.skipif(os.name != "nt", reason="Windows Codex desktop installation")
def test_codex_provider_finds_desktop_cli_without_path(monkeypatch, tmp_path):
    executable = tmp_path / "OpenAI" / "Codex" / "bin" / "current" / "codex.exe"
    executable.parent.mkdir(parents=True)
    executable.write_bytes(b"test")
    monkeypatch.delenv("CODEX_EXECUTABLE", raising=False)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    monkeypatch.setattr(llm.shutil, "which", lambda name: None)

    assert llm.CodexSubscriptionProvider().executable == str(executable)


def test_codex_provider_uses_signed_in_cli_not_api_key(monkeypatch):
    monkeypatch.setattr(llm.shutil, "which", lambda name: "codex-test")
    monkeypatch.setenv("OPENAI_API_KEY", "together-secret")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.together.xyz/v1")
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, '{"ok":true}\n', "")

    monkeypatch.setattr(llm.subprocess, "run", run)
    response = llm.CodexSubscriptionProvider(model="gpt-test").generate(
        "Return a JSON object", system="You are concise", json_mode=True)
    command, kwargs = calls[0]
    assert response.text == '{"ok":true}'
    assert response.model == "gpt-test"
    assert response.usage.cost_usd == 0
    assert "--sandbox" in command and "read-only" in command
    assert "--ephemeral" in command
    assert kwargs["env"].get("OPENAI_API_KEY") is None
    assert kwargs["env"].get("OPENAI_BASE_URL") is None
    assert "Return valid JSON only" in kwargs["input"]
    assert "You are concise" in kwargs["input"]


def test_codex_provider_accepts_image_and_cleans_it_up(monkeypatch):
    monkeypatch.setattr(llm.shutil, "which", lambda name: "codex-test")
    image_paths = []

    def run(command, **kwargs):
        image_paths.append(command[command.index("--image") + 1])
        assert open(image_paths[-1], "rb").read() == b"fake-png"
        return subprocess.CompletedProcess(command, 0, "saw image\n", "")

    monkeypatch.setattr(llm.subprocess, "run", run)
    provider = llm.CodexSubscriptionProvider(model="gpt-test")
    assert provider.supports_images()
    assert provider.generate("What is visible?", images=[b"fake-png"]).text == "saw image"
    from pathlib import Path
    assert not Path(image_paths[0]).exists()


def test_codex_provider_errors_are_not_silent(monkeypatch):
    monkeypatch.setattr(llm.shutil, "which", lambda name: "codex-test")
    monkeypatch.setattr(llm.subprocess, "run", lambda command, **kwargs:
                        subprocess.CompletedProcess(command, 1, "", "Not logged in"))
    with pytest.raises(llm.LLMError, match="Not logged in"):
        llm.CodexSubscriptionProvider().generate("hello")
